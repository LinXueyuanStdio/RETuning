from typing import Dict, Optional
import re
import uuid

# import weave


def extract_response(solution_str: str) -> Optional[str]:
    # Split response to isolate assistant output
    if "Assistant:" in solution_str:
        response_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        response_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        # print("[Error] Failed to locate model response header")
        return solution_str
    return response_str


def extract_answer(response: Optional[str]) -> Optional[str]:
    if not response:
        return None
    if "</think>" in response:
        response = response.rsplit("</think>", 1)[-1]
    # Extract final answer using XML-style tags
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, response, re.DOTALL))

    if not matches:
        # print("[Error] No valid answer tags found")
        return None

    final_answer = matches[-1].group(1).strip()
    return final_answer


def extract_pct_change(response: Optional[str]) -> Optional[int]:
    if not response:
        return None
    if "</think>" in response:
        response = response.rsplit("</think>", 1)[-1]
    pattern = r'\\box\{(.*?)\}'
    matches = list(re.finditer(pattern, response, re.DOTALL))

    if not matches:
        # print("[Error] No valid pct_change tags found")
        return None

    value = matches[-1].group(1).strip()
    try:
        value = float(value)
    except ValueError:
        # print("[Error] Failed to convert extracted value to float")
        return None
    return value

def extract_pct_change_2(response: Optional[str]) -> Optional[int]:
    if not response:
        return None
    if "</think>" in response:
        response = response.rsplit("</think>", 1)[-1]
    pattern = r'<pct_change>(.*?)</pct_change>'
    matches = list(re.finditer(pattern, response, re.DOTALL))

    if not matches:
        # print("[Error] No valid pct_change tags found")
        return None

    value = matches[-1].group(1).strip()
    try:
        value = float(value)
    except ValueError:
        # print("[Error] Failed to convert extracted value to float")
        return None
    return value


def validate_response_structure(response_str: str) -> bool:
    """Performs comprehensive validation of response structure.

    Args:
        response: Processed response string from the model

    Returns:
        Boolean indicating whether all formatting requirements are met
    """
    if not response_str:
        return False
    pattern = re.compile(
        r'^\s*<think>((?:(?!<(?:(?:/?think)|(?:/?answer))>).)*?)'
        r'</think>'
        r'((?:(?!<(?:(?:/?think)|(?:/?answer))>).)*?)'
        r'<answer>((?:(?!<(?:(?:/?think)|(?:/?answer))>).)*?)'
        r'</answer>\s*$',
        re.DOTALL  # 如果需要让 . 匹配换行
    )

    # tests = [
    #     "<think>foo</think>bar<answer>baz</answer>", # ✓
    #     "  <think>思考</think>中间内容<answer>回答</answer>  ", # ✓
    #     "  <think>思考</think><answer>回答</answer>  ", # ✓
    #     "  <think>思考</think>\n<answer>回答</answer>  ", # ✓
    #     "  <think>思考</think> <answer>回答</answer>  ", # ✓
    #     "  <think>思考\n</think> \n<answer>回答\n</answer>  ", # ✓
    #     "<think>a<answer>oops</answer></think>x<answer>y</answer>",  # ✗
    # ]

    # for t in tests:
    #     m = pattern.match(t)
    #     print(t)
    #     print("✓" if m else "✗")
    return True if pattern.match(response_str) else False


def parse_model_answer(answer_text: str):
    # return 'up' or 'down' based on the answer_text
    return answer_text


# @weave.op()
def log_to_weave(
    solution_str: str,
    ground_truth: str,
    change_pct: float,
    data_source: str = 'unknown',
    global_steps: Optional[int] = None,
    response_length: Optional[int] = None,
) -> Dict[str, float]:
    format_reward = 1
    logs = []
    response = solution_str
    log_row = {
        "response": response,
        "change_pct": change_pct,
        "data_source": data_source,
        "ground_truth": ground_truth,
        "global_steps": global_steps,
        "response_length": response_length,
    }
    if not ground_truth:
        # Parse ground truth data
        if change_pct >= 0.03:
            gt_status = "up"
        elif change_pct <= -0.03:
            gt_status = "down"
        else:
            gt_status = "hold"
    else:
        # Use provided label if available
        gt_status = ground_truth
        logs.append(f"[Ground Truth] Using provided label: {gt_status}")
    log_row['gt_status'] = gt_status
    logs.append(f"[Ground Truth] Final change_pct: {change_pct}, Status: {gt_status}")

    # Extract model answer
    pred_status = extract_answer(response)
    # Extract pct_change from response
    predict_change_pct = extract_pct_change(response)
    if not predict_change_pct:
        predict_change_pct = extract_pct_change_2(response)
    if predict_change_pct is None:
        logs.append("[Error] Failed to extract predict_change_pct from response")
    log_row['predict_change_pct'] = predict_change_pct
    # logs.append(f"\n[Model Response]\n{response}")

    # Validate response structure
    format_correct = validate_response_structure(response)
    format_score = format_reward if format_correct else -abs(format_reward)
    log_row['format_correct'] = format_correct
    log_row['format_score'] = format_score
    log_row['format_validation'] = 'PASS' if format_correct else 'FAIL'
    logs.append(f"\n  Format validation: {'PASS' if format_correct else 'FAIL'}")
    logs.append(f"  Format score: {format_score}")

    # Validate answer content
    answer_score = 0
    log_row['pred_status'] = pred_status
    log_row['answer_correct'] = False
    if pred_status in ["up", "down", "hold"] and pred_status == gt_status:
        log_row['answer_correct'] = True
    if pred_status:
        logs.append(f"\n[Content Validation]")
        logs.append(f"  Expected: {gt_status}")
        logs.append(f"  Predicted: {pred_status}")

        if pred_status == gt_status:
            answer_score = 2
            logs.append("  Content validation: FULL MATCH")
            log_row['answer_validation'] = 'FULL MATCH'
        else:
            # 错误补偿：虽然模型预测错了，但是可以原谅。即 change_pct 在 0.03 边缘。我们取暧昧区间为 0.03 +- 0.005
            if abs(change_pct - 0.03) < 0.005 and pred_status in ["up", "hold"]:
                answer_score = 1.8
                logs.append("  Content validation: FUZZY MATCH")
                log_row['answer_validation'] = 'FUZZY MATCH'
            elif abs(change_pct + 0.03) < 0.005 and pred_status in ["down", "hold"]:
                answer_score = 1.8
                logs.append("  Content validation: FUZZY MATCH")
                log_row['answer_validation'] = 'FUZZY MATCH'
            else:
                answer_score = -1.5
                logs.append("  Content validation: MISMATCH")
                log_row['answer_validation'] = 'MISMATCH'

        # 类别补偿：模型一般倾向于 up，为了鼓励模型采样出更多的 down 和 hold 的答案，我们按步数给出补偿
        if pred_status == gt_status:
            if pred_status == "down":
                logs.append("  Content validation: DOWN BONUS")
                if global_steps is not None and global_steps > 100:
                    answer_score += 2
                else:
                    answer_score += 4
            elif pred_status == "hold":
                logs.append("  Content validation: HOLD BONUS")
                if global_steps is not None and global_steps > 100:
                    answer_score += 1
                else:
                    answer_score += 0
            elif pred_status == "up":
                logs.append("  Content validation: UP BONUS")
                if global_steps is not None and global_steps > 100:
                    answer_score += 2
                else:
                    answer_score += 4
    else:
        answer_score = -2
        logs.append("\n[Content Validation] Skipped due to format errors or missing answer")
        log_row['answer_validation'] = 'SKIPPED'
    log_row['answer_score'] = answer_score

    # validate predict_change_pct
    predict_change_pct_score = 0
    if predict_change_pct is not None:
        if change_pct > 0.03 and predict_change_pct < 0:
            # 真正pct超过0.02，模型预测的pct也应该至少为+
            predict_change_pct_score = -2
            logs.append("  Content validation: change_pct MISMATCH")
            log_row["predict_change_pct_validation"] = 'MISMATCH'
        elif change_pct < -0.03 and predict_change_pct > 0:
            # 真正pct超过-0.02，模型预测的pct也应该至少为-
            predict_change_pct_score = -2
            logs.append("  Content validation: change_pct MISMATCH")
            log_row["predict_change_pct_validation"] = 'MISMATCH'
        elif abs(predict_change_pct - change_pct) < 0.01:
            # 模型预测正确，且预测的 pct_change 和 ground truth 的 pct_change 相差不大，这是一个完美的答案
            predict_change_pct_score = 1
            logs.append("  Content validation: change_pct GOLDEN MATCH")
            log_row["predict_change_pct_validation"] = 'GOLDEN MATCH'
        elif abs(predict_change_pct - change_pct) < 0.05:
            # 模型预测正确，但是预测的 pct_change 和 ground truth 的 pct_change 相差较大，可能是模型的预测不准确
            # 鼓励模型进行预测，尽管预测的值不准确
            predict_change_pct_score = 0.5
            logs.append("  Content validation: change_pct BETTER MATCH")
            log_row["predict_change_pct_validation"] = 'BETTER MATCH'
        elif change_pct > 0.03 and predict_change_pct > 0.03:
            # 鼓励往正确方向进行预测
            predict_change_pct_score = 0.3
            logs.append("  Content validation: change_pct PARTIAL MATCH")
            log_row["predict_change_pct_validation"] = 'PARTIAL MATCH'
        elif change_pct < -0.03 and predict_change_pct < -0.03:
            # 鼓励往正确方向进行预测
            predict_change_pct_score = 0.3
            logs.append("  Content validation: change_pct PARTIAL MATCH")
            log_row["predict_change_pct_validation"] = 'PARTIAL MATCH'
        else:
            # 鼓励模型进行预测
            predict_change_pct_score = 0.1
            logs.append("  Content validation: change_pct FULL MATCH")
            log_row['answer_validation'] = 'FULL MATCH'
    else:
        predict_change_pct_score = -2
        logs.append("[Error] Failed to extract predict_change_pct from response")
        log_row["predict_change_pct_validation"] = 'FAIL'
    log_row['predict_change_pct_score'] = predict_change_pct_score

    # validate consistency between answer and predict_change_pct
    consistency_score = 0
    if predict_change_pct is not None:
        if predict_change_pct > 0.03 and pred_status != "up":
            consistency_score = -1
            logs.append("  Content validation: change_pct and answer MISMATCH")
            log_row['consistency_validation'] = 'MISMATCH'
        elif predict_change_pct < -0.03 and pred_status != "down":
            consistency_score = -1
            logs.append("  Content validation: change_pct and answer MISMATCH")
            log_row['consistency_validation'] = 'MISMATCH'
        elif -0.03 < predict_change_pct < 0.03 and pred_status != "hold":
            consistency_score = -1
            logs.append("  Content validation: change_pct and answer MISMATCH")
            log_row['consistency_validation'] = 'MISMATCH'
        else:
            logs.append("  Content validation: change_pct and answer FULL MATCH")
            log_row['consistency_validation'] = 'FULL MATCH'
    else:
        logs.append("[Error] Failed to extract predict_change_pct from response")
        log_row['consistency_validation'] = 'FAIL'
    log_row['consistency_score'] = consistency_score

    response_length_score = 0
    if response_length and pred_status and pred_status == gt_status:
        if response_length > 4000:
            response_length_score = 2
            logs.append(f"  Content validation: response length: {response_length}")
            log_row['length_validation'] = 'EXCELLENT'
        elif response_length > 3000:
            response_length_score = 1.5
            logs.append(f"  Content validation: response length: {response_length}")
            log_row['length_validation'] = 'EXCELLENT'
        elif response_length > 2000:
            response_length_score = 1
            logs.append(f"  Content validation: response length: {response_length}")
            log_row['length_validation'] = 'BETTER'
        elif response_length > 1500:
            response_length_score = 0.5
            logs.append(f"  Content validation: response length: {response_length}")
            log_row['length_validation'] = 'GOOD'
        elif response_length < 1000:
            response_length_score = -1
            logs.append(f"  Content validation: response length: {response_length}")
            log_row['length_validation'] = 'PENALTY'
        elif response_length < 800:
            response_length_score = -2
            logs.append(f"  Content validation: response length: {response_length}")
            log_row['length_validation'] = 'PENALTY'
        elif response_length < 400:
            response_length_score = -4
            logs.append(f"  Content validation: response length: {response_length}")
            log_row['length_validation'] = 'PENALTY'
    log_row['response_length_score'] = response_length_score

    score = format_score + answer_score + predict_change_pct_score + consistency_score + response_length_score
    logs.append("\n" + "-"*80)
    logs.append(f" Final Score ".center(80, '-'))
    logs.append(f"  Format: {format_score}")
    logs.append(f"  Answer: {answer_score}")
    logs.append(f"  Total: {score}")
    logs.append("="*80 + "\n")
    log_row['logs'] = "\n".join(logs)
    log_row['score'] = score  # MUST be set
    top_keys = ["response", "gt_status", "pred_status"]
    new_log_row = {k: log_row[k] for k in top_keys if k in log_row}
    new_log_row.update(log_row)
    return new_log_row


def compute_score(solution_str: str, ground_truth: str, extra_info: Dict[str, str]={}):
    """Computes comprehensive score for model response.

    Args:
        solution_str: Raw model response string
        ground_truth: String containing ground truth data. Choice of 'up', 'down', or 'hold'
        extra_info: Optional additional information for scoring

    Returns:
        Total score (sum of format and answer rewards)
    """
    change_pct = extra_info.get('change_pct', 0.0)
    data_source = extra_info.get('data_source', 'unknown')
    global_steps = extra_info.get('global_steps', 0)
    response_length = extra_info.get('response_length', None)
    # extra_info["prompt_length"] = valid_prompt_length
    # extra_info["response_length"] = valid_response_length
    result = log_to_weave(
        solution_str,
        ground_truth,
        change_pct,
        data_source,
        global_steps,
        response_length,
    )
    score = result['score']

    stock = extra_info.get('stock', None)
    date = extra_info.get('date', None)
    split = extra_info.get('split', 'unknown')
    prompt = extra_info.get('prompt', None)
    result['prompt'] = prompt
    result['split'] = split
    result['stock'] = stock
    result['date'] = date
    result["uuid"] = str(uuid.uuid4())
    return {
        'score': score,
        'log_row': result,
    }

