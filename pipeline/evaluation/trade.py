import argparse
import os
import uuid
from collections import defaultdict
import random
from typing import Dict, Tuple, Optional
from datasets import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import torch
import numpy as np
from xlin import *
import re
import sys

# add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)


from pipeline.build_dataset.inference_engine import build_inference_engine, InferenceEngine

from dotenv import load_dotenv

load_dotenv()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if seed == 0:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False



def save_report(log_rows: list[dict], output_path: str):
    preds, labels = [], []
    for log_row in log_rows:
        preds.append(log_row["pred_status"] if "pred_status" in log_row else "missing")
        labels.append(log_row["gt_status"])
    report_row = generate_classification_report(preds, labels)
    if len(log_rows) > 0:
        for k in log_rows[0]:
            if "score" in k or k in ["response_length", "format_correct", "answer_correct"]:
                report_row[k] = np.mean([log_row[k] for log_row in log_rows])
    jsonable_report_row = convert_to_jsonable_report(report_row)
    append_to_json_list([jsonable_report_row], output_path)
    return report_row


def scoring(response: str):
    if "</think>" in response:
        response = response.rsplit("</think>", 1)[-1]
    check_pass = []
    answer = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if answer:
        answer = answer.group(1)
        if answer in ["up", "down", "hold"]:
            check_pass += [True]
    score = re.search(r"<score>(.*?)</score>", response, re.DOTALL)
    if score:
        score = score.group(1)
        try:
            scores = eval(score)
            if len(scores) == 2 and all(isinstance(i, (int, float)) for i in scores):
                check_pass += [True]
        except Exception as e:
            print(f"Error: {e}")
            pass
    pct_change = re.search(r"<pct_change>(.*?)</pct_change>", response, re.DOTALL)
    if pct_change:
        try:
            pct_change = float(pct_change.group(1))
            check_pass += [True]
        except Exception as e:
            print(f"Error: {e}")
            return 0
        if answer == "up":
            if pct_change > 0.03:
                check_pass += [True]
        elif answer == "down":
            if pct_change < -0.03:
                check_pass += [True]
        elif answer == "hold":
            if -0.03 < pct_change < 0.03:
                check_pass += [True]
    if "支持上涨的因素" in response and "支持下跌的因素" in response:
        check_pass += [True]
    return sum(check_pass) / len(check_pass) if check_pass else 0


def response_to_text(response: str) -> str:
    if isinstance(response, list):
        response = response[0]
    if isinstance(response, dict):
        if "text" in response:
            response = response["text"]
        elif "content" in response:
            response = response["content"]
            if isinstance(response, list):
                response = response_to_text(response)
    if not response:
        return ""
    return response

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
    #     "<think>a<answer>oops</answer></think>x<answer><think>y</think></answer>",  # ✗
    # ]

    # for t in tests:
    #     m = pattern.match(t)
    #     print(t)
    #     print("✓" if m else "✗")
    return True if pattern.match(response_str) else False


def majority_voting(dataset: Dataset, responses: list[list[str]], tokenizer=None):
    if not tokenizer:
        tokenizer_path = "/mnt/model/DeepSeek-R1-14B"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    total_log_rows = []
    best_log_rows = []
    for i, (row, response_list) in tqdm(enumerate(zip(dataset, responses))):
        label = row["label"]
        scores = []
        log_rows = []
        if not isinstance(response_list, list):
            response_list = [response_list]
        response_list = [response_to_text(response) for response in response_list]
        inputs = tokenizer(response_list, add_special_tokens=False)
        lengths = [len(input_ids) for input_ids in inputs["input_ids"]]
        answer2count = defaultdict(int)
        answer2response = defaultdict(list)
        for idx, (response, response_length) in enumerate(zip(response_list, lengths)):
            score = scoring(response) # 在 inference time 只能用【仅面向 response 】的打分函数，避免 ground_truth 泄漏
            generated_answer = extract_answer(response)
            if generated_answer in ["up", "down", "hold"]:
                answer2count[generated_answer] += 1
                answer2response[generated_answer].append(idx)
            log_row = row["extra_info"] | {
                "response_length": response_length,
                "response": response,
                "pred_status": generated_answer,
                "gt_status": label,
                "format_correct": validate_response_structure(response),
                "answer_correct": generated_answer in ["up", "down", "hold"] and generated_answer == label,
                "score": score,
            }
            log_row["best"] = False
            scores.append(score)
            log_rows.append(log_row)
        if len(answer2count) == 0:
            best_response_idx = np.argmax(scores)  # 模型采样出了一堆垃圾，我们只能捏着鼻子用 best_of_N 兜底，选 rule-based score 最高的
        else:
            best_answer = max(answer2count, key=answer2count.get)
            best_response_idx = random.choice(answer2response[best_answer])  # 用 majority_voting 多数投票选出最受欢迎的答案
        best_log_row = log_rows[best_response_idx]
        best_log_row["best"] = True
        total_log_rows.extend(log_rows)
        best_log_rows.append(best_log_row)
    return total_log_rows, best_log_rows


def main(args):
    output_dir = args.save_dir
    os.makedirs(output_dir, exist_ok=True)
    print("Output Folder:", output_dir)

    print(args)

    dataset = Dataset.from_parquet(args.data_path)
    if Path(args.model).exists():
        local = True
        tokenizer_path = args.model
    else:
        local = False
        tokenizer_path = "/mnt/model/DeepSeek-R1-14B"
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Prepare all prompts using multi-threading
    new_prompts = []
    for i, row in enumerate(dataset):
        if args.samples > 0 and len(new_prompts) >= args.samples:
            break
        messages = row["prompt"]
        if not local:
            new_prompts.append(messages)
            continue
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        if isinstance(prompt, list):
            prompt = prompt[0]
        if args.remove_cot:
            prompt = prompt.replace("你心中有一个金融世界，请回忆一下你在金融领域的知识和经验，结合当前的市场环境，给出一个合理的预测。\n\n你需要考虑：做好这个预测任务所需的关键条件；整理自己的状态；明确自己的分析范式；\n为了避免被材料中的情绪误导，请遵循以下思维范式：\n1. 理解所要分析的股票的个性，如蓝筹股、成长股、ST股等，不同类型股票的分析方法不同\n2. 理解所要预测的时间特征，关注节假日、周末、月末等，不同时间特征的分析方法不同\n3. 查看所提供的市场状态包含哪方面的信息，不同的信息覆盖的维度不同，因此分析方法也不同\n4. 初步动态构建分析方法逻辑\n5. 按分析方法逻辑分析各维度信息\n6. 开始整理这些信息，按支持涨的、跌的进行分类，每一类都要对每一条证据进行评分，10分制\n7. 进行假设检验，市场模拟，未来推演，反事实假设等，对这些证据进行反思，直到你确信你已经考虑了所有可能的情况。\n8. 综合平均这些评分，给出最终支持分数。如 <score>[a, b]</score> 表示支持涨的a分，支持跌的b分；a和b的范围在[0, 10]之间\n9. 给出最终的涨跌幅预测和方向预测\n\n", "")
        row["extra_info"]["prompt"] = prompt
        if i < 3:
            print(f"Prompt {i}: {prompt}")
        new_prompts.append(prompt)
    if len(new_prompts) == 0:
        return 0, []

    # Batch inference using vLLM
    if args.exist_response:
        responses = [row[args.exist_response] for row in dataset]
    elif args.model == "random":
        choices = ["up", "hold", "down"]
        responses = []
        for row in dataset:
            predict_label = random.choice(choices)
            if predict_label == "up":
                predict_pct_change = 0.03 + random.uniform(0, 0.1)
            elif predict_label == "hold":
                predict_pct_change = random.uniform(-0.03, 0.03)
            elif predict_label == "down":
                predict_pct_change = -0.03 + random.uniform(-0.1, 0)
            responses.append(f"<pct_change>{predict_pct_change:.4f}</pct_change><answer>{predict_label}</answer>")
    else:
        inference_engine: InferenceEngine = build_inference_engine(
            engine=args.engine,
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        responses = inference_engine.inference(
            new_prompts,
            n=args.n,
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )

    # Process results
    total_log_rows, best_log_rows = majority_voting(dataset, responses, tokenizer=tokenizer)

    save_json_list(total_log_rows, os.path.join(output_dir, f"total_best_of_{args.n}_log_rows.jsonl"))
    save_json_list(best_log_rows, os.path.join(output_dir, f"best_of_{args.n}_log_rows.jsonl"))
    save_report(total_log_rows, os.path.join(output_dir, f"total_best_of_{args.n}_reports.jsonl"))
    save_report(best_log_rows, os.path.join(output_dir, f"best_of_{args.n}_reports.jsonl"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation script for trade dataset")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/evaluation/Fin-2024-December.parquet",
        help="Data file path",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="output",
        help="Save directory",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name or path.",
    )
    parser.add_argument(
        "--engine",
        type=str,
        choices=["vllm", "api", "openai"],
        default="vllm",
        help="Inference engine to use",
    )
    parser.add_argument("--max_tokens", type=int, default=8192, help="Maximum number of tokens")
    parser.add_argument("--max_model_len", type=int, default=64*1024, help="Maximum model length")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95, help="GPU memory utilization for VLLM")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for VLLM")
    parser.add_argument("--temperature", type=float, default=0.6, help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p (nucleus) sampling")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size for VLLM",)
    parser.add_argument("--n", type=str, default="1", help="Number of responses to generate. example: 1,2,4,8,16,32,64")
    parser.add_argument("--timeout", type=int, default=100, help="Timeout for API requests")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--remove_cot", action="store_true", help="remove cot from prompt")
    parser.add_argument("--exist_response", type=str, default=None, help="Existing response column name in data file")
    parser.add_argument("--samples", type=int, default=-1, help="Number of samples to evaluate. -1 means all samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    set_seed(args.seed)
    if "," in args.n:
        n_list = [int(i) for i in args.n.split(",")]
        save_dir = args.save_dir
        for n in n_list:
            args.n = n
            args.save_dir = os.path.join(save_dir, f"n_{n}")
            print(f"Evaluating with n={n}...")
            main(args)
    else:
        args.n = int(args.n)
        main(args)
