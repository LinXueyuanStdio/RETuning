"""
Summarize stock classification experiment results.

This script collects and aggregates results from all experiments.
"""

import os
import glob
import argparse
import pandas as pd
from pathlib import Path


def parse_metrics_file(filepath):
    """Parse metrics.txt file and extract metrics."""
    metrics = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if ':' in line and not line.startswith('Setting') and not line.startswith('='):
                # Skip lines that are part of classification report
                if 'precision' in line.lower() or 'recall' in line.lower():
                    continue
                if 'support' in line.lower() or 'avg' in line.lower():
                    continue

                parts = line.split(':')
                if len(parts) == 2:
                    key = parts[0].strip()
                    try:
                        value = float(parts[1].strip())
                        metrics[key] = value
                    except ValueError:
                        pass
    return metrics


def extract_setting_info(setting_name):
    """Extract model, mode, and seq_len from setting name."""
    # Format: StockCls_{model}_mode{mode}_sl{seq_len}_dm{d_model}_nh{n_heads}_el{e_layers}_df{d_ff}_{des}_{itr}
    parts = setting_name.split('_')

    info = {
        'setting': setting_name,
        'model': None,
        'mode': None,
        'seq_len': None,
    }

    try:
        # Find model name (after StockCls_)
        if 'StockCls' in setting_name:
            idx = parts.index('StockCls')
            info['model'] = parts[idx + 1]

        # Find mode and seq_len
        for part in parts:
            if part.startswith('mode'):
                info['mode'] = int(part.replace('mode', ''))
            elif part.startswith('sl'):
                info['seq_len'] = int(part.replace('sl', ''))
    except:
        pass

    return info


def collect_results(results_dir='./results'):
    """Collect all results from results directory."""
    all_results = []

    # Find all metrics.txt files
    metrics_files = glob.glob(os.path.join(results_dir, 'StockCls_*', 'metrics.txt'))

    for filepath in metrics_files:
        setting_name = os.path.basename(os.path.dirname(filepath))

        # Extract setting info
        info = extract_setting_info(setting_name)

        # Parse metrics
        metrics = parse_metrics_file(filepath)

        # Combine info and metrics
        result = {**info, **metrics}
        all_results.append(result)

    return pd.DataFrame(all_results)


def main():
    parser = argparse.ArgumentParser(description='Summarize stock classification results')
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='Directory containing results')
    parser.add_argument('--output', type=str, default='./results/summary.csv',
                        help='Output summary CSV file')
    args = parser.parse_args()

    print(f"Collecting results from {args.results_dir}...")

    df = collect_results(args.results_dir)

    if df.empty:
        print("No results found!")
        return

    print(f"Found {len(df)} experiments")

    # Sort by model, mode, seq_len
    df = df.sort_values(['model', 'mode', 'seq_len'])

    # Save full results
    df.to_csv(args.output, index=False)
    print(f"Saved full results to {args.output}")

    # Print summary tables
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)

    # Key metrics
    key_metrics = ['accuracy', 'f1_macro', 'f1_weighted', 'recall_macro']

    for mode in [1, 2]:
        mode_df = df[df['mode'] == mode]
        if mode_df.empty:
            continue

        mode_name = "All History" if mode == 1 else "2024 Only"
        print(f"\n### Mode {mode} ({mode_name}) ###\n")

        # Pivot table: models x seq_len
        for metric in key_metrics:
            if metric not in mode_df.columns:
                continue

            pivot = mode_df.pivot_table(
                values=metric,
                index='model',
                columns='seq_len',
                aggfunc='first'
            )

            print(f"\n{metric.upper()}:")
            print(pivot.round(4).to_string())

    # Best results for each configuration
    print("\n" + "="*80)
    print("BEST RESULTS BY CONFIGURATION")
    print("="*80)

    for mode in [1, 2]:
        mode_name = "All History" if mode == 1 else "2024 Only"
        for seq_len in [5, 10, 20, 60]:
            subset = df[(df['mode'] == mode) & (df['seq_len'] == seq_len)]
            if subset.empty:
                continue

            best_idx = subset['f1_macro'].idxmax()
            best = subset.loc[best_idx]

            print(f"\nMode {mode} ({mode_name}), Seq_len {seq_len}:")
            print(f"  Best Model: {best['model']}")
            print(f"  Accuracy: {best.get('accuracy', 'N/A'):.4f}")
            print(f"  F1 (macro): {best.get('f1_macro', 'N/A'):.4f}")
            print(f"  Recall (macro): {best.get('recall_macro', 'N/A'):.4f}")


if __name__ == "__main__":
    main()
