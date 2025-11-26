"""
Build stock classification dataset for Time-Series-Library.

This script processes raw stock price data and generates:
1. NPZ files with sliding window samples for training
2. CSV metadata files for evaluation output

Label definition:
- 0 (down): overnight_rate < -3%
- 1 (hold): -3% <= overnight_rate <= 3%
- 2 (up): overnight_rate > 3%

Data splits:
- Mode 1: All historical data (before 2024.12) as train, 2024.12 as test
- Mode 2: Only 2024.01-2024.11 as train, 2024.12 as test
"""

import os
import glob
import argparse
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd


def load_price_data(raw_dir: str) -> pd.DataFrame:
    """Load and merge all CSV files from raw directory."""
    files = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
    if not files:
        raise RuntimeError(f"No CSV files found in {raw_dir}")

    dfs = []
    for f in files:
        # Skip files that might not be stock price data
        basename = os.path.basename(f).lower()
        if 'price' in basename:
            continue
        df = pd.read_csv(f)
        dfs.append(df)

    if not dfs:
        raise RuntimeError(f"No valid stock CSV files found in {raw_dir}")

    df = pd.concat(dfs, ignore_index=True)

    # Required columns
    required_cols = ["date", "code", "open", "high", "low", "close"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' in raw data")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["code", "date"]).reset_index(drop=True)

    # Compute pre_close as previous day's close for each stock
    if "pre_close" not in df.columns:
        df["pre_close"] = df.groupby("code")["close"].shift(1)
        # Drop rows without pre_close (first trading day of each stock)
        df = df.dropna(subset=["pre_close"]).reset_index(drop=True)

    return df


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute overnight rate and labels.
    overnight_rate = (open - pre_close) / pre_close
    """
    df = df.copy()
    overnight_ret = (df["open"] - df["pre_close"]) / df["pre_close"]
    cond_up = overnight_ret > 0.03
    cond_down = overnight_ret < -0.03
    labels = np.where(cond_up, 2, np.where(cond_down, 0, 1))
    df["overnight_ret"] = overnight_ret
    df["label"] = labels

    # Compute pct_change (daily return) for output
    df["pct_change"] = (df["close"] - df["pre_close"]) / df["pre_close"]

    return df


def split_train_test(df: pd.DataFrame, mode: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and test sets.

    Mode 1: All historical data before 2024.12 as train
    Mode 2: Only 2024.01-2024.11 as train
    Test: 2024.12
    """
    test_start = datetime(2024, 12, 1)
    test_end = datetime(2024, 12, 31)

    if mode == 1:
        train_mask = df["date"] < test_start
    elif mode == 2:
        train_mask = (df["date"] >= datetime(2024, 1, 1)) & (df["date"] < test_start)
    else:
        raise ValueError("mode must be 1 or 2")

    test_mask = (df["date"] >= test_start) & (df["date"] <= test_end)

    return df[train_mask].reset_index(drop=True), df[test_mask].reset_index(drop=True)


def build_examples(df: pd.DataFrame, seq_len: int, target_df: pd.DataFrame = None) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Build sliding window samples for each stock.

    Uses overnight_ret as the single feature (univariate time series).

    Args:
        df: DataFrame containing all available data for building windows
        seq_len: Length of the lookback window
        target_df: If provided, only build samples where the target date is in this DataFrame.
                   This allows using history from df but predicting only for target_df dates.
                   If None, build samples for all dates in df.

    Returns:
        x: features [N, seq_len, 1] - overnight_ret sequences
        y: labels [N] - current day's label
        meta_df: metadata for each sample
    """
    feature_col = "overnight_ret"

    xs = []
    ys = []
    metas = []

    # If target_df is provided, we need to identify which dates are targets
    if target_df is not None:
        target_dates_by_code = {}
        for code, g in target_df.groupby("code"):
            target_dates_by_code[code] = set(g["date"].values)

    for code, g in df.groupby("code"):
        g = g.sort_values("date").reset_index(drop=True)
        feats = g[feature_col].values.astype("float32")
        labels = g["label"].values.astype("int64")
        dates = g["date"].values
        opens = g["open"].values
        highs = g["high"].values
        lows = g["low"].values
        closes = g["close"].values
        pct_changes = g["pct_change"].values

        total = len(g)
        for i in range(seq_len, total):
            current_date = dates[i]

            # If target_df is specified, only include samples where target date is in target_df
            if target_df is not None:
                if code not in target_dates_by_code:
                    continue
                if current_date not in target_dates_by_code[code]:
                    continue

            # Feature: sequence of overnight rates (univariate)
            xs.append(feats[i - seq_len : i].reshape(-1, 1))
            # Label: current day's label
            ys.append(labels[i])
            # Metadata for evaluation output
            metas.append({
                "date": pd.Timestamp(dates[i]).strftime("%Y-%m-%d"),
                "code": code,
                "open": opens[i],
                "high": highs[i],
                "low": lows[i],
                "close": closes[i],
                "pct_change": pct_changes[i],
                "label": labels[i]
            })

    if not xs:
        return np.array([]).reshape(0, seq_len, 1), np.array([]), pd.DataFrame()

    x = np.stack(xs, axis=0)  # [N, seq_len, 1]
    y = np.array(ys, dtype="int64")  # [N]
    meta_df = pd.DataFrame(metas)

    return x, y, meta_df


def save_npz(out_dir: str, prefix: str, xs: np.ndarray, ys: np.ndarray) -> None:
    """Save dataset as compressed npz file."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{prefix}.npz")
    np.savez_compressed(path, x=xs, y=ys)
    print(f"Saved {path}, x shape={xs.shape}, y shape={ys.shape}")


def main():
    parser = argparse.ArgumentParser(description="Build stock classification dataset")
    parser.add_argument("--raw_dir", type=str, default="../../data/raw",
                        help="Directory containing raw stock CSV files")
    parser.add_argument("--out_root", type=str, default="./dataset",
                        help="Output root directory for processed datasets")
    parser.add_argument("--mode", type=int, choices=[1, 2], default=1,
                        help="1: long history train (all before 2024.12), 2: 2024 only train")
    parser.add_argument("--seq_len", type=int, default=20,
                        help="Lookback window size (trading days)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Rebuild dataset even if it already exists")
    args = parser.parse_args()

    dataset_name = f"Stock_mode{args.mode}_sl{args.seq_len}"
    out_dir = os.path.join(args.out_root, dataset_name)

    # Check if dataset already exists
    train_path = os.path.join(out_dir, "TRAIN.npz")
    test_path = os.path.join(out_dir, "TEST.npz")
    if os.path.exists(train_path) and os.path.exists(test_path) and not args.overwrite:
        print(f"Dataset {dataset_name} already exists at {out_dir}, skip (use --overwrite to rebuild)")
        return

    print(f"Building dataset: {dataset_name}")
    print(f"  Mode: {args.mode} ({'all history' if args.mode == 1 else '2024 only'})")
    print(f"  Sequence length: {args.seq_len}")

    # Load and process data
    print("\nLoading raw price data...")
    df = load_price_data(args.raw_dir)
    print(f"  Loaded {len(df)} records")

    print("Computing overnight rates and labels...")
    df = add_labels(df)
    print(f"  After processing: {len(df)} records")

    # Print label distribution
    print("\nOverall label distribution:")
    label_counts = df["label"].value_counts().sort_index()
    for label, count in label_counts.items():
        label_name = {0: "down", 1: "hold", 2: "up"}[label]
        print(f"  {label_name} ({label}): {count} ({100*count/len(df):.2f}%)")

    # Split data
    print("\nSplitting train/test...")
    train_df, test_df = split_train_test(df, args.mode)
    print(f"  Train: {len(train_df)} records")
    print(f"  Test: {len(test_df)} records")

    # Build sliding window samples
    # For training: use train_df only
    # For testing: use all data up to test period, but only predict for test dates
    print("\nBuilding sliding window samples...")
    x_train, y_train, meta_train = build_examples(train_df, args.seq_len)

    # For test set, we need to include history from train set to build windows
    # Combine train and test data, but only target test dates
    if args.mode == 1:
        # Mode 1: use all history up to and including test period
        all_data_for_test = df[df["date"] <= datetime(2024, 12, 31)].copy()
    else:
        # Mode 2: use 2024 data (Jan to Dec)
        all_data_for_test = df[(df["date"] >= datetime(2024, 1, 1)) &
                               (df["date"] <= datetime(2024, 12, 31))].copy()

    x_test, y_test, meta_test = build_examples(all_data_for_test, args.seq_len, target_df=test_df)

    print(f"  Train samples: {len(x_train)}")
    print(f"  Test samples: {len(x_test)}")

    # Print train label distribution
    print("\nTrain label distribution:")
    for label in [0, 1, 2]:
        count = (y_train == label).sum()
        label_name = {0: "down", 1: "hold", 2: "up"}[label]
        print(f"  {label_name} ({label}): {count} ({100*count/len(y_train):.2f}%)")

    # Save datasets
    print("\nSaving datasets...")
    save_npz(out_dir, "TRAIN", x_train, y_train)
    save_npz(out_dir, "TEST", x_test, y_test)

    # Save metadata for evaluation
    meta_train.to_csv(os.path.join(out_dir, "train_meta.csv"), index=False)
    meta_test.to_csv(os.path.join(out_dir, "test_meta.csv"), index=False)
    print(f"Saved metadata files")

    # Save label mapping
    with open(os.path.join(out_dir, "label_mapping.txt"), "w") as f:
        f.write("0: down (overnight_rate < -3%)\n")
        f.write("1: hold (-3% <= overnight_rate <= 3%)\n")
        f.write("2: up (overnight_rate > 3%)\n")

    print(f"\nDataset {dataset_name} built successfully at {out_dir}!")


if __name__ == "__main__":
    main()
