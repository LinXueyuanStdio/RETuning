import os
import argparse
import glob
import pandas as pd
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score, recall_score

from exp.exp_basic import Exp_Basic
from exp.exp_classification import Exp_Classification


def load_price_data(raw_dir):
    files = sorted(glob.glob(os.path.join(raw_dir, '*.csv')))
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    # ensure columns
    needed = ['date', 'code', 'open', 'high', 'low', 'close', 'pre_close']
    for c in needed:
        if c not in df.columns:
            raise ValueError(f'missing column {c} in data')
    # sort
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    return df


def add_overnight_label(df):
    r = (df['open'] - df['pre_close']) / df['pre_close']
    cond_up = r > 0.03
    cond_down = r < -0.03
    labels = np.where(cond_up, 2, np.where(cond_down, 0, 1))
    df = df.copy()
    df['overnight_ret'] = r
    df['label'] = labels
    return df


def split_train_test(df, mode):
    # mode 1: 10y train, 2024-12 test; here we approximate using all <2024-12 as train
    test_start = datetime(2024, 12, 1)
    test_end = datetime(2024, 12, 31)
    if mode == 1:
        train_mask = df['date'] < test_start
    elif mode == 2:
        train_mask = (df['date'] >= datetime(2024, 1, 1)) & (df['date'] < test_start)
    else:
        raise ValueError('mode must be 1 or 2')
    test_mask = (df['date'] >= test_start) & (df['date'] <= test_end)
    return df[train_mask].reset_index(drop=True), df[test_mask].reset_index(drop=True)


class StockDataset(Dataset):
    def __init__(self, df, seq_len):
        self.seq_len = seq_len
        self.samples = []
        features_cols = ['open', 'high', 'low', 'close', 'pre_close']
        self.feature_cols = features_cols
        for code, g in df.groupby('code'):
            g = g.sort_values('date').reset_index(drop=True)
            values = g[features_cols].values.astype(np.float32)
            labels = g['label'].values.astype(np.int64)
            dates = g['date'].values
            for i in range(seq_len, len(g)):
                x = values[i-seq_len:i]
                y = labels[i]
                meta = (dates[i], code,
                        g.loc[i, 'open'], g.loc[i, 'high'], g.loc[i, 'low'], g.loc[i, 'close'],
                        g.loc[i, 'overnight_ret'])
                self.samples.append((x, y, meta))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y, meta = self.samples[idx]
        padding_mask = np.ones(self.seq_len, dtype=np.float32)
        return torch.from_numpy(x), torch.tensor(y), torch.from_numpy(padding_mask), meta


def collate_stock(batch):
    xs, ys, masks, metas = zip(*batch)
    xs = torch.stack(xs, 0)  # B,L,C
    ys = torch.stack(ys, 0).unsqueeze(-1)
    masks = torch.stack(masks, 0)
    return xs, ys, masks, metas


class StockExp(Exp_Classification):
    def __init__(self, args, train_ds, test_ds, num_features):
        self.train_ds = train_ds
        self.test_ds = test_ds
        args.enc_in = num_features
        args.seq_len = train_ds.seq_len
        args.pred_len = 0
        args.num_class = 3
        super().__init__(args)

    def _get_data(self, flag):
        if flag == 'TRAIN':
            ds = self.train_ds
        else:
            ds = self.test_ds
        loader = DataLoader(ds, batch_size=self.args.batch_size, shuffle=(flag == 'TRAIN'),
                            num_workers=self.args.num_workers, drop_last=False,
                            collate_fn=collate_stock)
        return ds, loader


def run_one_setting(args, mode, seq_len, model_name, raw_dir, out_dir):
    df = load_price_data(raw_dir)
    df = add_overnight_label(df)
    train_df, test_df = split_train_test(df, mode)

    train_ds = StockDataset(train_df, seq_len)
    test_ds = StockDataset(test_df, seq_len)

    num_features = len(train_ds.feature_cols)
    exp_args = argparse.Namespace(**vars(args))
    exp_args.model = model_name
    exp_args.task_name = 'classification'
    exp_args.data = 'custom'
    exp_args.root_path = ''
    exp = StockExp(exp_args, train_ds, test_ds, num_features)

    setting = f'stock_mode{mode}_sl{seq_len}_{model_name}'
    exp.train(setting)

    # evaluation on test with meta and metrics
    _, test_loader = exp._get_data('TEST')
    preds = []
    trues = []
    metas_all = []
    exp.model.eval()
    with torch.no_grad():
        for batch_x, label, padding_mask, metas in test_loader:
            batch_x = batch_x.float().to(exp.device)
            padding_mask = padding_mask.float().to(exp.device)
            label = label.to(exp.device)
            outputs = exp.model(batch_x, padding_mask, None, None)
            preds.append(outputs.detach().cpu())
            trues.append(label.cpu())
            metas_all.extend(metas)

    preds = torch.cat(preds, 0)
    trues = torch.cat(trues, 0).view(-1).numpy()
    probs = torch.softmax(preds, dim=1).numpy()
    pred_labels = probs.argmax(axis=1)

    f1 = f1_score(trues, pred_labels, average='macro')
    acc = accuracy_score(trues, pred_labels)
    recall = recall_score(trues, pred_labels, average='macro')
    print(setting, 'F1', f1, 'Acc', acc, 'Recall', recall)

    rows = []
    for (d, code, o, h, l, c, r), y, yhat in zip(metas_all, trues, pred_labels):
        rows.append({
            'date': pd.to_datetime(d).strftime('%Y-%m-%d'),
            'code': code,
            'open': o,
            'high': h,
            'low': l,
            'close': c,
            'pct_change': r,
            'label': int(y),
            'predicted_label': int(yhat)
        })
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f'{setting}.csv')
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print('saved', csv_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dir', type=str, default='../data/raw')
    parser.add_argument('--out_dir', type=str, default='./stock_results')
    parser.add_argument('--train_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--gpu_type', type=str, default='cuda')
    args = parser.parse_args()

    models = ['PatchTST', 'Informer', 'DLinear', 'Autoformer', 'TimesNet', 'TimeMixer']
    seq_lens = [5, 10, 20, 60]
    for mode in [1, 2]:
        for sl in seq_lens:
            for m in models:
                run_one_setting(args, mode, sl, m, args.raw_dir, args.out_dir)


if __name__ == '__main__':
    main()
