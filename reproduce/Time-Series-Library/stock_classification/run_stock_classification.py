"""
Run Stock Classification with Time-Series-Library Models.

Usage:
    python run_stock_classification.py \
        --model TimesNet \
        --mode 1 \
        --seq_len 20 \
        --is_training 1

Models supported:
    - PatchTST
    - Informer
    - DLinear
    - Autoformer
    - TimesNet
    - TimeMixer
"""

import argparse
import os
import sys
import random
import numpy as np
import torch
import torch.backends

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.print_args import print_args


def main():
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='Stock Classification with Time-Series Models')

    # Basic config
    parser.add_argument('--is_training', type=int, required=True, default=1,
                        help='1: training, 0: testing')
    parser.add_argument('--model_id', type=str, default='stock_cls',
                        help='model id')
    parser.add_argument('--model', type=str, required=True, default='TimesNet',
                        help='model name: [PatchTST, Informer, DLinear, Autoformer, TimesNet, TimeMixer]')

    # Data config
    parser.add_argument('--data', type=str, default='Stock',
                        help='dataset type')
    parser.add_argument('--root_path', type=str, default='./stock_classification/dataset',
                        help='root path of the dataset')
    parser.add_argument('--data_path', type=str, default='STOCK.csv',
                        help='data file (placeholder for compatibility)')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task features: [M, S, MS]')
    parser.add_argument('--target', type=str, default='overnight_rate',
                        help='target feature')
    parser.add_argument('--mode', type=int, choices=[1, 2], default=1,
                        help='1: long history train, 2: 2024 only train')
    parser.add_argument('--seq_len', type=int, default=20,
                        help='input sequence length (lookback window)')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/',
                        help='location of model checkpoints')
    parser.add_argument('--seasonal_patterns', type=str, default='Daily',
                        help='seasonal patterns (for print_args compatibility)')
    parser.add_argument('--inverse', action='store_true', default=False,
                        help='inverse output data (for print_args compatibility)')

    # Model parameters
    parser.add_argument('--enc_in', type=int, default=1,
                        help='encoder input size (1 for overnight_rate only)')
    parser.add_argument('--dec_in', type=int, default=1,
                        help='decoder input size')
    parser.add_argument('--c_out', type=int, default=3,
                        help='output size (3 classes)')
    parser.add_argument('--d_model', type=int, default=64,
                        help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=4,
                        help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2,
                        help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1,
                        help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=128,
                        help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=5,
                        help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1,
                        help='attn factor')
    parser.add_argument('--distil', action='store_false', default=True,
                        help='whether to use distilling in encoder')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding: [timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu',
                        help='activation function')
    parser.add_argument('--freq', type=str, default='d',
                        help='freq for time features encoding: d=daily')

    # TimesNet specific
    parser.add_argument('--top_k', type=int, default=3,
                        help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6,
                        help='for Inception')

    # TimeMixer specific
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='channel independence for TimeMixer')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='decomposition method for TimeMixer')
    parser.add_argument('--use_norm', type=int, default=1,
                        help='whether to use normalize')
    parser.add_argument('--down_sampling_layers', type=int, default=0,
                        help='num of down sampling layers for TimeMixer')
    parser.add_argument('--down_sampling_window', type=int, default=1,
                        help='down sampling window size for TimeMixer')
    parser.add_argument('--down_sampling_method', type=str, default=None,
                        help='down sampling method for TimeMixer')

    # PatchTST specific
    parser.add_argument('--patch_len', type=int, default=4,
                        help='patch length for PatchTST')

    # Optimization
    parser.add_argument('--num_workers', type=int, default=4,
                        help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1,
                        help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=30,
                        help='train epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--patience', type=int, default=5,
                        help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='Exp',
                        help='exp description')
    parser.add_argument('--loss', type=str, default='CE',
                        help='loss function')
    parser.add_argument('--lradj', type=str, default='type1',
                        help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', default=False,
                        help='use automatic mixed precision training')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True,
                        help='use gpu')
    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu')
    parser.add_argument('--gpu_type', type=str, default='cuda',
                        help='gpu type: cuda or mps')
    parser.add_argument('--use_multi_gpu', action='store_true', default=False,
                        help='use multiple gpus')
    parser.add_argument('--devices', type=str, default='0,1,2,3',
                        help='device ids of multiple gpus')

    # De-stationary Projector Params (for print_args compatibility)
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden dimensions of projector')
    parser.add_argument('--p_hidden_layers', type=int, default=2,
                        help='hidden layers of projector')

    # Classification specific (will be set automatically)
    parser.add_argument('--task_name', type=str, default='classification',
                        help='task name')
    parser.add_argument('--label_len', type=int, default=0,
                        help='label length (not used for classification)')
    parser.add_argument('--pred_len', type=int, default=0,
                        help='prediction length (not used for classification)')

    # Overwrite control
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='overwrite existing results (default: skip if results exist)')
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='directory to save results')

    args = parser.parse_args()

    # Force classification task
    args.task_name = 'classification'
    args.label_len = 0
    args.pred_len = 0

    # Device setup
    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device('cuda:{}'.format(args.gpu))
        print('Using GPU')
    else:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            args.device = torch.device("mps")
        else:
            args.device = torch.device("cpu")
        print('Using CPU or MPS')

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print_args(args)

    # Import experiment class
    from stock_classification.exp_stock_classification import Exp_Stock_Classification
    Exp = Exp_Stock_Classification

    if args.is_training:
        for ii in range(args.itr):
            # Setting string for experiment identification
            setting = 'StockCls_{}_mode{}_sl{}_dm{}_nh{}_el{}_df{}_{}_{}'.format(
                args.model,
                args.mode,
                args.seq_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_ff,
                args.des,
                ii
            )

            # Check if results already exist
            result_path = os.path.join(args.results_dir, setting, 'metrics.txt')
            if os.path.exists(result_path) and not args.overwrite:
                print(f'>>>>>>>SKIP (already exists): {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
                print(f'       Results found at: {result_path}')
                print(f'       Use --overwrite to re-run this experiment')
                continue

            exp = Exp(args)

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            # Clean up GPU memory
            if args.gpu_type == 'mps':
                torch.backends.mps.empty_cache()
            elif args.gpu_type == 'cuda':
                torch.cuda.empty_cache()
    else:
        ii = 0
        setting = 'StockCls_{}_mode{}_sl{}_dm{}_nh{}_el{}_df{}_{}_{}'.format(
            args.model,
            args.mode,
            args.seq_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_ff,
            args.des,
            ii
        )

        exp = Exp(args)
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)

        if args.gpu_type == 'mps':
            torch.backends.mps.empty_cache()
        elif args.gpu_type == 'cuda':
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
