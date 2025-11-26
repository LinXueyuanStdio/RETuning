"""
Stock Classification Experiment Class for Time-Series-Library.

Supports:
- Training with cross-entropy loss
- Evaluation with F1, Accuracy, Recall metrics
- Output detailed CSV with predictions
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report, confusion_matrix

from exp.exp_basic import Exp_Basic
from stock_classification.stock_data_loader import data_provider_stock
from utils.tools import EarlyStopping, adjust_learning_rate

warnings.filterwarnings('ignore')


class Exp_Stock_Classification(Exp_Basic):
    """
    Experiment class for stock overnight rate classification.

    Uses existing Time-Series-Library models in classification mode.
    """

    def __init__(self, args):
        super(Exp_Stock_Classification, self).__init__(args)

    def _build_model(self):
        """Build model with classification head."""
        # Get data to determine input dimensions
        train_data, _ = self._get_data(flag='TRAIN')

        # Set model parameters
        self.args.seq_len = train_data.seq_len
        self.args.pred_len = 0
        self.args.enc_in = train_data.feature_dim
        self.args.num_class = train_data.num_classes
        self.args.c_out = train_data.feature_dim

        # Build model
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        return model

    def _get_data(self, flag):
        """Get data loader for stock classification."""
        return data_provider_stock(self.args, flag)

    def _select_optimizer(self):
        """Select optimizer."""
        return optim.AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=1e-4)

    def _select_criterion(self):
        """Select loss function with optional class weights."""
        # Get class distribution from training data
        train_data, _ = self._get_data(flag='TRAIN')

        # Compute class weights for imbalanced data
        class_counts = np.bincount(train_data.y, minlength=3)
        total = len(train_data.y)

        # Inverse frequency weighting
        weights = total / (3 * class_counts + 1e-6)
        weights = torch.FloatTensor(weights).to(self.device)

        print(f"Class distribution: {class_counts}")
        print(f"Class weights: {weights.cpu().numpy()}")

        return nn.CrossEntropyLoss(weight=weights)

    def vali(self, vali_data, vali_loader, criterion):
        """Validation step."""
        self.model.eval()
        total_loss = []
        preds = []
        trues = []

        with torch.no_grad():
            for batch_x, label, padding_mask in vali_loader:
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                # Forward pass
                outputs = self.model(batch_x, padding_mask, None, None)

                loss = criterion(outputs, label.long())
                total_loss.append(loss.item())

                preds.append(outputs.detach().cpu())
                trues.append(label.cpu())

        total_loss = np.mean(total_loss)

        # Compute predictions
        preds = torch.cat(preds, dim=0)
        trues = torch.cat(trues, dim=0)
        probs = torch.softmax(preds, dim=1)
        predictions = torch.argmax(probs, dim=1).numpy()
        trues = trues.numpy()

        # Compute metrics
        accuracy = accuracy_score(trues, predictions)
        f1_macro = f1_score(trues, predictions, average='macro', zero_division=0)
        f1_weighted = f1_score(trues, predictions, average='weighted', zero_division=0)
        recall_macro = recall_score(trues, predictions, average='macro', zero_division=0)

        self.model.train()
        return total_loss, accuracy, f1_macro, f1_weighted, recall_macro

    def train(self, setting):
        """Training loop."""
        train_data, train_loader = self._get_data(flag='TRAIN')
        vali_data, vali_loader = self._get_data(flag='TEST')

        # Create checkpoint directory
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            model_optim, T_max=self.args.train_epochs, eta_min=1e-6
        )

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, label, padding_mask) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                # Forward pass
                outputs = self.model(batch_x, padding_mask, None, None)
                loss = criterion(outputs, label.long())
                train_loss.append(loss.item())

                # Logging
                if (i + 1) % 100 == 0:
                    print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(f'\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s')
                    iter_count = 0
                    time_now = time.time()

                # Backward pass
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            scheduler.step()

            print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time:.2f}s")

            train_loss = np.mean(train_loss)
            vali_loss, vali_acc, vali_f1_macro, vali_f1_weighted, vali_recall = self.vali(
                vali_data, vali_loader, criterion
            )

            print(f"Epoch: {epoch + 1}, Steps: {train_steps} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Vali Loss: {vali_loss:.4f} Acc: {vali_acc:.4f} "
                  f"F1_macro: {vali_f1_macro:.4f} F1_weighted: {vali_f1_weighted:.4f} "
                  f"Recall: {vali_recall:.4f}")

            # Early stopping based on F1 score
            early_stopping(-vali_f1_macro, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        # Load best model
        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        """Test and generate evaluation results."""
        test_data, test_loader = self._get_data(flag='TEST')

        if test:
            print('Loading model...')
            self.model.load_state_dict(
                torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'))
            )

        preds = []
        trues = []
        probs_list = []

        # Create results directory
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for batch_x, label, padding_mask in test_loader:
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)

                probs = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(probs, dim=1)

                preds.append(predictions.cpu().numpy())
                trues.append(label.cpu().numpy())
                probs_list.append(probs.cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        probs_all = np.concatenate(probs_list, axis=0)

        print(f'Test shape: predictions={preds.shape}, trues={trues.shape}')

        # Compute metrics
        accuracy = accuracy_score(trues, preds)
        f1_macro = f1_score(trues, preds, average='macro', zero_division=0)
        f1_weighted = f1_score(trues, preds, average='weighted', zero_division=0)
        f1_per_class = f1_score(trues, preds, average=None, zero_division=0)
        recall_macro = recall_score(trues, preds, average='macro', zero_division=0)
        recall_per_class = recall_score(trues, preds, average=None, zero_division=0)

        # Classification report
        class_names = ['down', 'hold', 'up']
        report = classification_report(trues, preds, target_names=class_names, zero_division=0)
        conf_matrix = confusion_matrix(trues, preds)

        print("\n" + "="*60)
        print("Classification Report:")
        print(report)
        print("\nConfusion Matrix:")
        print(conf_matrix)
        print("="*60)

        # Save metrics
        results_folder = './results/' + setting + '/'
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)

        metrics = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'f1_down': f1_per_class[0],
            'f1_hold': f1_per_class[1],
            'f1_up': f1_per_class[2],
            'recall_macro': recall_macro,
            'recall_down': recall_per_class[0],
            'recall_hold': recall_per_class[1],
            'recall_up': recall_per_class[2],
        }

        print(f"\nMetrics Summary:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1 (macro): {f1_macro:.4f}")
        print(f"  F1 (weighted): {f1_weighted:.4f}")
        print(f"  Recall (macro): {recall_macro:.4f}")

        # Save metrics to file
        with open(os.path.join(results_folder, 'metrics.txt'), 'w') as f:
            f.write(f"Setting: {setting}\n")
            f.write("="*60 + "\n")
            for key, value in metrics.items():
                f.write(f"{key}: {value:.6f}\n")
            f.write("\n" + "="*60 + "\n")
            f.write("Classification Report:\n")
            f.write(report)
            f.write("\n" + "="*60 + "\n")
            f.write("Confusion Matrix:\n")
            f.write(str(conf_matrix))

        # Generate detailed evaluation CSV
        self._save_evaluation_csv(test_data, preds, probs_all, results_folder, setting)

        return metrics

    def _save_evaluation_csv(self, test_data, predictions, probs, results_folder, setting):
        """
        Save detailed evaluation CSV with columns:
        date, code, open, high, low, close, pct_change, label, predicted_label
        """
        if test_data.meta_df is None:
            print("Warning: No metadata available for evaluation CSV")
            return

        # Create output DataFrame
        output_df = test_data.meta_df.copy()
        output_df['predicted_label'] = predictions

        # Add probability columns
        output_df['prob_down'] = probs[:, 0]
        output_df['prob_hold'] = probs[:, 1]
        output_df['prob_up'] = probs[:, 2]

        # Add label names
        label_map = {0: 'down', 1: 'hold', 2: 'up'}
        output_df['label_name'] = output_df['label'].map(label_map)
        output_df['predicted_label_name'] = output_df['predicted_label'].map(label_map)

        # Reorder columns
        columns_order = [
            'date', 'code', 'open', 'high', 'low', 'close', 'pct_change',
            'label', 'label_name', 'predicted_label', 'predicted_label_name',
            'prob_down', 'prob_hold', 'prob_up'
        ]
        output_df = output_df[[c for c in columns_order if c in output_df.columns]]

        # Save CSV
        csv_path = os.path.join(results_folder, 'evaluation_results.csv')
        output_df.to_csv(csv_path, index=False)
        print(f"Saved evaluation results to {csv_path}")

        # Also save a summary by date
        if 'date' in output_df.columns:
            date_summary = output_df.groupby('date').agg({
                'label': 'value_counts',
                'predicted_label': 'value_counts',
            }).reset_index()

            # Compute daily accuracy
            daily_acc = output_df.groupby('date').apply(
                lambda x: (x['label'] == x['predicted_label']).mean()
            ).reset_index(name='daily_accuracy')

            daily_summary_path = os.path.join(results_folder, 'daily_summary.csv')
            daily_acc.to_csv(daily_summary_path, index=False)
            print(f"Saved daily summary to {daily_summary_path}")
