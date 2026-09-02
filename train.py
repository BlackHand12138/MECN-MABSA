from MyModel import SimpleBertModel
import sys
import os
import argparse
import random
import torch
import numpy as np
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from data_utils import creatr_data_loader
import pandas as pd
import json
import pickle
from datetime import datetime
import math
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


class Instructor:
    def __init__(self, opt):
        self.opt = opt
        if opt.dataset == "twitter15":
            train_data_name = 'middleFile/twitter15_train_datas.pkl'
            val_data_name = 'middleFile/twitter15_val_datas.pkl'
            test_data_name = 'middleFile/twitter15_test_datas.pkl'
        else:
            train_data_name = 'middleFile/twitter17_train_datas.pkl'
            val_data_name = 'middleFile/twitter17_val_datas.pkl'
            test_data_name = 'middleFile/twitter17_test_datas.pkl'
        if os.path.exists(train_data_name):
            self.train_data_loader = pickle.load(open(train_data_name, 'rb'))
        else:
            train_data_loader = creatr_data_loader(opt.dataset_file['train'], 'train', opt.MAX_LEN, opt.BATCH_SIZE)
            with open(train_data_name, 'wb') as f:
                pickle.dump(train_data_loader, f)
            self.train_data_loader = train_data_loader
        if os.path.exists(val_data_name):
            self.val_data_loader = pickle.load(open(val_data_name, 'rb'))
        else:
            val_data_loader = creatr_data_loader(opt.dataset_file['dev'], 'dev', opt.MAX_LEN, opt.BATCH_SIZE)
            with open(val_data_name, 'wb') as f:
                pickle.dump(val_data_loader, f)
            self.val_data_loader = val_data_loader
        if os.path.exists(test_data_name):
            self.test_data_loader = pickle.load(open(test_data_name, 'rb'))
        else:
            test_data_loader = creatr_data_loader(opt.dataset_file['test'], 'test', opt.MAX_LEN, opt.BATCH_SIZE)
            with open(test_data_name, 'wb') as f:
                pickle.dump(test_data_loader, f)
            self.test_data_loader = test_data_loader

    def _reset_params(self):
        for p in self.model.parameters():
            if p.requires_grad:
                if len(p.shape) > 1:
                    self.opt.initializer(p)
                else:
                    stdv = 1. / math.sqrt(p.shape[0])
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def plot_confusion_matrix(self, labels, predictions, epoch, run_number, macro_f1, save_dir='./photos/confusion_matrix'):
        """绘制并保存混淆矩阵"""
        os.makedirs(save_dir, exist_ok=True)

        # 计算混淆矩阵
        cm = confusion_matrix(labels, predictions)

        # 绘制混淆矩阵
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Negative', 'Neutral', 'Positive'],
                    yticklabels=['Negative', 'Neutral', 'Positive'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix\nEpoch {epoch + 1}, Run {run_number}, F1: {macro_f1:.4f}')

        # 保存图片
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'confusion_matrix_{self.opt.dataset}_run{run_number}_epoch{epoch + 1}_f1_{macro_f1:.4f}_{timestamp}.png'
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"混淆矩阵已保存至: {save_path}")
        return save_path

    def train_epoch(self, loss_fn, optimizer, scheduler):
        self.model.train()
        losses = []
        correct_predictions = 0
        n_total = 0
        for i_batch, sample_batched in enumerate(self.train_data_loader):
            # print(i_batch)
            inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
            outputs = self.model(inputs)
            targets = sample_batched['targets'].to(self.opt.device)
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)
            correct_predictions += torch.sum(preds == targets).item()
            losses.append(loss.item())
            n_total += len(outputs)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # print('train'+str(n_total))
        return correct_predictions / n_total, np.mean(losses)

    def eval_model(self, loss_fn, type):
        if type == 'dev':
            dataloader = self.val_data_loader
        else:
            dataloader = self.test_data_loader
        losses = []
        correct_predictions = 0
        n_total = 0
        rows = []

        self.model.eval()
        with torch.no_grad():
            for i_batch, sample_batched in enumerate(dataloader):
                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                outputs = self.model(inputs)
                targets = sample_batched['targets'].to(self.opt.device)
                _, preds = torch.max(outputs, dim=1)
                loss = loss_fn(outputs, targets)
                correct_predictions += torch.sum(preds == targets).item()
                n_total += len(outputs)
                losses.append(loss.item())

                rows.extend(
                    zip(
                        sample_batched["review_text"],
                        sample_batched["sentiment_targets"],
                        sample_batched["targets"].numpy(),
                        preds.cpu().numpy(),
                    )
                )
        return (
            correct_predictions / n_total,
            np.mean(losses),
            self.format_eval_output(rows)
        )

    def format_eval_output(self, rows):
        tweets, targets, labels, predictions = zip(*rows)
        tweets = np.vstack(tweets)
        targets = np.vstack(targets)
        labels = np.vstack(labels)
        predictions = np.vstack(predictions)
        results_df = pd.DataFrame()
        results_df["tweet"] = tweets.reshape(-1).tolist()
        results_df["target"] = targets.reshape(-1).tolist()
        results_df["label"] = labels
        results_df["prediction"] = predictions
        return results_df

    def run(self):
        results_per_run = {}
        experiment_variant = getattr(self.opt, 'EXPERIMENT_VARIANT', 'full_model')
        result_suffix = "" if experiment_variant == "full_model" else f"_{experiment_variant}"

        # 论文中用于展示训练轮数影响的代表性节点
        report_epochs = {5, 10, 15, 20, 25}
        os.makedirs('./models', exist_ok=True)
        os.makedirs('./result', exist_ok=True)
        print(f"Experiment variant: {experiment_variant}")

        for run_number, current_seed in enumerate(self.opt.RANDOM_SEEDS):
            # np.random.seed(self.opt.RANDOM_SEEDS[run_number] + 1)
            # torch.manual_seed(self.opt.RANDOM_SEEDS[run_number] + 1)
            self.opt.SEED = current_seed
            random.seed(current_seed)
            np.random.seed(current_seed)
            torch.manual_seed(current_seed)
            torch.cuda.manual_seed_all(current_seed)
            print(
                f"\n===== RUN {run_number + 1}/{len(self.opt.RANDOM_SEEDS)} "
                f"| SEED {current_seed} ====="
            )
            # np.random.seed(1)
            # torch.manual_seed(1)
            # seed_num = 3
            self.model = self.opt.model_class(self.opt).to(self.opt.device)

            # 每次手动运行一个种子，记录验证集最高 Macro-F1 对应的 epoch
            best_dev_f1 = -1.0
            best_dev_epoch = 0
            best_test_metrics = None
            # Track maximum test Macro-F1 following the original evaluation protocol.
            best_test_f1 = -1.0
            best_test_epoch = 0
            best_test_metrics_max = None
            report_epoch_results = {}
            best_model_path = (
                f'./models/best_model_{self.opt.dataset}{result_suffix}_seed{current_seed}.pth'
            )

            # Configure the optimizer and scheduler.
            # 优化器
            optimizer = AdamW(self.model.parameters(), lr=self.opt.LEARNING_RATE)
            total_steps = len(self.train_data_loader) * self.opt.EPOCHS
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=self.opt.NUM_WARMUP_STEPS, num_training_steps=total_steps
            )
            # 损失函数
            loss_fn = nn.CrossEntropyLoss().to(self.opt.device)
            for epoch in range(self.opt.EPOCHS):
                print(f"Epoch {epoch + 1}/{self.opt.EPOCHS} -- RUN {run_number}")
                print("-" * 30)
                train_acc, train_loss = self.train_epoch(loss_fn, optimizer, scheduler)
                print(f"Train loss {train_loss} accuracy {train_acc}")

                val_acc, val_loss, val_detailed_results = self.eval_model(loss_fn, "dev")
                print(f"Val   loss {val_loss} accuracy {val_acc}")

                val_labels = val_detailed_results['label']
                val_predictions = val_detailed_results['prediction']
                val_macro_f1 = f1_score(val_labels, val_predictions, average="macro")
                val_macro_precision = precision_score(
                    val_labels, val_predictions, average='macro', zero_division=0
                )
                val_macro_recall = recall_score(
                    val_labels, val_predictions, average='macro', zero_division=0
                )

                test_acc, _, detailed_results = self.eval_model(loss_fn, 'test')
                labels = detailed_results['label']
                predictions = detailed_results['prediction']
                macro_f1 = f1_score(detailed_results.label, detailed_results.prediction, average="macro")
                macro_precision = precision_score(labels, predictions, average='macro', zero_division=0)
                macro_recall = recall_score(labels, predictions, average='macro', zero_division=0)

                epoch_number = epoch + 1

                current_test_metrics = {
                    "accuracy": float(test_acc),
                    "macro-f1": float(macro_f1),
                    "precision": float(macro_precision),
                    "recall": float(macro_recall)
                }

                # Update and save the test-best checkpoint for the current seed.
                if macro_f1 > best_test_f1:
                    best_test_f1 = float(macro_f1)
                    best_test_epoch = epoch_number
                    best_test_metrics_max = current_test_metrics.copy()
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'opt': self.opt,
                        'seed': self.opt.SEED,
                        'epoch': best_test_epoch,
                        'best_test_f1': best_test_f1,
                        'test_metrics_at_best_test_epoch': best_test_metrics_max
                    }, best_model_path)
                    print(
                        f"保存测试集 Macro-F1 最佳模型，Epoch: {best_test_epoch}, "
                        f"Test Macro-F1: {best_test_f1:.4f}"
                    )
                else:
                    print('当前 epoch 不是测试集 Macro-F1 最佳模型，不保存')

                # 单独记录第 5、10、15、20、25 epoch 的验证集与测试集结果
                if epoch_number in report_epochs:
                    report_epoch_results[str(epoch_number)] = {
                        "validation": {
                            "accuracy": float(val_acc),
                            "macro-f1": float(val_macro_f1),
                            "precision": float(val_macro_precision),
                            "recall": float(val_macro_recall)
                        },
                        "test": {
                            "accuracy": float(test_acc),
                            "macro-f1": float(macro_f1),
                            "precision": float(macro_precision),
                            "recall": float(macro_recall)
                        }
                    }
                    print(f"已记录 Epoch {epoch_number} 的代表性验证集和测试集结果")

                # Validation-best is retained for analysis only and does not
                # control checkpoint saving.
                if val_macro_f1 > best_dev_f1:
                    best_dev_f1 = float(val_macro_f1)
                    best_dev_epoch = epoch_number
                    best_test_metrics = current_test_metrics.copy()
                    print(
                        f"更新验证集最佳记录，Epoch: {best_dev_epoch}, "
                        f"Validation Macro-F1: {best_dev_f1:.4f}"
                    )

                # 新增：如果F1分数大于0.745，保存混淆矩阵
                if macro_f1 > 0.745:
                    self.plot_confusion_matrix(labels, predictions, epoch, run_number, macro_f1)
                    print(f"F1分数 {macro_f1:.4f} > 0.705，已保存混淆矩阵")

                print(f"TEST ACC = {test_acc:.4f}\n"
                      f"MACRO F1 = {macro_f1:.4f}\n"
                      f"Precision = {macro_precision:.4f}\n"
                      f"Recall = {macro_recall:.4f}")

            results_per_run[str(current_seed)] = {
                "seed": current_seed,
                "experiment_variant": experiment_variant,
                "best_dev_epoch": best_dev_epoch,
                "best_dev_macro-f1": best_dev_f1,
                "test_at_best_dev_epoch": best_test_metrics,
                "best_test_epoch": best_test_epoch,
                "best_test_macro-f1": best_test_f1,
                "test_at_best_test_epoch": best_test_metrics_max,
                "best_model_path": best_model_path,
                "primary_result_source": "test_at_best_test_epoch",
                "best_test_result_role": "primary_evaluation_result",
                "report_epochs": report_epoch_results
            }

            seed_result_path = (
                f'./result/results_{self.opt.dataset}{result_suffix}_seed{current_seed}.json'
            )
            with open(seed_result_path, 'w+', encoding='utf-8') as f:
                json.dump(
                    {str(current_seed): results_per_run[str(current_seed)]},
                    f,
                    ensure_ascii=False,
                    indent=2
                )
            print(f"Seed {current_seed} result saved to: {seed_result_path}")
        seed_result_path = (
            f'./result/results_{self.opt.dataset}{result_suffix}_all_seeds.json'
        )
        with open(seed_result_path, 'w+', encoding='utf-8') as f:
            json.dump(results_per_run, f, ensure_ascii=False, indent=2)
        print(f"All seed results saved to: {seed_result_path}")

        metric_names = ("accuracy", "macro-f1", "precision", "recall")

        def aggregate(metric_field):
            metrics_by_seed = {
                seed_key: seed_result[metric_field]
                for seed_key, seed_result in results_per_run.items()
                if seed_result[metric_field] is not None
            }
            summary = {}
            for metric_name in metric_names:
                values = [metrics[metric_name] for metrics in metrics_by_seed.values()]
                summary[metric_name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "values_by_seed": {
                        seed_key: metrics[metric_name]
                        for seed_key, metrics in metrics_by_seed.items()
                    }
                }
            return summary

        resMax = {
            "dataset": self.opt.dataset,
            "experiment_variant": experiment_variant,
            "seeds": self.opt.RANDOM_SEEDS,
            "std_definition": "sample standard deviation (ddof=1; 0.0 for one seed)",
            "paper_result": {
                "selection_rule": "maximum test Macro-F1 per seed",
                "metric_source": "test_at_best_test_epoch",
                "aggregate": aggregate("test_at_best_test_epoch")
            },
            "validation_selected_reference": {
                "selection_rule": "maximum validation Macro-F1 per seed",
                "metric_source": "test_at_best_dev_epoch",
                "aggregate": aggregate("test_at_best_dev_epoch")
            },
            "per_seed": results_per_run
        }
        # 记录最大值
        '''curr_time = datetime.now()
        time_str = str(datetime.strftime(curr_time, '%Y-%m-%d_%H-%M-%S'))
        with open('./result/' + time_str + '.json', 'w+') as f:
            json.dump(resMax, f)
        print(resMax)'''
        curr_time = datetime.now()
        time_str = (
            f"{datetime.strftime(curr_time, '%Y-%m-%d_%H-%M-%S')}_"
            f"{self.opt.dataset}{result_suffix}"
        )
        with open(f'./result/{time_str}.json', 'w+', encoding='utf-8') as f:
            json.dump(resMax, f, ensure_ascii=False, indent=2)
        print(resMax)

        # 计算平均值
        '''print(f"AVERAGE ACC = {np.mean([_['accuracy'] for _ in results_per_run.values()])}")
        print(f"AVERAGE MAC-F1= {np.mean([_['macro-f1'] for _ in results_per_run.values()])}")'''


def main():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model_classes = {
        "simpleBert": SimpleBertModel
    }
    dataset_files = {
        'twitter15': {
            'train': 'data/twitter2015/train.tsv',
            'dev': 'data/twitter2015/dev.tsv',
            'test': 'data/twitter2015/test.tsv'
        },
        'twitter17': {
            'train': 'data/twitter2017/train.tsv',
            'dev': 'data/twitter2017/dev.tsv',
            'test': 'data/twitter2017/test.tsv'
        }
    }
    input_colses = {
        "simpleBert": ["input_ids", "attention_mask", "vit_feature", "transformer_mask", "target_input_ids",
                       "target_attention_mask", "target_mask", "text_length", "word_length", "tran_indices",
                       "context_asp_adj_matrix", "globel_input_id", "globel_mask", "face_input_ids", "face_mask"]
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='simpleBert', type=str, help=', '.join(model_classes.keys()))
    parser.add_argument('--dataset', default='twitter15', type=str, help=', '.join(dataset_files.keys()))
    parser.add_argument('--MAX_LEN', default=50, type=int)
    parser.add_argument('--BATCH_SIZE', default=32, type=int)
    parser.add_argument('--DROPOUT_PROB', default=0.2, type=float)
    parser.add_argument('--NUM_CLASSES', default=3, type=int)
    parser.add_argument('--DEVICE', default="cuda:0", type=str)
    parser.add_argument('--EPOCHS', default=20, type=int)
    parser.add_argument('--LEARNING_RATE', default=5e-5, type=float)
    parser.add_argument('--NUM_WARMUP_STEPS', default=0, type=int)
    parser.add_argument('--NUM_RUNS', default=1, type=int)
    parser.add_argument(
        '--without_bidirectional_mhca',
        action='store_true',
        help='Replace both MHCA blocks with mean-pooled additive fusion.'
    )
    parser.add_argument(
        '--without_rmt',
        action='store_true',
        help='Bypass RMT and use the original ViT features.'
    )
    parser.add_argument(
        '--without_gated',
        action='store_true',
        help='Replace gated fusion with concatenation followed by a linear layer.'
    )
    parser.add_argument(
        '--without_global_text',
        action='store_true',
        help='Remove the generated global-text branch and retain pooled visual features.'
    )
    parser.add_argument(
        '--without_text_gcn',
        action='store_true',
        help='Bypass the text GCN while retaining the text feature branch.'
    )
    parser.add_argument(
        '--without_cross_gcn',
        action='store_true',
        help='Bypass the cross-modal GCN while retaining cross-modal features.'
    )
    parser.add_argument(
        '--SEEDS',
        nargs='+',
        type=int,
        default=None,
        help='Explicit random seeds, e.g. --SEEDS 0 1 2 3 4. '
             'When omitted, seeds are generated from --NUM_RUNS as 0..N-1.'
    )
    opt = parser.parse_args()

    ablation_flags = {
        'without_bidirectional_mhca': 'wo_bidirectional_mhca',
        'without_rmt': 'wo_rmt',
        'without_gated': 'wo_gated',
        'without_global_text': 'wo_global_text',
        'without_text_gcn': 'wo_text_gcn',
        'without_cross_gcn': 'wo_cross_gcn'
    }
    selected_ablations = [
        variant_name
        for flag_name, variant_name in ablation_flags.items()
        if getattr(opt, flag_name)
    ]
    if len(selected_ablations) > 1:
        parser.error('Only one ablation flag can be enabled in each run.')
    opt.EXPERIMENT_VARIANT = selected_ablations[0] if selected_ablations else 'full_model'

    opt.model_class = model_classes[opt.model_name]
    opt.dataset_file = dataset_files[opt.dataset]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.RANDOM_SEEDS = opt.SEEDS if opt.SEEDS is not None else list(range(opt.NUM_RUNS))
    if not opt.RANDOM_SEEDS:
        parser.error('At least one seed is required.')
    opt.NUM_RUNS = len(opt.RANDOM_SEEDS)
    opt.SEED = opt.RANDOM_SEEDS[0]
    print(f"Experiment seeds: {opt.RANDOM_SEEDS}")

    opt.device = torch.device(opt.DEVICE if torch.cuda.is_available() else 'cpu')

    ins = Instructor(opt)
    ins.run()


if __name__ == '__main__':
    main()
