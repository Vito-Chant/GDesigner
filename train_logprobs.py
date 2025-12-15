import os
import glob
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

# ==========================================
# 配置项
# ==========================================
TRAIN_DIR = "training_data_v1"
VAL_DIR = "validation_data_v1"
MODEL_SAVE_PATH = "logprob_relative_best.pth"

BATCH_SIZE = 32  # Task Batch Size
LEARNING_RATE = 5e-5  # 【关键】进一步降低 LR，Ranking Loss 对 LR 很敏感
NUM_EPOCHS = 200
MAX_SEQ_LEN = 512  # 适当减小序列长度，减少噪声
DROPOUT = 0.5  # 【关键】增大 Dropout 防止过拟合
MARGIN = 0.1  # Ranking Margin (Score_pos - Score_neg > 0.1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ==========================================
# 1. 模型 (简化版)
# ==========================================
class MaskedGlobalPooling(nn.Module):
    def forward(self, x, mask):
        # x: [B, C, L], mask: [B, L] (True is padding)
        mask_expanded = mask.unsqueeze(1)

        # Mean
        x_zeroed = x.masked_fill(mask_expanded, 0.0)
        sum_pooled = x_zeroed.sum(dim=2)
        lengths = (~mask).sum(dim=1, keepdim=True).float().clamp(min=1.0)
        mean_pooled = sum_pooled / lengths

        # Max
        x_neg = x.masked_fill(mask_expanded, -1e9)
        max_pooled = x_neg.max(dim=2)[0]

        return torch.cat([mean_pooled, max_pooled], dim=1)


class RelativeLogprobCNN(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=32, dropout=0.5):  # hidden_dim 减小到 32
        super().__init__()

        # 结构简化，减少参数量
        self.layer1 = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim * 2, 5, padding=2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.pooling = MaskedGlobalPooling()

        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2 * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x, mask):
        x = x.permute(0, 2, 1)  # [B, L, C] -> [B, C, L]
        x = self.layer1(x)
        x = self.layer2(x)
        feat = self.pooling(x, mask)
        score = self.head(feat)
        return score


# ==========================================
# 2. 数据处理
# ==========================================
class TaskGroupedDataset(Dataset):
    def __init__(self, data_dir):
        self.tasks = []
        if not os.path.exists(data_dir): return

        pkl_files = glob.glob(os.path.join(data_dir, "*.pkl"))
        raw_groups = {}

        for pkl_path in tqdm(pkl_files, desc=f"Loading {os.path.basename(data_dir)}"):
            try:
                with open(pkl_path, "rb") as f:
                    batch_data = pickle.load(f)
                for item in batch_data:
                    sid = item['sample_id']
                    if sid not in raw_groups: raw_groups[sid] = []

                    feat = item['features']
                    if feat.shape[0] > MAX_SEQ_LEN:
                        feat = feat[:MAX_SEQ_LEN]

                    raw_groups[sid].append({
                        'feat': feat,
                        'label': float(item['is_correct'])
                    })
            except:
                pass

        # 仅保留既有正样本又有负样本的 Task，因为只有这种才能做 Ranking
        for sid, items in raw_groups.items():
            labels = [x['label'] for x in items]
            if 1.0 in labels and 0.0 in labels:
                self.tasks.append(items)

        print(f"[{data_dir}] Loaded {len(self.tasks)} pair-able tasks.")

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        return self.tasks[idx]


def collate_fn_task_relative(batch_tasks):
    all_feats, all_labels, task_indices = [], [], []

    for task_idx, role_list in enumerate(batch_tasks):
        # 1. 统计
        valid_values = []
        for role in role_list:
            f = np.clip(role['feat'], -20.0, 0.0)  # Clip
            valid_mask = f > -19.0
            if valid_mask.any(): valid_values.append(f[valid_mask])

        if valid_values:
            valid_values = np.concatenate(valid_values)
            task_mean = valid_values.mean()
            task_std = valid_values.std() + 1e-5
        else:
            task_mean, task_std = -2.0, 1.0

        # 2. 归一化
        for role in role_list:
            raw_feat = np.clip(role['feat'], -20.0, 0.0)
            norm_feat = (raw_feat - task_mean) / task_std
            all_feats.append(torch.tensor(norm_feat, dtype=torch.float32))
            all_labels.append(role['label'])
            task_indices.append(task_idx)

    feat_padded = pad_sequence(all_feats, batch_first=True, padding_value=0.0)

    # Mask
    B, L, _ = feat_padded.shape
    mask = torch.zeros((B, L), dtype=torch.bool)
    for i, f in enumerate(all_feats):
        mask[i, len(f):] = True

    labels = torch.tensor(all_labels, dtype=torch.float32)
    task_indices = torch.tensor(task_indices, dtype=torch.long)
    return feat_padded, mask, labels, task_indices


# ==========================================
# 3. 核心修改：Pairwise Loss
# ==========================================
def compute_pairwise_ranking_loss(scores, labels, task_indices, margin=0.1):
    """
    在每个 Task 内部，构建 (Positive, Negative) 对。
    Loss = max(0, margin - (Score_Pos - Score_Neg))
    """
    scores = scores.squeeze(1)
    unique_tasks = torch.unique(task_indices)
    total_loss = 0.0
    num_pairs = 0

    for tid in unique_tasks:
        mask = (task_indices == tid)
        t_scores = scores[mask]
        t_labels = labels[mask]

        # 找到正负样本的索引
        pos_indices = torch.nonzero(t_labels == 1.0, as_tuple=True)[0]
        neg_indices = torch.nonzero(t_labels == 0.0, as_tuple=True)[0]

        if len(pos_indices) == 0 or len(neg_indices) == 0:
            continue

        # 策略：对于每个正样本，随机选一个负样本（或者求平均，这里用随机采样更稳健）
        # 为了充分利用数据，我们可以构建网格：所有正 vs 所有负
        # 但为了计算速度，我们把 t_scores 分成两组

        s_pos = t_scores[pos_indices]  # [N_pos]
        s_neg = t_scores[neg_indices]  # [N_neg]

        # 广播减法: [N_pos, 1] - [1, N_neg] = [N_pos, N_neg]
        diff_matrix = s_pos.unsqueeze(1) - s_neg.unsqueeze(0)

        # 我们希望 diff > margin, 即 margin - diff < 0
        loss_matrix = torch.clamp(margin - diff_matrix, min=0.0)

        total_loss += loss_matrix.mean()  # 当前 Task 的平均 Pair Loss
        num_pairs += 1

    if num_pairs > 0:
        return total_loss / num_pairs
    else:
        return torch.tensor(0.0, requires_grad=True).to(scores.device)


# ==========================================
# 4. 训练
# ==========================================
def train_relative():
    train_dataset = TaskGroupedDataset(TRAIN_DIR)
    val_dataset = TaskGroupedDataset(VAL_DIR)

    if len(train_dataset) == 0: return

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn_task_relative, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=collate_fn_task_relative, num_workers=4)

    model = RelativeLogprobCNN(input_dim=20, hidden_dim=32, dropout=DROPOUT).to(device)

    # 【关键】增加 weight_decay 防止过拟合
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    best_val_acc = 0.0

    print("\n=== Starting Pairwise Ranking Training ===")

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [Train]")
        for inputs, mask, labels, task_indices in pbar:
            inputs, mask, labels, task_indices = inputs.to(device), mask.to(device), labels.to(device), task_indices.to(
                device)

            optimizer.zero_grad()
            scores = model(inputs, mask)

            # 使用 Ranking Loss
            loss = compute_pairwise_ranking_loss(scores, labels, task_indices, margin=MARGIN)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader)

        # === Validation ===
        model.eval()
        correct_tasks = 0
        total_tasks = 0

        if len(val_loader) > 0:
            with torch.no_grad():
                for inputs, mask, labels, task_indices in tqdm(val_loader, desc=f"Epoch {epoch + 1} [Val]"):
                    inputs, mask, labels, task_indices = inputs.to(device), mask.to(device), labels.to(
                        device), task_indices.to(device)

                    scores = model(inputs, mask).squeeze(1)

                    unique_tasks = torch.unique(task_indices)
                    for tid in unique_tasks:
                        mask_t = (task_indices == tid)
                        t_scores = scores[mask_t]
                        t_labels = labels[mask_t]

                        # 只有当 Task 里既有1也有0时，Top-1 Acc 才有意义
                        if len(t_scores) > 0 and 1.0 in t_labels:
                            best_idx = torch.argmax(t_scores)
                            if t_labels[best_idx] == 1.0:
                                correct_tasks += 1
                            total_tasks += 1

            val_acc = correct_tasks / total_tasks if total_tasks > 0 else 0.0
        else:
            val_acc = 0.0

        print(f"Epoch {epoch + 1}: Loss={avg_train_loss:.4f} | Val Top-1 Acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Saved Best Model (Acc: {best_val_acc:.4f})")


if __name__ == "__main__":
    train_relative()