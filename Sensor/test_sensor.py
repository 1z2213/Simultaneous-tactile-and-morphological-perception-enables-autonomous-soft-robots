import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
import time
from matplotlib.colors import Normalize
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)  # 为所有GPU设置种子
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn.functional as F

# ---------------------- Dataset ----------------------
class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pe, std=0.02)
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        x = x + self.alpha * self.pe[:, :x.size(1)]
        return x

# ---------------------- Dataset ----------------------
class EITSeqDataset():
    def __init__(self, V, D, seq_len=5):
        self.V = V
        self.D = D
        self.seq_len = seq_len

    def __len__(self):
        return len(self.V) - self.seq_len + 1

    def __getitem__(self, idx):
        v_seq = self.V[idx:idx + self.seq_len]
        d_seq = self.D[idx + self.seq_len - 1]
        return v_seq, d_seq

# ---------------------- Model ----------------------
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim 必须能被 num_heads 整除"
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)

        self.W_o = nn.Linear(hidden_dim, hidden_dim)
    def forward(self, x):
        B, T, H = x.shape

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        def split_heads(tensor):
            return tensor.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        Q, K, V = map(split_heads, (Q, K, V))

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, T, H)
        out = self.W_o(out)
        return out

# ---------------- Transformer Encoder Block ----------------
class TransformerEncoderBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, ff_dim=None, dropout=0.3):
        super(TransformerEncoderBlock, self).__init__()
        if ff_dim is None:
            ff_dim = hidden_dim * 4
        self.self_attn = MultiHeadSelfAttention(hidden_dim, num_heads)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, hidden_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        attn_out = self.self_attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


class LSTMTransformerDecoderEIT(nn.Module):
    def __init__(self, input_dim=104, hidden_dim=256, num_layers=2,
                 seq_len=5, n_heads=4, n_transformer_layers=4):
        super(LSTMTransformerDecoderEIT, self).__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, bidirectional=False)

        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderBlock(hidden_dim, n_heads)
            for _ in range(n_transformer_layers)
        ])

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * seq_len, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2209 * 3)
        )
        self.pos_encoding = LearnablePositionalEncoding(hidden_dim, max_len=seq_len)

    def forward(self, x):

        out, _ = self.lstm(x)
        out = self.pos_encoding(out)
        for block in self.transformer_blocks:
            out = block(out)

        concat_out = out.reshape(x.size(0), -1)

        d_pred = self.mlp(concat_out)
        d_pred = d_pred.view(-1, 2209, 3)
        return d_pred

# ---------------------- 测试集数据准备 ----------------------
V_test = scipy.io.loadmat('./Data/V_test/V_quant.mat')['V_quant']
seq_len = 5

# 构建滑动窗口序列: 每个样本包含 seq_len 帧电压
windows = []
for i in range(len(V_test) - seq_len + 1):
    windows.append(V_test[i:i + seq_len])
V_test_seq = np.stack(windows, axis=0)  # 形状 (M, seq_len, 104)

# 转为 Tensor 并创建 DataLoader
Signal_test_tensor = torch.tensor(V_test_seq, dtype=torch.float32)
test_dataset = TensorDataset(Signal_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ---------------------- 模型初始化 ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMTransformerDecoderEIT(input_dim=104, hidden_dim=256, num_layers=2,
                                  seq_len=5, n_heads=4, n_transformer_layers=4).to(device)


model = nn.DataParallel(model, device_ids=[0])
model.load_state_dict(torch.load('best_model_rmse_0.149943.pth', map_location=device))


model.eval()
all_preds = []

torch.cuda.synchronize()

with torch.no_grad():
    for (v_seq_batch,) in test_loader:
        v_seq_batch = v_seq_batch.to(device)
        d_seq_pred = model(v_seq_batch)
        all_preds.append(d_seq_pred.cpu().numpy())

# 拼接所有预测并保存
all_preds = np.concatenate(all_preds, axis=0)
M = all_preds.shape[0]
Pre_numpy = all_preds.reshape(M, -1)
np.savetxt('Pre_quant_lins.txt', Pre_numpy, fmt='%f')

print(f"Saved {M} frames of predictions to '.txt'.")
