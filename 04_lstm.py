"""
Шаг 4: Предиктивная модель (LSTM).

Задача: по временному ряду транзакционного поведения клиента
за N дней до бури предсказать изменение его активности.

Архитектура:
  - Вход: последовательность дневных агрегаций (tx_count, tx_sum, kp_avg, ...)
  - LSTM encoder → линейная голова
  - Таргет: tx_change в день бури (регрессия) или сегмент (классификация)

Attention-механизм позволяет интерпретировать, какие дни
до бури наиболее информативны.
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, classification_report
from sklearn.utils.class_weight import compute_class_weight
from joblib import Parallel, delayed

BASE   = Path(__file__).parent
DATA   = BASE / "data"
PLOTS  = BASE / "plots"
PLOTS.mkdir(exist_ok=True)

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LEN   = 14   # смотрим 14 дней до бури
FEAT_COLS = ["tx_count", "tx_sum", "tx_mean", "mcc_unique",
             "share_survival", "share_food", "share_fun", "kp_avg"]
SEED      = 42
N_JOBS    = 3
torch.manual_seed(SEED)
torch.set_num_threads(N_JOBS)
torch.set_num_interop_threads(N_JOBS)
np.random.seed(SEED)


# ──────────────────────────────────────────────
# Подготовка последовательностей (параллельно)
# ──────────────────────────────────────────────
def _build_seqs_batch(cd_batch: pd.DataFrame, seg_map: dict, storm_timestamps,
                      label_enc: dict, feat_cols: list, seq_len: int):
    """Обрабатывает один батч клиентов — запускается в отдельном процессе."""
    tx_idx = feat_cols.index("tx_count")
    day_ns = np.timedelta64(1, 'D')

    X_list, y_reg_list, y_cls_list, client_list = [], [], [], []
    for client_id, cdf_raw in cd_batch.groupby("client"):
        if client_id not in seg_map:
            continue
        seg_label = label_enc.get(seg_map[client_id], 1)

        cdf_s     = cdf_raw.sort_values("date")
        dates_np  = cdf_s["date"].values          # numpy datetime64[ns]
        feats_np  = cdf_s[feat_cols].values.astype(np.float32)

        for storm_ts in storm_timestamps:
            end_ts   = storm_ts - day_ns
            start_ts = storm_ts - seq_len * day_ns
            mask     = (dates_np >= start_ts) & (dates_np <= end_ts)
            win      = feats_np[mask]

            if len(win) < seq_len // 2:
                continue
            if len(win) < seq_len:
                pad = np.zeros((seq_len - len(win), len(feat_cols)), dtype=np.float32)
                win = np.vstack([pad, win])
            seq = win[-seq_len:]

            storm_mask = dates_np == storm_ts
            if storm_mask.any():
                storm_tx = feats_np[storm_mask][0, tx_idx]
                pre_tx   = seq[:, tx_idx]
                pre_tx   = pre_tx[pre_tx > 0]
                avg_pre  = pre_tx.mean() if len(pre_tx) else 0.0
                y_val    = (storm_tx - avg_pre) / (avg_pre + 1e-6)
            else:
                y_val = 0.0

            X_list.append(seq)
            y_reg_list.append(float(y_val))
            y_cls_list.append(seg_label)
            client_list.append(client_id)
    return X_list, y_reg_list, y_cls_list, client_list


TOP_STORMS = 50  # берём только самые сильные бури

def build_sequences(client_day: pd.DataFrame, segments: pd.DataFrame,
                    label_enc: dict, seq_len: int = SEQ_LEN):
    client_day = client_day.sort_values(["client", "date"])
    seg_map    = segments.set_index("client")["segment"].to_dict()

    # топ-50 дней по kp_max
    storm_days = (client_day[client_day["is_storm"]]
                  .groupby("date")["kp_max"].max()
                  .nlargest(TOP_STORMS)
                  .index)
    storm_timestamps = storm_days.values.astype("datetime64[ns]")

    clients = client_day["client"].unique()
    batches = np.array_split(clients, N_JOBS)
    print(f"  {len(clients):,} клиентов × {len(storm_timestamps)} дней бурь → {N_JOBS} батча...", flush=True)

    results = Parallel(n_jobs=N_JOBS)(
        delayed(_build_seqs_batch)(
            client_day[client_day["client"].isin(b)],
            seg_map, storm_timestamps, label_enc, FEAT_COLS, seq_len
        )
        for b in batches
    )

    X_list, y_reg_list, y_cls_list, client_list = [], [], [], []
    for r in results:
        X_list.extend(r[0]); y_reg_list.extend(r[1])
        y_cls_list.extend(r[2]); client_list.extend(r[3])

    X  = np.array(X_list, dtype=np.float32)
    yr = np.array(y_reg_list, dtype=np.float32)
    yc = np.array(y_cls_list, dtype=np.int64)
    return X, yr, yc, np.array(client_list)


# ──────────────────────────────────────────────
# Модель: LSTM с Attention
# ──────────────────────────────────────────────
class AttentionLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 n_layers: int = 2, n_classes: int = 3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers,
                            batch_first=True, dropout=0.2)
        # Attention: скалярный вес для каждого шага последовательности
        self.attn = nn.Linear(hidden_dim, 1)
        self.fc_cls = nn.Linear(hidden_dim, n_classes)
        self.fc_reg = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)          # (B, T, H)
        attn_w = torch.softmax(self.attn(out), dim=1)   # (B, T, 1)
        context = (attn_w * out).sum(dim=1)              # (B, H)
        return self.fc_cls(context), self.fc_reg(context).squeeze(-1), attn_w.squeeze(-1)


# ──────────────────────────────────────────────
# Обучение
# ──────────────────────────────────────────────
def train(model, loader, opt, cls_loss_fn, reg_loss_fn, alpha=0.5):
    model.train()
    total = 0.0
    for X_b, yr_b, yc_b in loader:
        X_b, yr_b, yc_b = X_b.to(DEVICE), yr_b.to(DEVICE), yc_b.to(DEVICE)
        logits, pred_reg, _ = model(X_b)
        loss = alpha * cls_loss_fn(logits, yc_b) + (1 - alpha) * reg_loss_fn(pred_reg, yr_b)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += loss.item()
    return total / len(loader)


def evaluate(model, loader):
    model.eval()
    all_logits, all_reg, all_yc, all_yr = [], [], [], []
    with torch.no_grad():
        for X_b, yr_b, yc_b in loader:
            logits, pred_reg, _ = model(X_b.to(DEVICE))
            all_logits.append(logits.cpu())
            all_reg.append(pred_reg.cpu())
            all_yc.append(yc_b)
            all_yr.append(yr_b)
    logits = torch.cat(all_logits)
    preds_cls = logits.argmax(dim=1).numpy()
    preds_reg = torch.cat(all_reg).numpy()
    y_cls = torch.cat(all_yc).numpy()
    y_reg = torch.cat(all_yr).numpy()
    return preds_cls, preds_reg, y_cls, y_reg


# ──────────────────────────────────────────────
# Визуализация attention
# ──────────────────────────────────────────────
def plot_attention(model, X_sample: np.ndarray, title: str = "Attention по шагам"):
    model.eval()
    x = torch.tensor(X_sample[np.newaxis], dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        _, _, attn_w = model(x)
    weights = attn_w[0].cpu().numpy()
    days_before = list(range(-SEQ_LEN, 0))

    plt.figure(figsize=(9, 4))
    plt.bar(days_before, weights, color="#4477AA")
    plt.xlabel("Дней до бури")
    plt.ylabel("Attention weight")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(PLOTS / "attention_weights.png", dpi=150)
    plt.close()
    print("Сохранено: plots/attention_weights.png")


# ──────────────────────────────────────────────
# Главный пайплайн
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Устройство: {DEVICE}\n")

    client_day = pd.read_parquet(DATA / "client_day.parquet")
    segments   = pd.read_parquet(DATA / "client_segments.parquet")[["client", "segment"]]

    print("Строим последовательности (это может занять несколько минут)...")
    # Для демо берём выборку клиентов; убрать .sample для полного датасета
    sample_clients = segments["client"].sample(n=min(5000, len(segments)), random_state=SEED)
    cd_sample  = client_day[client_day["client"].isin(sample_clients)]
    seg_sample = segments[segments["client"].isin(sample_clients)]

    # Определяем классы из данных (не хардкод)
    seg_names = sorted(seg_sample["segment"].unique())
    n_classes = len(seg_names)
    label_enc = {name: i for i, name in enumerate(seg_names)}
    print(f"Сегментов: {n_classes} → {seg_names}")

    X, yr, yc, clients = build_sequences(cd_sample, seg_sample, label_enc)
    print(f"Последовательностей: {len(X):,}  |  shape: {X.shape}")

    # Нормализация по признакам
    n, t, f = X.shape
    X_2d = X.reshape(-1, f)
    scaler = StandardScaler()
    X_2d = scaler.fit_transform(X_2d)
    X = X_2d.reshape(n, t, f).astype(np.float32)

    # Train/val split
    idx_tr, idx_val = train_test_split(np.arange(len(X)), test_size=0.2,
                                       random_state=SEED, stratify=yc)
    to_tensor = lambda i: TensorDataset(
        torch.tensor(X[i]), torch.tensor(yr[i]), torch.tensor(yc[i])
    )
    loader_tr  = DataLoader(to_tensor(idx_tr),  batch_size=512, shuffle=True,  num_workers=0)
    loader_val = DataLoader(to_tensor(idx_val), batch_size=512, num_workers=0)

    model = AttentionLSTM(input_dim=len(FEAT_COLS), hidden_dim=64,
                          n_layers=2, n_classes=n_classes).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=7, gamma=0.5)

    cls_weights = compute_class_weight("balanced", classes=np.arange(n_classes), y=yc[idx_tr])
    print(f"Веса классов: { {seg_names[i]: f'{w:.3f}' for i, w in enumerate(cls_weights)} }")
    cls_fn = nn.CrossEntropyLoss(weight=torch.tensor(cls_weights, dtype=torch.float32).to(DEVICE))
    reg_fn = nn.MSELoss()

    N_EPOCHS = 20
    print(f"\nОбучение LSTM ({N_EPOCHS} эпох)...")
    history = []
    for epoch in range(N_EPOCHS):
        loss = train(model, loader_tr, opt, cls_fn, reg_fn)
        scheduler.step()
        history.append(loss)
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{N_EPOCHS}  loss={loss:.4f}")

    torch.save(model.state_dict(), DATA / "lstm.pt")

    # Оценка
    preds_cls, preds_reg, y_cls_val, y_reg_val = evaluate(model, loader_val)
    print("\n=== Классификация сегментов ===")
    print(classification_report(y_cls_val, preds_cls, target_names=seg_names))
    print(f"MAE (регрессия tx_change): {mean_absolute_error(y_reg_val, preds_reg):.4f}")

    # Attention для среднего примера
    plot_attention(model, X[idx_val[0]], "Важность дней до бури (Attention LSTM)")

    # Loss curve
    plt.figure(figsize=(7, 4))
    plt.plot(history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Обучение LSTM")
    plt.tight_layout()
    plt.savefig(PLOTS / "lstm_loss.png", dpi=150)
    plt.close()

    print("\nМодель сохранена: data/lstm.pt")
    print("Шаг 4 завершён.")
