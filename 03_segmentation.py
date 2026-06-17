"""
Шаг 3: Сегментация клиентов.

Baseline: K-means на вручную сконструированных признаках.
Advanced:  Автоэнкодер (PyTorch) — обучает латентное представление поведения,
           затем K-means кластеризация в латентном пространстве.
Интерпретация через поведенческую экономику.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from joblib import Parallel, delayed
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

BASE  = Path(__file__).parent
DATA  = BASE / "data"
PLOTS = BASE / "plots"
PLOTS.mkdir(exist_ok=True)

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
N_CLUSTERS = 3
SEED       = 42
N_JOBS     = 3
torch.manual_seed(SEED)
torch.set_num_threads(N_JOBS)
torch.set_num_interop_threads(N_JOBS)
np.random.seed(SEED)


# ──────────────────────────────────────────────
# Признаки для кластеризации
# ──────────────────────────────────────────────
FEATURE_COLS = [
    "tx_change",          # изменение частоты транзакций (главный признак)
    "amt_change",         # изменение суммы трат
    "survival_shift",     # сдвиг доли товаров первой необходимости
    "food_shift",         # сдвиг доли еды
    "fun_shift",          # сдвиг доли развлечений
    "avg_mcc_unique",     # разнообразие категорий
    "tx_per_day_quiet",   # базовая активность
]


# ──────────────────────────────────────────────
# Автоэнкодер
# ──────────────────────────────────────────────
class BehaviorAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(), nn.BatchNorm1d(32),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16), nn.ReLU(),
            nn.Linear(16, 32), nn.ReLU(), nn.BatchNorm1d(32),
            nn.Linear(32, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


def train_autoencoder(X_scaled: np.ndarray, latent_dim: int = 8,
                      epochs: int = 100, batch_size: int = 512):
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE)
    loader   = DataLoader(TensorDataset(X_tensor), batch_size=batch_size, shuffle=True, drop_last=True)
    model    = BehaviorAutoencoder(X_scaled.shape[1], latent_dim).to(DEVICE)
    opt      = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn  = nn.MSELoss()

    history = []
    model.train()
    for epoch in range(epochs):
        total = 0.0
        for (batch,) in loader:
            recon, _ = model(batch)
            loss = loss_fn(recon, batch)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        avg = total / len(loader)
        history.append(avg)
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{epochs}  loss={avg:.6f}")

    torch.save(model.state_dict(), DATA / "autoencoder.pt")

    model.eval()
    with torch.no_grad():
        _, Z = model(torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE))
    return Z.cpu().numpy(), history


# ──────────────────────────────────────────────
# Утилиты кластеризации
# ──────────────────────────────────────────────
def assign_segment_names(features: pd.DataFrame, labels: np.ndarray) -> dict:
    """Сопоставляем кластеры с названиями по знаку tx_change."""
    df = features.copy()
    df["cluster"] = labels
    centers = df.groupby("cluster")["tx_change"].mean()
    sorted_c = centers.sort_values()
    mapping = {sorted_c.index[0]: "Сберегающие", sorted_c.index[-1]: "Метеозависимые"}
    for c in centers.index:
        if c not in mapping:
            mapping[c] = "Нейтральные"
    return mapping


# ──────────────────────────────────────────────
# Визуализация
# ──────────────────────────────────────────────
def plot_clusters(X_2d, labels_or_seg, title, fname, is_series=False):
    df = pd.DataFrame(X_2d, columns=["PC1", "PC2"])
    if is_series:
        df["segment"] = labels_or_seg.values
    else:
        df["segment"] = labels_or_seg
    palette = {"Метеозависимые": "#2ca02c", "Сберегающие": "#d62728", "Нейтральные": "#1f77b4"}
    plt.figure(figsize=(9, 7))
    sns.scatterplot(data=df, x="PC1", y="PC2", hue="segment", palette=palette,
                    alpha=0.5, s=15, linewidth=0)
    plt.title(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(PLOTS / fname, dpi=150)
    plt.close()
    print(f"Сохранено: plots/{fname}")


def plot_segment_profiles(features, fname="segment_profiles.png"):
    cols = ["tx_change", "amt_change", "survival_shift", "food_shift", "fun_shift"]
    df = features.groupby("segment")[cols].mean()
    fig, axes = plt.subplots(1, len(cols), figsize=(16, 5))
    colors = {"Метеозависимые": "#2ca02c", "Сберегающие": "#d62728", "Нейтральные": "#1f77b4"}
    for ax, col in zip(axes, cols):
        vals = df[col]
        ax.barh(vals.index, vals.values, color=[colors.get(s, "gray") for s in vals.index])
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(col, fontsize=10)
    plt.suptitle("Поведенческие профили сегментов (среднее изменение в дни бурь)", fontsize=12)
    plt.tight_layout()
    plt.savefig(PLOTS / fname, dpi=150)
    plt.close()
    print(f"Сохранено: plots/{fname}")


def plot_ae_loss(history):
    plt.figure(figsize=(7, 4))
    plt.plot(history)
    plt.xlabel("Epoch"); plt.ylabel("MSE Loss")
    plt.title("Обучение автоэнкодера")
    plt.tight_layout()
    plt.savefig(PLOTS / "ae_loss.png", dpi=150)
    plt.close()


# ──────────────────────────────────────────────
# Главный пайплайн
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Устройство: {DEVICE}\n")

    features = pd.read_parquet(DATA / "client_features.parquet")
    print(f"Клиентов: {len(features):,}")

    X = features[FEATURE_COLS].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=SEED)
    X_2d = pca.fit_transform(X_scaled)

    # ── Baseline: K-means ──
    print("\n=== Baseline: K-means ===")
    km_base = KMeans(n_clusters=N_CLUSTERS, random_state=SEED, n_init=10)
    labels_base = km_base.fit_predict(X_scaled)
    mapping_base = assign_segment_names(features, labels_base)
    features["segment_base"] = pd.Series(labels_base).map(mapping_base).values
    print(features["segment_base"].value_counts().to_string())
    sil_base = silhouette_score(X_scaled, labels_base)
    db_base  = davies_bouldin_score(X_scaled, labels_base)
    print(f"Silhouette={sil_base:.3f}  Davies-Bouldin={db_base:.3f}")
    plot_clusters(X_2d, features["segment_base"],
                  "Сегментация клиентов (K-means baseline)", "clusters_baseline.png", is_series=True)

    # ── Advanced: Автоэнкодер + K-means ──
    print("\n=== Автоэнкодер + K-means ===")
    Z, ae_history = train_autoencoder(X_scaled, latent_dim=8, epochs=100)
    plot_ae_loss(ae_history)

    Z_scaler = StandardScaler()
    Z = Z_scaler.fit_transform(Z)

    km_ae = KMeans(n_clusters=N_CLUSTERS, random_state=SEED, n_init=10)
    labels_ae = km_ae.fit_predict(Z)
    mapping_ae = assign_segment_names(features, labels_ae)
    features["segment"] = pd.Series(labels_ae).map(mapping_ae).values
    print(features["segment"].value_counts().to_string())
    sil_ae = silhouette_score(Z, labels_ae)
    db_ae  = davies_bouldin_score(Z, labels_ae)
    print(f"Silhouette={sil_ae:.3f}  Davies-Bouldin={db_ae:.3f}")

    Z_2d = PCA(n_components=2, random_state=SEED).fit_transform(Z)
    plot_clusters(Z_2d, features["segment"],
                  "Сегментация клиентов (Автоэнкодер + K-means)", "clusters_autoencoder.png",
                  is_series=True)

    # ── Профили сегментов ──
    plot_segment_profiles(features)

    # ── Сохраняем результат ──
    features.to_parquet(DATA / "client_segments.parquet", index=False)
    print("\nРезультаты сохранены: data/client_segments.parquet")

    print("\n=== Сравнение методов ===")
    print(f"{'Метод':<25} {'Silhouette':>12} {'Davies-Bouldin':>15}")
    print(f"{'K-means (baseline)':<25} {sil_base:>12.3f} {db_base:>15.3f}")
    print(f"{'Автоэнкодер + K-means':<25} {sil_ae:>12.3f} {db_ae:>15.3f}")
    print("\nШаг 3 завершён.")
