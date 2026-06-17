"""
Шаг 2: Разведочный анализ.
- Анализ MCC-категорий на изменение чека во время бурь
- Статистическая проверка значимости (t-test, Mann-Whitney)
- Поведенческие профили по value-группам
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
from joblib import Parallel, delayed
from statsmodels.stats.multitest import multipletests

BASE    = Path(__file__).parent
DATA    = BASE / "data"
PLOTS   = BASE / "plots"
PLOTS.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
N_JOBS = 3


def load_data(sample_frac: float = 0.2):
    """Читает транзакции чанками, берёт sample_frac выборку — не грузит всё в RAM."""
    df_kp  = pd.read_parquet(DATA / "kp.parquet")
    df_mcc = pd.read_parquet(DATA / "mcc.parquet")

    parts = []
    for chunk in pd.read_csv(
        BASE / "transactions.csv",
        dtype={"client": "int32", "mcc": "str", "group": "category", "value": "category"},
        parse_dates=["date"],
        chunksize=500_000,
    ):
        parts.append(chunk.sample(frac=sample_frac, random_state=42))

    df_trans = pd.concat(parts, ignore_index=True)
    df_trans["mcc"] = df_trans["mcc"].str.zfill(4)
    df_trans["amt"] = df_trans["amt"].astype("float32")
    # После concat category-типы могут содержать числовые коды — сбрасываем
    df_trans["group"] = df_trans["group"].astype(str).astype("category")
    df_trans["value"] = df_trans["value"].astype(str).astype("category")
    print(f"  Загружено: {len(df_trans):,} строк (20% выборка)")
    return df_trans, df_kp, df_mcc


def _analyze_one_mcc(mcc_name, subset):
    if len(subset) < 50:
        return None
    storm = subset[subset["is_storm"]]["amt"]
    quiet = subset[~subset["is_storm"]]["amt"]
    if len(storm) < 10 or len(quiet) < 10:
        return None
    _, p_val = stats.mannwhitneyu(storm, quiet, alternative="two-sided")
    pct_change = (storm.mean() - quiet.mean()) / (quiet.mean() + 1e-6) * 100
    return {
        "mcc_name":    mcc_name,
        "n_storm":     len(storm),
        "n_quiet":     len(quiet),
        "avg_quiet":   quiet.mean(),
        "avg_storm":   storm.mean(),
        "pct_change":  pct_change,
        "p_value":     p_val,
        "significant": p_val < 0.05,
    }


def analyze_mcc_categories(df_full):
    """Изменение среднего чека по MCC-категориям в дни бурь."""
    groups = [(name, grp) for name, grp in df_full.groupby("mcc_name")]
    rows = Parallel(n_jobs=N_JOBS)(
        delayed(_analyze_one_mcc)(name, grp) for name, grp in groups
    )
    df = pd.DataFrame([r for r in rows if r is not None]).dropna()
    # Поправка Бенджамини-Хохберга на множественное тестирование
    _, p_adj, _, _ = multipletests(df["p_value"], method="fdr_bh")
    df["p_value_adj"] = p_adj
    df["significant"] = p_adj < 0.05
    df = df[df["significant"]].sort_values("pct_change")
    return df


KNOWN_VALUES = {"survival", "self_realization", "socialization", "money"}

def analyze_value_groups(df_full):
    """Поведенческая экономика: как меняется структура трат по value-группам."""
    result = {}
    for grp, subset in df_full.groupby("value", observed=True):
        if str(grp) not in KNOWN_VALUES:
            continue
        storm = subset[subset["is_storm"]]
        quiet = subset[~subset["is_storm"]]
        result[str(grp)] = {
            "avg_amt_quiet": quiet["amt"].mean(),
            "avg_amt_storm": storm["amt"].mean(),
            "tx_quiet":      len(quiet),
            "tx_storm":      len(storm),
        }
    return pd.DataFrame(result).T


def plot_mcc_shifts(df_mcc_analysis):
    top = pd.concat([df_mcc_analysis.head(12), df_mcc_analysis.tail(12)]).drop_duplicates()
    colors = ["#d62728" if x < 0 else "#2ca02c" for x in top["pct_change"]]
    labels = [s[:45] + "…" if len(s) > 45 else s for s in top["mcc_name"]]

    fig, ax = plt.subplots(figsize=(13, 10))
    bars = ax.barh(labels, top["pct_change"], color=colors)
    ax.axvline(0, color="black", linewidth=1.2)
    ax.set_xlabel("Изменение среднего чека (%)")
    ax.set_title("Реакция MCC-категорий на геомагнитные бури (Kp ≥ 4)\n(только статистически значимые, FDR-adjusted p < 0.05)", fontsize=13)

    for bar in bars:
        w = bar.get_width()
        ax.text(w + (1 if w > 0 else -1), bar.get_y() + bar.get_height() / 2,
                f"{w:+.1f}%", va="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(PLOTS / "mcc_shifts.png", dpi=150)
    plt.close()
    print("Сохранено: plots/mcc_shifts.png")


def plot_kp_timeline(df_kp):
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(df_kp["date"], df_kp["kp_max"], alpha=0.4, color="#4477AA")
    ax.axhline(4, color="red", linestyle="--", linewidth=1, label="Порог бури (Kp=4)")
    ax.set_title("Геомагнитная активность (Kp-индекс) за период исследования")
    ax.set_ylabel("Kp_max")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS / "kp_timeline.png", dpi=150)
    plt.close()
    print("Сохранено: plots/kp_timeline.png")


def plot_value_groups(df_value):
    df_val = df_value[["avg_amt_quiet", "avg_amt_storm"]].copy()
    df_val.columns = ["Спокойные дни", "Дни бурь"]
    df_val.plot(kind="bar", figsize=(8, 5), color=["#4477AA", "#EE6677"])
    plt.title("Средний чек по value-группам: спокойные дни vs дни бурь")
    plt.ylabel("Средний чек (руб.)")
    plt.xlabel("")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(PLOTS / "value_groups.png", dpi=150)
    plt.close()
    print("Сохранено: plots/value_groups.png")


if __name__ == "__main__":
    print("Загружаем данные...")
    df_trans, df_kp, df_mcc = load_data()

    df_full = df_trans.merge(
        df_kp[["date", "kp_max", "kp_avg", "is_storm"]], on="date", how="inner"
    ).merge(df_mcc, on="mcc", how="left")
    df_full["mcc_name"] = df_full["mcc_name"].fillna("MCC " + df_full["mcc"])

    print("\n--- Анализ MCC-категорий ---")
    df_mcc_analysis = analyze_mcc_categories(df_full)
    print(f"Значимых категорий: {len(df_mcc_analysis)}")
    print("\nТОП-10 снижений:")
    print(df_mcc_analysis.head(10)[["mcc_name", "pct_change", "p_value"]].to_string(index=False))
    print("\nТОП-10 роста:")
    print(df_mcc_analysis.tail(10)[["mcc_name", "pct_change", "p_value"]].to_string(index=False))
    df_mcc_analysis.to_csv(DATA / "mcc_analysis.csv", index=False)

    print("\n--- Анализ value-групп (поведенческая экономика) ---")
    df_value = analyze_value_groups(df_full)
    print(df_value.round(2))

    print("\n--- Построение графиков ---")
    plot_kp_timeline(df_kp)
    plot_mcc_shifts(df_mcc_analysis)
    plot_value_groups(df_value)

    print("\nEDA завершён.")
