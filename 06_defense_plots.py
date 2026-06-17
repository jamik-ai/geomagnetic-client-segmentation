"""
Шаг 6: Графики для защиты НИР.
Генерирует набор высококачественных статичных PNG для презентации.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
import joblib

BASE  = Path(__file__).parent
DATA  = BASE / "data"
PLOTS = BASE / "plots" / "defense"
PLOTS.mkdir(parents=True, exist_ok=True)

SEG_COLORS = {
    "Метеозависимые": "#2ca02c",
    "Сберегающие":    "#d62728",
    "Нейтральные":    "#1f77b4",
}
plt.rcParams.update({
    "font.family":  "DejaVu Sans",
    "font.size":    12,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})


def load():
    client_day  = pd.read_parquet(DATA / "client_day.parquet")
    predictions = pd.read_parquet(DATA / "client_predictions.parquet")
    features    = pd.read_parquet(DATA / "client_features.parquet")
    segments    = pd.read_parquet(DATA / "client_segments.parquet")[["client", "segment"]]
    kp          = pd.read_parquet(DATA / "kp.parquet")
    push        = pd.read_csv(DATA / "push_schedule.csv", parse_dates=["storm_date", "send_date"])
    return client_day, predictions, features, segments, kp, push


# ── 1. Радар-диаграмма профилей сегментов ─────
def plot_radar(features, segments):
    feat_seg = features.merge(segments, on="client")
    cols = ["tx_change", "amt_change", "survival_shift", "food_shift", "fun_shift"]
    labels = ["Транзакции\n(изм.)", "Сумма\n(изм.)", "Товары\n1-й необх.", "Еда", "Развлечения"]
    profiles = feat_seg.groupby("segment")[cols].mean()

    angles = np.linspace(0, 2*np.pi, len(cols), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for seg, color in SEG_COLORS.items():
        if seg not in profiles.index:
            continue
        vals = profiles.loc[seg].values.tolist()
        vals += vals[:1]
        ax.plot(angles, vals, "o-", color=color, linewidth=2, label=seg)
        ax.fill(angles, vals, color=color, alpha=0.12)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_title("Поведенческие профили сегментов\n(среднее изменение в дни бурь)", fontsize=13, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=11)
    plt.tight_layout()
    plt.savefig(PLOTS / "radar_segments.png", dpi=180, bbox_inches="tight")
    plt.close()
    print("Сохранено: defense/radar_segments.png")


# ── 2. KP-индекс + просадка оборота ───────────
def plot_kp_impact(client_day, kp):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=False)

    # KP timeline
    ax1.fill_between(kp["date"], kp["kp_max"], alpha=0.4, color="#4477AA")
    ax1.plot(kp["date"], kp["kp_max"], color="#4477AA", linewidth=0.8)
    storm_kp = kp[kp["is_storm"]]
    ax1.scatter(storm_kp["date"], storm_kp["kp_max"], color="#d62728", s=20, zorder=5, label="Буря (Kp≥4)")
    ax1.axhline(4, color="red", linestyle="--", linewidth=1.2, label="Порог Kp=4")
    ax1.set_ylabel("Kp-индекс")
    ax1.set_title("Геомагнитная активность и влияние на транзакционный оборот", fontsize=13)
    ax1.legend(fontsize=10)

    # Daily avg tx_sum
    daily = client_day.groupby("date").agg(avg_sum=("tx_sum", "mean"), is_storm=("is_storm", "first")).reset_index()
    ax2.fill_between(daily["date"], daily["avg_sum"], alpha=0.3, color="#555")
    ax2.plot(daily["date"], daily["avg_sum"], color="#444", linewidth=0.8)
    storm_days = daily[daily["is_storm"]]
    ax2.scatter(storm_days["date"], storm_days["avg_sum"], color="#d62728", s=20, zorder=5)
    ax2.set_ylabel("Ср. оборот (руб.)")
    ax2.set_xlabel("Дата")

    plt.tight_layout()
    plt.savefig(PLOTS / "kp_impact.png", dpi=180, bbox_inches="tight")
    plt.close()
    print("Сохранено: defense/kp_impact.png")


# ── 3. Сравнение оборота storm vs quiet по сегментам ──
def plot_segment_impact(client_day, segments):
    seg_cd = client_day.merge(segments, on="client")
    rows = []
    for seg in ["Метеозависимые", "Нейтральные", "Сберегающие"]:
        sub = seg_cd[seg_cd["segment"] == seg]
        rows.append({"Сегмент": seg,
                     "Спокойные дни": sub[~sub["is_storm"]]["tx_sum"].mean(),
                     "Дни бурь":      sub[sub["is_storm"]]["tx_sum"].mean()})
    df = pd.DataFrame(rows).set_index("Сегмент")

    x = np.arange(len(df))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - w/2, df["Спокойные дни"], w, label="Спокойные дни", color="#4477AA", alpha=0.85)
    bars2 = ax.bar(x + w/2, df["Дни бурь"],      w, label="Дни бурь",      color="#d62728", alpha=0.85)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=10)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=10, color="#d62728")

    ax.set_xticks(x)
    ax.set_xticklabels(df.index, fontsize=12)
    ax.set_ylabel("Средний оборот на клиента (руб.)")
    ax.set_title("Влияние геомагнитных бурь на оборот по сегментам", fontsize=13)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(PLOTS / "segment_impact.png", dpi=180, bbox_inches="tight")
    plt.close()
    print("Сохранено: defense/segment_impact.png")


# ── 4. Waterfall: механика удержания оборота ──
def plot_waterfall(client_day, predictions):
    storm_mask = client_day["is_storm"]
    avg_quiet  = client_day[~storm_mask]["tx_sum"].mean()
    avg_storm  = client_day[storm_mask]["tx_sum"].mean()
    drop       = avg_quiet - avg_storm
    recovery   = drop * 0.35  # 35% конверсия

    labels = ["Оборот\n(спокойный)", "Просадка\nв бурю", "Удержание\nофферами", "Итог\nбури"]
    bar_bottoms = [avg_quiet - drop * 0.5,
                   avg_storm,
                   avg_storm,
                   avg_storm + recovery - drop * 0.5]
    bar_heights = [drop * 0.5 + drop * 0.5,   # будет перезаписан
                   drop,
                   recovery,
                   drop * 0.5 + recovery]
    colors = ["#4477AA", "#d62728", "#2ca02c", "#4477AA"]

    # Y-ось обрезаем: показываем только верхние ~drop*8 рублей
    margin = drop * 4
    y_min = avg_storm - margin
    y_max = avg_quiet + margin * 0.6

    fig, ax = plt.subplots(figsize=(9, 5))

    # Бар 0 — спокойный оборот (от y_min до avg_quiet)
    ax.bar(0, avg_quiet - y_min, bottom=y_min, color="#4477AA", alpha=0.85, width=0.55)
    ax.text(0, avg_quiet + margin * 0.08, f"{avg_quiet:.0f} ₽",
            ha="center", va="bottom", fontsize=11, fontweight="bold", color="#4477AA")

    # Бар 1 — просадка (от avg_storm до avg_quiet)
    ax.bar(1, drop, bottom=avg_storm, color="#d62728", alpha=0.85, width=0.55)
    ax.text(1, avg_quiet + margin * 0.08, f"−{drop:.0f} ₽",
            ha="center", va="bottom", fontsize=11, fontweight="bold", color="#d62728")
    # Оставшаяся часть бара — заглушка до y_min
    ax.bar(1, avg_storm - y_min, bottom=y_min, color="#d62728", alpha=0.25, width=0.55)

    # Бар 2 — удержание (от avg_storm до avg_storm+recovery)
    ax.bar(2, recovery, bottom=avg_storm, color="#2ca02c", alpha=0.85, width=0.55)
    ax.text(2, avg_storm + recovery + margin * 0.08, f"+{recovery:.0f} ₽",
            ha="center", va="bottom", fontsize=11, fontweight="bold", color="#2ca02c")
    ax.bar(2, avg_storm - y_min, bottom=y_min, color="#2ca02c", alpha=0.15, width=0.55)

    # Бар 3 — итог (от y_min до avg_storm+recovery)
    result = avg_storm + recovery
    ax.bar(3, result - y_min, bottom=y_min, color="#4477AA", alpha=0.65, width=0.55)
    ax.text(3, result + margin * 0.08, f"{result:.0f} ₽",
            ha="center", va="bottom", fontsize=11, fontweight="bold", color="#4477AA")

    # Горизонтальные опорные линии
    ax.axhline(avg_quiet, color="#4477AA", linestyle="--", linewidth=1.2, alpha=0.5,
               label=f"Базовый оборот {avg_quiet:.0f} ₽")
    ax.axhline(avg_storm, color="#d62728", linestyle=":", linewidth=1.0, alpha=0.5,
               label=f"Оборот в бурю {avg_storm:.0f} ₽")

    # Стрелка просадки
    ax.annotate("", xy=(1.28, avg_storm + drop * 0.1), xytext=(1.28, avg_quiet - drop * 0.1),
                arrowprops=dict(arrowstyle="<->", color="#d62728", lw=1.5))
    ax.text(1.35, (avg_quiet + avg_storm) / 2, f"−{drop/avg_quiet*100:.1f}%",
            va="center", fontsize=10, color="#d62728", fontweight="bold")

    ax.set_ylim(y_min, y_max)
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Ср. оборот на клиента (руб.)")
    ax.set_title("Механика удержания оборота через персональные офферы\n"
                 f"(ось Y обрезана; базовый оборот {avg_quiet:.0f} ₽)", fontsize=12)
    ax.legend(fontsize=9, loc="lower right")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    plt.tight_layout()
    plt.savefig(PLOTS / "waterfall.png", dpi=180, bbox_inches="tight")
    plt.close()
    print("Сохранено: defense/waterfall.png")


# ── 5. Матрица ошибок модели ──────────────────
def plot_confusion_heatmap(features, segments):
    model = joblib.load(DATA / "storm_model.pkl")
    le    = joblib.load(DATA / "label_encoder.pkl")
    predictions = pd.read_parquet(DATA / "client_predictions.parquet")

    try:
        cm_labels = list(le.classes_)
        y_true = le.transform(predictions["segment"]) if "segment" in predictions.columns else None
        y_pred = le.transform(predictions["predicted_segment"])

        if y_true is None:
            print("Пропускаем confusion matrix — нет ground truth в predictions.")
            return

        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=cm_labels, yticklabels=cm_labels, ax=ax)
        ax.set_xlabel("Предсказано", fontsize=12)
        ax.set_ylabel("Факт (сегментация)", fontsize=12)
        ax.set_title("Матрица ошибок классификатора", fontsize=13)
        plt.tight_layout()
        plt.savefig(PLOTS / "confusion_matrix.png", dpi=180, bbox_inches="tight")
        plt.close()
        print("Сохранено: defense/confusion_matrix.png")
    except Exception as e:
        print(f"Confusion matrix: {e}")


# ── 6. Распределение кэшбэка ─────────────────
def plot_cashback_dist(push):
    storm_push = push[push["segment"] != "Нейтральные"]
    fig, ax = plt.subplots(figsize=(9, 4))
    for seg, color in SEG_COLORS.items():
        if seg == "Нейтральные":
            continue
        sub = storm_push[storm_push["segment"] == seg]["cashback_pct"]
        ax.hist(sub, bins=20, alpha=0.65, color=color, label=seg, edgecolor="white")

    ax.set_xlabel("Персональный кэшбэк (%)", fontsize=12)
    ax.set_ylabel("Кол-во офферов", fontsize=12)
    ax.set_title("Распределение персонального кэшбэка по сегментам", fontsize=13)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(PLOTS / "cashback_distribution.png", dpi=180, bbox_inches="tight")
    plt.close()
    print("Сохранено: defense/cashback_distribution.png")


# ── 7. Пайплайн исследования (схема) ─────────
def plot_pipeline():
    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 3)
    ax.axis("off")

    steps = [
        (1,   "Данные\nтранзакций\n+ Kp-индекс", "#4477AA"),
        (3.2, "EDA:\nMCC-категории\nvalue-группы", "#6699CC"),
        (5.4, "Сегментация:\nАвтоэнкодер\n+ K-means (k=3)", "#AABB44"),
        (7.6, "Модель:\nXGBoost", "#EE8833"),
        (9.8, "Прогноз\nсегмента\nклиента", "#EE6677"),
        (12,  "Персональный\nкэшбэк перед\nбурей", "#2ca02c"),
    ]

    for x, label, color in steps:
        box = mpatches.FancyBboxPatch((x - 0.9, 0.6), 1.8, 1.8,
                                       boxstyle="round,pad=0.1",
                                       linewidth=1.5, edgecolor=color,
                                       facecolor=color + "33")
        ax.add_patch(box)
        ax.text(x, 1.5, label, ha="center", va="center",
                fontsize=9.5, fontweight="bold", color="#222", linespacing=1.4)

    # Стрелки
    for i in range(len(steps) - 1):
        x1 = steps[i][0] + 0.9
        x2 = steps[i+1][0] - 0.9
        ax.annotate("", xy=(x2, 1.5), xytext=(x1, 1.5),
                    arrowprops=dict(arrowstyle="->", color="#555", lw=1.8))

    ax.set_title("Пайплайн исследования: от данных к персональным рекомендациям",
                 fontsize=13, pad=10)
    plt.tight_layout()
    plt.savefig(PLOTS / "pipeline.png", dpi=180, bbox_inches="tight")
    plt.close()
    print("Сохранено: defense/pipeline.png")


# ── 8. Сводный дашборд (один файл, 4 графика) ─
def plot_summary_dashboard(client_day, segments, features, push):
    seg_cd = client_day.merge(segments, on="client")
    feat_seg = features.merge(segments, on="client")

    fig = plt.figure(figsize=(16, 11))
    gs = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.32)

    # — A. Профили сегментов —
    ax1 = fig.add_subplot(gs[0, 0])
    profile_cols = ["tx_change", "amt_change", "survival_shift", "food_shift", "fun_shift"]
    labels_short = ["Транзакции", "Сумма", "Выживание", "Еда", "Развлечения"]
    profiles = feat_seg.groupby("segment")[profile_cols].mean()
    x = np.arange(len(profile_cols))
    w = 0.25
    for i, (seg, color) in enumerate(SEG_COLORS.items()):
        if seg not in profiles.index:
            continue
        ax1.bar(x + i*w, profiles.loc[seg].values, w, label=seg, color=color, alpha=0.85)
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_xticks(x + w)
    ax1.set_xticklabels(labels_short, fontsize=9, rotation=15)
    ax1.set_title("А. Профили сегментов", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=8)

    # — B. Оборот по сегментам —
    ax2 = fig.add_subplot(gs[0, 1])
    rows = []
    for seg in ["Метеозависимые", "Нейтральные", "Сберегающие"]:
        sub = seg_cd[seg_cd["segment"] == seg]
        rows.append({"seg": seg,
                     "quiet": sub[~sub["is_storm"]]["tx_sum"].mean(),
                     "storm": sub[sub["is_storm"]]["tx_sum"].mean()})
    df_s = pd.DataFrame(rows)
    xp = np.arange(len(df_s))
    ax2.bar(xp - 0.2, df_s["quiet"], 0.38, color="#4477AA", label="Спокойно", alpha=0.85)
    ax2.bar(xp + 0.2, df_s["storm"], 0.38, color="#d62728", label="Буря",     alpha=0.85)
    ax2.set_xticks(xp)
    ax2.set_xticklabels(df_s["seg"], fontsize=9)
    ax2.set_title("Б. Оборот: спокойные дни vs бури", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.set_ylabel("Ср. оборот (руб.)")

    # — C. Распределение сегментов —
    ax3 = fig.add_subplot(gs[1, 0])
    pred = pd.read_parquet(DATA / "client_predictions.parquet")
    seg_counts = pred["predicted_segment"].value_counts()
    colors_list = [SEG_COLORS[s] for s in seg_counts.index]
    wedges, texts, autotexts = ax3.pie(
        seg_counts.values, labels=seg_counts.index,
        colors=colors_list, autopct="%1.0f%%",
        startangle=90, pctdistance=0.55, labeldistance=1.15,
    )
    for t in autotexts:
        t.set_fontsize(12); t.set_fontweight("bold")
    for t in texts:
        t.set_fontsize(9)
    ax3.set_title("В. Распределение сегментов\n(предсказания модели)", fontsize=11, fontweight="bold")

    # — D. Кэшбэк по категориям —
    ax4 = fig.add_subplot(gs[1, 1])
    cat_avg = (push[push["segment"] != "Нейтральные"]
               .groupby("category")["cashback_pct"].mean()
               .sort_values(ascending=True))
    bars = ax4.barh(cat_avg.index, cat_avg.values, color="#2ca02c", alpha=0.8)
    ax4.set_xlabel("Средний кэшбэк (%)")
    ax4.set_title("Г. Персональный кэшбэк по категориям", fontsize=11, fontweight="bold")
    for bar in bars:
        ax4.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                 f"{bar.get_width():.1f}%", va="center", fontsize=9)

    fig.suptitle(
        "Сегментация банковских клиентов по чувствительности к геомагнитным бурям",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.savefig(PLOTS / "summary_dashboard.png", dpi=180, bbox_inches="tight")
    plt.close()
    print("Сохранено: defense/summary_dashboard.png")


# ── main ──────────────────────────────────────
if __name__ == "__main__":
    print("Загрузка данных...")
    client_day, predictions, features, segments, kp, push = load()

    print("\nГенерируем графики для защиты...")
    plot_radar(features, segments)
    plot_kp_impact(client_day, kp)
    plot_segment_impact(client_day, segments)
    plot_waterfall(client_day, predictions)
    plot_confusion_heatmap(features, segments)
    plot_cashback_dist(push)
    plot_pipeline()
    plot_summary_dashboard(client_day, segments, features, push)

    print(f"\nВсе графики сохранены в: plots/defense/")
    print("Файлы:")
    for f in sorted(PLOTS.glob("*.png")):
        print(f"  {f.name}")
