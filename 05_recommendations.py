"""
Шаг 5: Персональные рекомендации с повышенным кэшбэком.

Логика: прогноз геомагнитной бури + предсказанный сегмент (из шага 4)
→ триггерный оффер с кэшбэком, масштабированным по уверенности модели.

Чем увереннее модель в storm-сегменте клиента, тем выше кэшбэк.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from joblib import Parallel, delayed

BASE   = Path(__file__).parent
DATA   = BASE / "data"
PLOTS  = BASE / "plots"
PLOTS.mkdir(exist_ok=True)
N_JOBS = 3

# ──────────────────────────────────────────────
# Базовые правила офферов по сегменту
# ──────────────────────────────────────────────
# cashback_base — базовый %, cashback_max — максимум при proba=1.0
OFFER_RULES = {
    # Метеозависимые: повышают траты во время бурь (тревожное/компенсаторное потребление)
    # → направляем в домашние каналы, монетизируем их активность
    "Метеозависимые": [
        {
            "trigger_days_before": 2,
            "category": "Доставка еды",
            "cashback_base": 7,
            "cashback_max":  15,
            "rationale": "Компенсаторное потребление → направляем в удобный домашний канал",
        },
        {
            "trigger_days_before": 1,
            "category": "Аптеки",
            "cashback_base": 5,
            "cashback_max":  12,
            "rationale": "Тревожное потребление: превентивная забота о здоровье",
        },
        {
            "trigger_days_before": 3,
            "category": "Стриминг / кино",
            "cashback_base": 5,
            "cashback_max":  10,
            "rationale": "Снижаем стресс через развлечения дома, без физических усилий",
        },
    ],
    # Сберегающие: резко снижают все траты (избегают физических действий, сидят дома)
    # → удерживаем оборот переводом в онлайн/доставку, не зовём в физический магазин
    "Сберегающие": [
        {
            "trigger_days_before": 2,
            "category": "Доставка продуктов",
            "cashback_base": 5,
            "cashback_max":  12,
            "rationale": "Клиент не идёт в магазин → переводим покупку продуктов в доставку",
        },
        {
            "trigger_days_before": 1,
            "category": "Доставка еды",
            "cashback_base": 8,
            "cashback_max":  15,
            "rationale": "Удерживаем ресторанный оборот через доставку вместо похода в кафе",
        },
        {
            "trigger_days_before": 3,
            "category": "Онлайн-кинотеатры",
            "cashback_base": 3,
            "cashback_max":  8,
            "rationale": "Монетизируем пассивный отдых дома: минимум усилий для клиента",
        },
    ],
    "Нейтральные": [
        {
            "trigger_days_before": 1,
            "category": "Рестораны",
            "cashback_base": 3,
            "cashback_max":  3,
            "rationale": "Базовое удержание, нет специфической реакции на бури",
        },
    ],
}


def _cashback_pct(base: int, max_cb: int, proba: float) -> int:
    """Кэшбэк интерполируется между base и max по уверенности модели."""
    return round(base + (max_cb - base) * proba)


def _push_for_client(client_id, segment, proba, storm_days):
    rules = OFFER_RULES.get(segment, [])
    records = []
    for storm_date in storm_days:
        for rule in rules:
            cb = _cashback_pct(rule["cashback_base"], rule["cashback_max"], proba)
            send_date = pd.Timestamp(storm_date) - pd.Timedelta(days=rule["trigger_days_before"])
            records.append({
                "client":        client_id,
                "segment":       segment,
                "storm_proba":   round(proba, 3),
                "storm_date":    storm_date,
                "send_date":     send_date,
                "days_before":   rule["trigger_days_before"],
                "category":      rule["category"],
                "cashback_pct":  cb,
                "rationale":     rule["rationale"],
            })
    return records


def generate_push_schedule(kp_forecast: pd.DataFrame,
                            client_predictions: pd.DataFrame) -> pd.DataFrame:
    storm_days = kp_forecast[kp_forecast["kp_max_forecast"] >= 4.0]["date"].tolist()

    rows_list = Parallel(n_jobs=N_JOBS)(
        delayed(_push_for_client)(
            row["client"], row["predicted_segment"], row["storm_proba"], storm_days
        )
        for _, row in client_predictions.iterrows()
    )
    records = [r for batch in rows_list for r in batch]
    return pd.DataFrame(records)


def estimate_business_effect(push_schedule: pd.DataFrame,
                             client_day: pd.DataFrame,
                             predictions: pd.DataFrame) -> dict:
    n_total  = predictions["client"].nunique()
    n_pushed = push_schedule["client"].nunique()

    storm_mask   = client_day["is_storm"]
    avg_storm    = client_day[storm_mask]["tx_sum"].mean()
    avg_quiet    = client_day[~storm_mask]["tx_sum"].mean()
    drop_pct     = (avg_storm - avg_quiet) / (avg_quiet + 1e-6) * 100

    high_conf = predictions[predictions["storm_proba"] >= 0.75]

    return {
        "Клиентов всего":                          n_total,
        "Клиентов с персональным пушем":           n_pushed,
        "Охват (%)":                               round(n_pushed / n_total * 100, 1),
        "Просадка оборота в бури (%)":             round(drop_pct, 2),
        "Высокая уверенность модели (≥75%)":       len(high_conf),
        "Средний кэшбэк (Метеозависимые)":         _avg_cb(push_schedule, "Метеозависимые"),
        "Средний кэшбэк (Сберегающие)":            _avg_cb(push_schedule, "Сберегающие"),
    }


def _avg_cb(push_schedule, segment):
    sub = push_schedule[push_schedule["segment"] == segment]
    if sub.empty:
        return "—"
    return f"{sub['cashback_pct'].mean():.1f}%"


def plot_cashback_distribution(push_schedule: pd.DataFrame):
    storm_segs = ["Метеозависимые", "Сберегающие"]
    df = push_schedule[push_schedule["segment"].isin(storm_segs)]
    if df.empty:
        return

    plt.figure(figsize=(9, 5))
    for seg in storm_segs:
        sub = df[df["segment"] == seg]["cashback_pct"]
        plt.hist(sub, bins=20, alpha=0.6, label=seg)
    plt.xlabel("Кэшбэк (%)")
    plt.ylabel("Кол-во офферов")
    plt.title("Распределение персонального кэшбэка по сегментам")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS / "cashback_distribution.png", dpi=150)
    plt.close()
    print("Сохранено: plots/cashback_distribution.png")


def plot_offer_calendar(push_schedule: pd.DataFrame):
    daily = push_schedule.groupby(["send_date", "segment"])["client"].nunique().reset_index()
    daily.columns = ["date", "segment", "n_clients"]
    daily = daily.sort_values("date")
    pivot = daily.pivot_table(index="date", columns="segment",
                              values="n_clients", fill_value=0)

    colors = {"Метеозависимые": "#2ca02c", "Сберегающие": "#d62728", "Нейтральные": "#1f77b4"}
    fig, ax = plt.subplots(figsize=(14, 5))
    bottom = np.zeros(len(pivot))
    for seg in pivot.columns:
        ax.bar(pivot.index, pivot[seg], bottom=bottom,
               color=colors.get(seg, "gray"), label=seg, alpha=0.85)
        bottom += pivot[seg].values

    ax.set_title("Расписание триггерных пушей по дням и сегментам")
    ax.set_xlabel("Дата")
    ax.set_ylabel("Кол-во клиентов с пушем")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS / "push_calendar.png", dpi=150)
    plt.close()
    print("Сохранено: plots/push_calendar.png")


def print_example_pushes(push_schedule: pd.DataFrame, n: int = 8):
    print("\n" + "="*70)
    print("ПРИМЕРЫ ПЕРСОНАЛЬНЫХ PUSH-УВЕДОМЛЕНИЙ")
    print("="*70)
    sample = (push_schedule[push_schedule["segment"] != "Нейтральные"]
              .drop_duplicates(["segment", "category"])
              .head(n))
    for _, row in sample.iterrows():
        print(f"\n  Клиент:    {row['client']}")
        print(f"  Сегмент:   {row['segment']}  (уверенность модели: {row['storm_proba']:.0%})")
        print(f"  Отправить: {row['send_date'].date()} "
              f"(за {row['days_before']} дн. до бури {row['storm_date'].date()})")
        print(f"  Категория: {row['category']}")
        print(f"  Кэшбэк:    {row['cashback_pct']}%")
        print(f"  Логика:    {row['rationale']}")


if __name__ == "__main__":
    predictions = pd.read_parquet(DATA / "client_predictions.parquet")
    client_day  = pd.read_parquet(DATA / "client_day.parquet")

    print(f"Клиентов для рекомендаций: {len(predictions):,}")
    print(predictions["predicted_segment"].value_counts().to_string())

    # Симулируем прогноз Kp на 30 дней вперёд
    last_date = client_day["date"].max()
    forecast_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=30)
    np.random.seed(42)
    kp_forecast = pd.DataFrame({
        "date": forecast_dates,
        "kp_max_forecast": np.random.exponential(2, 30).clip(0, 9),
    })
    kp_forecast.loc[kp_forecast["kp_max_forecast"] >= 4, "kp_max_forecast"] *= 1.2
    kp_forecast["kp_max_forecast"] = kp_forecast["kp_max_forecast"].clip(0, 9)
    n_storm = (kp_forecast["kp_max_forecast"] >= 4).sum()
    print(f"\nПрогнозируемых дней бурь: {n_storm} из 30")

    print("Генерируем расписание пушей...")
    push_schedule = generate_push_schedule(kp_forecast, predictions)
    print(f"Записей в расписании: {len(push_schedule):,}")

    print_example_pushes(push_schedule)

    print("\n=== Оценка бизнес-эффекта ===")
    effect = estimate_business_effect(push_schedule, client_day, predictions)
    for k, v in effect.items():
        print(f"  {k:<45} {v}")

    push_schedule.to_csv(DATA / "push_schedule.csv", index=False)
    plot_cashback_distribution(push_schedule)
    plot_offer_calendar(push_schedule)

    print("\nШаг 5 завершён.")
