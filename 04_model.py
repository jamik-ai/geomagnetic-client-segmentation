"""
Шаг 4: Предиктивная модель storm-сегмента (честная, без leakage).

Три группы признаков:
  1. Past-storm track record  — поведение клиента в РАННИХ бурях (первые 60% по дате).
     Используем как «историческую память» → не leakage, потому что это прошлые события.
  2. Rich quiet-day features  — статистика из ТИХИХ дней: CV, волатильность,
     концентрация трат, персентили. Данные бурь не используются.
  3. Pre-storm window         — агрегаты за 14 дней ДО каждой бури (без дня бури).

Временной сплит: модель обучается предсказывать сегмент по признакам,
которые реально были бы доступны ДО наступления бури.
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
import joblib

BASE  = Path(__file__).parent
DATA  = BASE / "data"
PLOTS = BASE / "plots"
PLOTS.mkdir(exist_ok=True)

SEQ_LEN    = 14
TOP_STORMS = 50
PAST_RATIO = 0.6   # первые 60% бурь → признаки track record
SEED       = 42
N_JOBS     = 3
np.random.seed(SEED)

FEAT_COLS = ["tx_count", "tx_sum", "tx_mean", "mcc_unique",
             "share_survival", "share_food", "share_fun", "kp_avg"]


# ──────────────────────────────────────────────
# 1. Past-storm track record
# ──────────────────────────────────────────────
def build_past_storm_features(client_day: pd.DataFrame,
                               storm_dates_sorted: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Для каждого клиента считаем среднее поведение в РАННИХ бурях
    (первые PAST_RATIO бурь по времени).
    Сравниваем с тихим базисом → получаем storm-sensitivity без leakage.
    """
    n_past = max(1, int(len(storm_dates_sorted) * PAST_RATIO))
    past_storms = set(storm_dates_sorted[:n_past])

    records = []
    for client_id, cdf in client_day.groupby("client"):
        storm_days  = cdf[cdf["date"].isin(past_storms)]
        quiet_days  = cdf[~cdf["is_storm"]]

        if len(storm_days) < 3 or len(quiet_days) < 10:
            continue

        q_tx  = quiet_days["tx_count"].mean() + 1e-6
        q_amt = quiet_days["tx_sum"].mean()   + 1e-6

        rec = {
            "client": client_id,
            # Storm-sensitivity из прошлых бурь
            "past_tx_change":       (storm_days["tx_count"].mean() - q_tx) / q_tx,
            "past_amt_change":      (storm_days["tx_sum"].mean()   - q_amt) / q_amt,
            "past_food_shift":      storm_days["share_food"].mean() - quiet_days["share_food"].mean(),
            "past_surv_shift":      storm_days["share_survival"].mean() - quiet_days["share_survival"].mean(),
            "past_fun_shift":       storm_days["share_fun"].mean() - quiet_days["share_fun"].mean(),
            "past_mcc_shift":       storm_days["mcc_unique"].mean() - quiet_days["mcc_unique"].mean(),
            "past_n_storms":        len(storm_days),
            # Вариативность реакции (насколько стабильна реакция клиента)
            "past_tx_change_std":   storm_days["tx_count"].std() / q_tx if len(storm_days) > 1 else 0,
        }
        records.append(rec)

    return pd.DataFrame(records)


# ──────────────────────────────────────────────
# 2. Rich quiet-day features
# ──────────────────────────────────────────────
def build_quiet_features(client_day: pd.DataFrame) -> pd.DataFrame:
    """
    Богатые признаки из ТИХИХ дней — никакой storm-day информации.
    """
    records = []
    for client_id, cdf in client_day.groupby("client"):
        q = cdf[~cdf["is_storm"]]
        if len(q) < 10:
            continue

        tx  = q["tx_count"]
        amt = q["tx_sum"]

        # Агрегируем по неделям для волатильности
        q2 = q.copy()
        q2["week"] = pd.to_datetime(q2["date"]).dt.to_period("W")
        weekly = q2.groupby("week")[["tx_count","tx_sum"]].sum()

        rec = {
            "client": client_id,
            # Базовые
            "q_tx_mean":    tx.mean(),
            "q_tx_median":  tx.median(),
            "q_amt_mean":   amt.mean(),
            "q_amt_median": amt.median(),
            # Волатильность
            "q_tx_cv":      tx.std() / (tx.mean() + 1e-6),
            "q_amt_cv":     amt.std() / (amt.mean() + 1e-6),
            "q_tx_week_cv": weekly["tx_count"].std() / (weekly["tx_count"].mean() + 1e-6),
            # Персентили
            "q_amt_p25":    amt.quantile(0.25),
            "q_amt_p75":    amt.quantile(0.75),
            "q_amt_iqr":    amt.quantile(0.75) - amt.quantile(0.25),
            # Структура трат в тихие дни
            "q_food_mean":  q["share_food"].mean(),
            "q_food_std":   q["share_food"].std(),
            "q_surv_mean":  q["share_survival"].mean(),
            "q_surv_std":   q["share_survival"].std(),
            "q_fun_mean":   q["share_fun"].mean(),
            "q_fun_std":    q["share_fun"].std(),
            "q_mcc_mean":   q["mcc_unique"].mean(),
            "q_mcc_std":    q["mcc_unique"].std(),
            # Концентрация трат (насколько одна категория доминирует)
            "q_concentration": q[["share_food","share_survival","share_fun"]].max(axis=1).mean(),
            # Активность в неделю
            "q_active_days_pct": (tx > 0).mean(),
            "q_n_days":     len(q),
        }
        records.append(rec)

    return pd.DataFrame(records)


# ──────────────────────────────────────────────
# 3. Pre-storm window (14 дней до бури)
# ──────────────────────────────────────────────
def _trend(arr):
    if len(arr) < 2: return 0.0
    return float(np.polyfit(np.arange(len(arr), dtype=float), arr, 1)[0])


def build_window_features(client_day: pd.DataFrame,
                           segments: pd.DataFrame,
                           storm_dates_sorted: pd.DatetimeIndex) -> pd.DataFrame:
    seg_map  = segments.set_index("client")["segment"].to_dict()
    day_ns   = np.timedelta64(1, "D")
    storm_ts = storm_dates_sorted.values.astype("datetime64[ns]")

    records = []
    for client_id, cdf in client_day.groupby("client"):
        if client_id not in seg_map:
            continue
        cdf_s    = cdf.sort_values("date")
        dates_np = cdf_s["date"].values
        feats_np = cdf_s[FEAT_COLS].values.astype(np.float32)

        windows = []
        for storm in storm_ts:
            mask = (dates_np >= storm - SEQ_LEN * day_ns) & (dates_np <= storm - day_ns)
            win  = feats_np[mask]
            if len(win) >= SEQ_LEN // 2:
                windows.append(win)
        if not windows:
            continue

        agg = {"client": client_id, "segment": seg_map[client_id]}
        for i, col in enumerate(FEAT_COLS):
            vals = np.concatenate([w[:, i] for w in windows])
            agg[f"win_{col}_mean"] = float(vals.mean())
            agg[f"win_{col}_std"]  = float(vals.std())
        agg["win_tx_trend"] = float(np.median(
            [_trend(w[:, 0]) for w in windows]))
        records.append(agg)

    return pd.DataFrame(records)


# ──────────────────────────────────────────────
# Визуализация
# ──────────────────────────────────────────────
def plot_importance(model, feature_names):
    imp = pd.Series(model.feature_importances_, index=feature_names).sort_values().tail(20)
    plt.figure(figsize=(9, 7))
    colors = ["#ef4444" if "past_" in n else "#2563eb" if "q_" in n else "#16a34a"
              for n in imp.index]
    imp.plot(kind="barh", color=colors, alpha=0.85)
    # Легенда
    from matplotlib.patches import Patch
    plt.legend(handles=[
        Patch(color="#ef4444", label="Track record (прошлые бури)"),
        Patch(color="#2563eb", label="Тихие дни"),
        Patch(color="#16a34a", label="Пре-штормовое окно"),
    ], loc="lower right", fontsize=9)
    plt.xlabel("Feature importance")
    plt.title("Важность признаков по группам (XGBoost)")
    plt.tight_layout()
    plt.savefig(PLOTS / "feature_importance.png", dpi=150)
    plt.close()
    print("Сохранено: plots/feature_importance.png")


def plot_confusion(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Предсказано"); plt.ylabel("Факт")
    plt.title("Матрица ошибок (XGBoost, без leakage)")
    plt.tight_layout()
    plt.savefig(PLOTS / "confusion_matrix.png", dpi=150)
    plt.close()
    print("Сохранено: plots/confusion_matrix.png")


# ──────────────────────────────────────────────
# Главный пайплайн
# ──────────────────────────────────────────────
if __name__ == "__main__":
    client_day = pd.read_parquet(DATA / "client_day.parquet")
    segments   = pd.read_parquet(DATA / "client_segments.parquet")[["client","segment"]]

    print(f"Клиентов: {len(segments):,}")
    print(segments["segment"].value_counts().to_string())

    # Топ-50 бурей, отсортированных по времени
    storm_dates_sorted = (client_day[client_day["is_storm"]]
                          .groupby("date")["kp_max"].max()
                          .nlargest(TOP_STORMS).index.sort_values())
    n_past = int(len(storm_dates_sorted) * PAST_RATIO)
    print(f"\nБурь всего: {len(storm_dates_sorted)} | "
          f"Ранних (признаки): {n_past} | "
          f"Поздних (таргет): {len(storm_dates_sorted)-n_past}")

    # ── Строим три группы признаков ──
    print("\n[1/3] Past-storm track record...")
    past_df = build_past_storm_features(client_day, storm_dates_sorted)
    print(f"  Клиентов с ≥3 прошлых бурями: {len(past_df):,}")

    print("[2/3] Rich quiet-day features...")
    quiet_df = build_quiet_features(client_day)
    print(f"  Клиентов с ≥10 тихих дней: {len(quiet_df):,}")

    print("[3/3] Pre-storm window features...")
    win_df = build_window_features(client_day, segments, storm_dates_sorted)
    print(f"  Клиентов с окнами: {len(win_df):,}")

    # ── Объединяем ──
    # cf_* признаки намеренно исключены: они содержат tx_change/amt_change за весь период,
    # что совпадает с признаками кластеризации → circular reasoning.
    df = (win_df
          .merge(past_df,  on="client", how="left")
          .merge(quiet_df, on="client", how="left"))

    feat_cols = [c for c in df.columns if c not in ("client","segment")]
    X = df[feat_cols].fillna(0).values

    le = LabelEncoder()
    y  = le.fit_transform(df["segment"])
    seg_names = list(le.classes_)

    print(f"\nИтого признаков: {len(feat_cols)}")
    print(f"  - past-storm:  {sum(1 for c in feat_cols if 'past_' in c)}")
    print(f"  - quiet-day:   {sum(1 for c in feat_cols if 'q_' in c)}")
    print(f"  - win-window:  {sum(1 for c in feat_cols if 'win_' in c)}")
    print(f"\nРаспределение классов:")
    for i, s in enumerate(seg_names):
        print(f"  {s}: {(y==i).sum():,} ({(y==i).mean()*100:.0f}%)")

    # ── Train/val ──
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)

    scaler    = StandardScaler()
    X_tr_s    = scaler.fit_transform(X_tr)
    X_val_s   = scaler.transform(X_val)
    X_all_s   = scaler.transform(X)

    sw = compute_sample_weight("balanced", y_tr)

    print("\nОбучаем XGBoost...")
    xgb = XGBClassifier(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.75,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.5,
        eval_metric="mlogloss",
        random_state=SEED,
        n_jobs=N_JOBS,
    )
    xgb.fit(X_tr_s, y_tr, sample_weight=sw,
            eval_set=[(X_val_s, y_val)], verbose=False)

    # CV
    from sklearn.pipeline import Pipeline
    pipe = Pipeline([("sc", StandardScaler()), ("xgb", XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.75,
        eval_metric="mlogloss", random_state=SEED, n_jobs=N_JOBS))])
    cv = cross_val_score(pipe, X, y,
                         cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                         scoring="f1_macro", n_jobs=1)
    print(f"\nCV F1-macro: {cv.mean():.3f} ± {cv.std():.3f}")

    # Val
    y_pred = xgb.predict(X_val_s)
    print("\n=== Классификация сегментов ===")
    print(classification_report(y_val, y_pred, target_names=seg_names))

    plot_confusion(le.inverse_transform(y_val), le.inverse_transform(y_pred), seg_names)
    plot_importance(xgb, feat_cols)

    # ── Предсказания ──
    calibrated = CalibratedClassifierCV(xgb, cv="prefit", method="isotonic")
    calibrated.fit(X_val_s, y_val)

    df["predicted_segment"] = le.inverse_transform(calibrated.predict(X_all_s))
    proba = calibrated.predict_proba(X_all_s)
    storm_idx = list(le.classes_).index("Метеозависимые")
    df["storm_proba"] = proba[:, storm_idx]

    result = segments.merge(
        df[["client","predicted_segment","storm_proba"]], on="client", how="left")
    result["predicted_segment"] = result["predicted_segment"].fillna(result["segment"])
    result["storm_proba"]       = result["storm_proba"].fillna(0.5)

    result.to_parquet(DATA / "client_predictions.parquet", index=False)
    print(f"\nПредсказания: data/client_predictions.parquet")
    print(result["predicted_segment"].value_counts().to_string())

    joblib.dump(xgb,        DATA / "storm_model.pkl")
    joblib.dump(le,         DATA / "label_encoder.pkl")
    joblib.dump(scaler,     DATA / "scaler.pkl")
    joblib.dump(calibrated, DATA / "storm_model_calibrated.pkl")
    joblib.dump(feat_cols,  DATA / "feature_names.pkl")
    print("\nШаг 4 завершён.")
