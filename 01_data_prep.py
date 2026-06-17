"""
Шаг 1: Загрузка и подготовка данных.
Параллелизм: transactions.csv → 3 части → multiprocessing.Pool.
Вывод воркеров виден в реальном времени.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from joblib import Parallel, delayed
import multiprocessing as mp
from datetime import datetime

BASE      = Path(__file__).parent
TRANS_CSV = BASE / "transactions.csv"
KP_CSV    = BASE / "kp.csv"
MCC_CSV   = BASE / "mcc_codes.csv"
OUT_DIR   = BASE / "data"
OUT_DIR.mkdir(exist_ok=True)

N_JOBS     = 3
CHUNK_SIZE = 500_000


def ts():
    return datetime.now().strftime("%H:%M:%S")


# ──────────────────────────────────────────────
# Справочники
# ──────────────────────────────────────────────
def load_kp():
    df = pd.read_csv(KP_CSV, parse_dates=["date"])
    df["date"] = df["date"].dt.normalize()
    df["is_storm"] = df["kp_max"] >= 4.0
    df["storm_level"] = pd.cut(
        df["kp_max"],
        bins=[-1, 2, 4, 6, 100],
        labels=["quiet", "unsettled", "storm", "severe_storm"],
    )
    return df


def load_mcc():
    df = pd.read_csv(MCC_CSV, dtype={"mcc": "str"})
    df["mcc"] = df["mcc"].astype(str).str.zfill(4)
    desc_col = "edited_description" if "edited_description" in df.columns else df.columns[1]
    return df[["mcc", desc_col]].rename(columns={desc_col: "mcc_name"})


# ──────────────────────────────────────────────
# Разбивка CSV на части (один проход)
# ──────────────────────────────────────────────
def split_csv(src: Path, n_parts: int) -> list:
    paths = [OUT_DIR / f"_tmp_part_{i}.csv" for i in range(n_parts)]
    print(f"[{ts()}] Делим {src.name} на {n_parts} части...", flush=True)
    handles = [open(p, "w", buffering=1 << 20) for p in paths]
    with open(src, "r", buffering=1 << 22) as f:
        header = f.readline()
        for h in handles:
            h.write(header)
        for i, line in enumerate(f):
            handles[i % n_parts].write(line)
    for h in handles:
        h.close()
    sizes = [p.stat().st_size // 1_000_000 for p in paths]
    print(f"[{ts()}] Части готовы: {sizes} МБ", flush=True)
    return paths


# ──────────────────────────────────────────────
# Обработка одной части (воркер)
# Должна быть на уровне модуля для multiprocessing
# ──────────────────────────────────────────────
def process_part(args):
    part_idx, part_path, kp_df, mcc_df = args
    pfx = f"[{ts()}] Часть {part_idx}"

    print(f"{pfx}: старт чтения {Path(part_path).stat().st_size // 1_000_000} МБ", flush=True)

    kp_idx  = kp_df.set_index("date")[["kp_max", "kp_avg", "is_storm"]]
    mcc_idx = mcc_df.set_index("mcc")["mcc_name"]

    chunks_done = []
    reader = pd.read_csv(
        part_path,
        dtype={"client": "int32", "card": "int32", "mcc": "str",
               "group": "category", "value": "category"},
        parse_dates=["date"],
        chunksize=CHUNK_SIZE,
    )

    for chunk_num, chunk in enumerate(reader, 1):
        print(f"{pfx}: чанк {chunk_num} — {len(chunk):,} строк...", flush=True)

        chunk["date"] = chunk["date"].dt.normalize()
        chunk["mcc"]  = chunk["mcc"].astype(str).str.zfill(4)
        chunk["amt"]  = chunk["amt"].astype("float32")

        chunk = chunk.join(kp_idx, on="date", how="inner")
        chunk["mcc_name"] = chunk["mcc"].map(mcc_idx).fillna("MCC " + chunk["mcc"])

        # Бинарные столбцы — без lambda, векторизованно
        chunk["is_survival"]  = (chunk["value"] == "survival").astype("float32")
        chunk["is_self_real"] = (chunk["value"] == "self_realization").astype("float32")
        chunk["is_food"]      = (chunk["group"] == "food").astype("float32")
        chunk["is_fun"]       = (chunk["group"] == "fun").astype("float32")

        agg = chunk.groupby(["client", "date"]).agg(
            tx_count        = ("amt",          "count"),
            tx_sum          = ("amt",          "sum"),
            tx_mean         = ("amt",          "mean"),
            mcc_unique      = ("mcc",          "nunique"),
            share_survival  = ("is_survival",  "mean"),
            share_self_real = ("is_self_real", "mean"),
            share_food      = ("is_food",      "mean"),
            share_fun       = ("is_fun",       "mean"),
            kp_max          = ("kp_max",       "first"),
            kp_avg          = ("kp_avg",       "first"),
            is_storm        = ("is_storm",     "first"),
        ).reset_index()

        chunks_done.append(agg)
        print(f"{pfx}: чанк {chunk_num} готов — {len(agg):,} client-day строк", flush=True)

    print(f"{pfx}: финальная агрегация {len(chunks_done)} чанков...", flush=True)
    if not chunks_done:
        print(f"{pfx}: ПУСТО — нет данных!", flush=True)
        return pd.DataFrame()

    df = pd.concat(chunks_done, ignore_index=True)
    result = df.groupby(["client", "date"]).agg(
        tx_count        = ("tx_count",      "sum"),
        tx_sum          = ("tx_sum",        "sum"),
        tx_mean         = ("tx_mean",       "mean"),
        mcc_unique      = ("mcc_unique",    "max"),
        share_survival  = ("share_survival","mean"),
        share_self_real = ("share_self_real","mean"),
        share_food      = ("share_food",    "mean"),
        share_fun       = ("share_fun",     "mean"),
        kp_max          = ("kp_max",        "first"),
        kp_avg          = ("kp_avg",        "first"),
        is_storm        = ("is_storm",      "first"),
    ).reset_index()

    print(f"{pfx}: ГОТОВО — {len(result):,} строк, {result['client'].nunique():,} клиентов", flush=True)
    return result


# ──────────────────────────────────────────────
# Клиентские признаки (параллельно)
# ──────────────────────────────────────────────
def _agg_client_batch(df_batch: pd.DataFrame, days_storm: int, days_quiet: int) -> pd.DataFrame:
    def agg_client(g):
        storm      = g[g["is_storm"]]
        quiet      = g[~g["is_storm"]]
        tx_storm   = storm["tx_count"].sum() / max(days_storm, 1)
        tx_quiet   = quiet["tx_count"].sum() / max(days_quiet, 1)
        change     = (tx_storm - tx_quiet) / (tx_quiet + 1e-6)
        amt_storm  = storm["tx_sum"].mean() if len(storm) else 0
        amt_quiet  = quiet["tx_sum"].mean() if len(quiet) else 0
        amt_change = (amt_storm - amt_quiet) / (amt_quiet + 1e-6)
        return pd.Series({
            "tx_per_day_quiet":  tx_quiet,
            "tx_per_day_storm":  tx_storm,
            "tx_change":         change,
            "amt_per_day_quiet": amt_quiet,
            "amt_per_day_storm": amt_storm,
            "amt_change":        amt_change,
            "avg_mcc_unique":    g["mcc_unique"].mean(),
            "survival_shift":    (storm["share_survival"].mean() if len(storm) else 0)
                               - (quiet["share_survival"].mean() if len(quiet) else 0),
            "food_shift":        (storm["share_food"].mean() if len(storm) else 0)
                               - (quiet["share_food"].mean() if len(quiet) else 0),
            "fun_shift":         (storm["share_fun"].mean() if len(storm) else 0)
                               - (quiet["share_fun"].mean() if len(quiet) else 0),
        })
    return df_batch.groupby("client").apply(agg_client).reset_index()


def build_client_features(client_day: pd.DataFrame, df_kp: pd.DataFrame) -> pd.DataFrame:
    days_storm = df_kp[df_kp["is_storm"]]["date"].nunique()
    days_quiet = df_kp[~df_kp["is_storm"]]["date"].nunique()

    clients = client_day["client"].unique()
    batches = np.array_split(clients, N_JOBS)
    print(f"[{ts()}] build_client_features: {len(clients):,} клиентов → {N_JOBS} батча", flush=True)

    parts = Parallel(n_jobs=N_JOBS)(
        delayed(_agg_client_batch)(
            client_day[client_day["client"].isin(b)], days_storm, days_quiet
        )
        for b in batches
    )
    features = pd.concat(parts, ignore_index=True)
    features = features[features["tx_per_day_quiet"] * days_quiet >= 10]
    return features


# ──────────────────────────────────────────────
# main
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print(f"[{ts()}] ── Шаг 1: подготовка данных ──", flush=True)

    print(f"[{ts()}] Загружаем справочники...", flush=True)
    df_kp  = load_kp()
    df_mcc = load_mcc()
    print(f"[{ts()}] Kp: {df_kp['is_storm'].sum()} дней бурь из {len(df_kp)} "
          f"({df_kp['date'].min().date()} — {df_kp['date'].max().date()})", flush=True)
    print(f"[{ts()}] MCC: {len(df_mcc)} кодов", flush=True)

    # Используем существующие части если они есть
    part_paths = [OUT_DIR / f"_tmp_part_{i}.csv" for i in range(N_JOBS)]
    if all(p.exists() for p in part_paths):
        sizes = [p.stat().st_size // 1_000_000 for p in part_paths]
        print(f"[{ts()}] Части уже есть: {sizes} МБ — пропускаем разбивку", flush=True)
    else:
        part_paths = split_csv(TRANS_CSV, N_JOBS)

    print(f"\n[{ts()}] Запускаем {N_JOBS} воркера...", flush=True)
    args = [(i, str(p), df_kp, df_mcc) for i, p in enumerate(part_paths)]

    with mp.Pool(processes=N_JOBS) as pool:
        results = pool.map(process_part, args)

    print(f"\n[{ts()}] Финальная агрегация всех частей...", flush=True)
    client_day = pd.concat(results, ignore_index=True)
    client_day = client_day.groupby(["client", "date"]).agg(
        tx_count        = ("tx_count",       "sum"),
        tx_sum          = ("tx_sum",         "sum"),
        tx_mean         = ("tx_mean",        "mean"),
        mcc_unique      = ("mcc_unique",     "max"),
        share_survival  = ("share_survival", "mean"),
        share_self_real = ("share_self_real","mean"),
        share_food      = ("share_food",     "mean"),
        share_fun       = ("share_fun",      "mean"),
        kp_max          = ("kp_max",         "first"),
        kp_avg          = ("kp_avg",         "first"),
        is_storm        = ("is_storm",       "first"),
    ).reset_index()

    print(f"[{ts()}] client_day: {len(client_day):,} строк | {client_day['client'].nunique():,} клиентов", flush=True)
    client_day.to_parquet(OUT_DIR / "client_day.parquet", index=False)
    print(f"[{ts()}] Сохранено: data/client_day.parquet", flush=True)

    for p in part_paths:
        p.unlink()
    print(f"[{ts()}] Временные файлы удалены", flush=True)

    print(f"\n[{ts()}] Строим клиентские признаки...", flush=True)
    features = build_client_features(client_day, df_kp)
    features.to_parquet(OUT_DIR / "client_features.parquet", index=False)
    print(f"[{ts()}] Сохранено: data/client_features.parquet ({len(features):,} клиентов)", flush=True)

    df_kp.to_parquet(OUT_DIR / "kp.parquet", index=False)
    df_mcc.to_parquet(OUT_DIR / "mcc.parquet", index=False)

    print(f"\n[{ts()}] ── Шаг 1 завершён ──", flush=True)
