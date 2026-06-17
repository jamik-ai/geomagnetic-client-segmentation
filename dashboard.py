"""Storm Analytics — дашборд НИР. Запуск: streamlit run dashboard.py"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path
import joblib

BASE = Path(__file__).parent
DATA = BASE / "data"

st.set_page_config(page_title="Storm Analytics", page_icon="🌩️", layout="wide")

st.markdown("""
<style>
div[data-testid="stSidebarContent"] { background:#1a1f2e; }
div[data-testid="stSidebarContent"] * { color:#c9d1e0 !important; }
div[data-testid="stSidebarContent"] hr { border-color:#2e3650 !important; }
.stRadio label { font-size:0.95rem !important; padding:6px 0; }
.block-container { padding-top:2rem; }
.kpi-row { display:flex; gap:16px; margin-bottom:8px; }
.kpi { background:#fff; border:1px solid #e8ecf0; border-radius:8px;
       padding:16px 20px; flex:1; }
.kpi-label { font-size:0.72rem; color:#6b7280; font-weight:600;
             text-transform:uppercase; letter-spacing:.06em; margin-bottom:6px; }
.kpi-value { font-size:1.75rem; font-weight:700; color:#111827; line-height:1; }
.kpi-sub   { font-size:0.78rem; color:#9ca3af; margin-top:4px; }
.kpi-sub.down { color:#ef4444; }
.kpi-sub.up   { color:#16a34a; }
.insight { background:#f0f7ff; border-left:3px solid #2563eb;
           border-radius:0 6px 6px 0; padding:14px 18px;
           font-size:0.9rem; color:#1e3a5f; margin:12px 0; line-height:1.65; }
.insight b { color:#1d4ed8; }
.chart-title { font-size:1rem; font-weight:600; color:#111827;
               margin:20px 0 8px; border-bottom:1px solid #f3f4f6; padding-bottom:6px; }
</style>
""", unsafe_allow_html=True)


# ── helpers ────────────────────────────────────
def ch(fig, h=320, ml=10, mr=10, mb=40, mt=16, legend=True):
    """Применяет единый стиль к plotly-фигуре."""
    fig.update_layout(
        paper_bgcolor="white", plot_bgcolor="#fafafa",
        font=dict(family="sans-serif", size=12, color="#374151"),
        height=h, margin=dict(t=mt, b=mb, l=ml, r=mr),
        hoverlabel=dict(bgcolor="white", bordercolor="#e5e7eb", font_size=12),
    )
    if legend:
        fig.update_layout(legend=dict(
            orientation="h", y=-0.22, bgcolor="rgba(0,0,0,0)", font_size=11))
    fig.update_xaxes(gridcolor="#f0f0f0", linecolor="#e8ecf0", zeroline=False)
    fig.update_yaxes(gridcolor="#f0f0f0", linecolor="#e8ecf0", zeroline=False)
    return fig

def kpi(label, value, sub="", sub_cls=""):
    return (f'<div class="kpi"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div>'
            f'<div class="kpi-sub {sub_cls}">{sub}</div></div>')

def insight(text):
    st.markdown(f'<div class="insight">{text}</div>', unsafe_allow_html=True)

def ctitle(text):
    st.markdown(f'<div class="chart-title">{text}</div>', unsafe_allow_html=True)

SEG_C = {"Метеозависимые": "#16a34a", "Сберегающие": "#ef4444", "Нейтральные": "#2563eb"}


# ── данные (кэш) ───────────────────────────────
@st.cache_data
def load():
    cd   = pd.read_parquet(DATA / "client_day.parquet")
    pred = pd.read_parquet(DATA / "client_predictions.parquet")
    feat = pd.read_parquet(DATA / "client_features.parquet")
    kp   = pd.read_parquet(DATA / "kp.parquet")
    push = pd.read_csv(DATA / "push_schedule.csv",
                       parse_dates=["storm_date", "send_date"])
    seg  = pd.read_parquet(DATA / "client_segments.parquet")[["client", "segment"]]
    mcc  = pd.read_csv(DATA / "mcc_analysis.csv")
    return cd, pred, feat, kp, push, seg, mcc

@st.cache_data
def compute_pca(feat, seg):
    fs = feat.merge(seg, on="client")
    cols = ["tx_change","amt_change","survival_shift","food_shift",
            "fun_shift","avg_mcc_unique","tx_per_day_quiet"]
    X = fs[cols].fillna(0).values
    Z = PCA(n_components=2, random_state=42).fit_transform(
        StandardScaler().fit_transform(X))
    fs["PC1"] = Z[:,0]; fs["PC2"] = Z[:,1]
    return fs.sample(min(2500, len(fs)), random_state=42)

cd, pred, feat, kp, push, seg, mcc = load()
pca_df = compute_pca(feat, seg)
feat_seg = feat.merge(seg, on="client")

# pre-compute globals
storm_mask = cd["is_storm"]
avg_q = cd[~storm_mask]["tx_sum"].mean()
avg_s = cd[storm_mask]["tx_sum"].mean()
drop_pct = (avg_s - avg_q) / avg_q * 100
drop_abs  = avg_q - avg_s
seg_cd = cd.merge(seg, on="client")
pcols = ["tx_change","amt_change","survival_shift","food_shift","fun_shift"]
plabels = ["Транзакции", "Сумма трат", "Выживание", "Еда", "Развлечения"]
profiles = feat_seg.groupby("segment")[pcols].mean()

seg_stats = {}
for s in ["Метеозависимые","Сберегающие","Нейтральные"]:
    sub = seg_cd[seg_cd["segment"]==s]
    seg_stats[s] = {
        "n":  pred[pred["predicted_segment"]==s]["client"].nunique(),
        "q":  sub[~sub["is_storm"]]["tx_sum"].mean(),
        "st": sub[sub["is_storm"]]["tx_sum"].mean(),
    }
    seg_stats[s]["d"] = (seg_stats[s]["st"]-seg_stats[s]["q"])/seg_stats[s]["q"]*100

# ── сайдбар ────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='padding:16px 0 8px;text-align:center'>"
        "<span style='font-size:2rem'>🌩️</span><br>"
        "<b style='font-size:1.05rem;color:#e2e8f0'>Storm Analytics</b><br>"
        "<span style='font-size:0.75rem;color:#6b7280'>НИР · Банковская аналитика</span>"
        "</div>", unsafe_allow_html=True)
    st.divider()
    page = st.radio("Навигация", [
        "📊 Главная",
        "🗂 Сегменты",
        "⚡ Влияние бурь",
        "👤 Клиент",
        "💡 Рекомендации",
    ], label_visibility="collapsed")
    st.divider()
    st.markdown(
        f"<div style='font-size:.8rem;color:#6b7280;line-height:2'>"
        f"Клиентов: <b style='color:#94a3b8'>{pred['client'].nunique():,}</b><br>"
        f"Дней данных: <b style='color:#94a3b8'>{cd['date'].nunique():,}</b><br>"
        f"Дней бурь: <b style='color:#94a3b8'>{kp['is_storm'].sum()}</b></div>",
        unsafe_allow_html=True)


# ══════════════════════════════════════════════
# ГЛАВНАЯ
# ══════════════════════════════════════════════
if page == "📊 Главная":
    st.markdown("## 📊 Обзор исследования")
    st.caption("Сегментация банковских клиентов по чувствительности к геомагнитным бурям")

    # KPI
    n_high = (pred["storm_proba"] >= 0.75).sum()
    storm_share = kp["is_storm"].mean() * 100
    st.markdown(
        f'<div class="kpi-row">'
        + kpi("Клиентов в базе", f"{pred['client'].nunique():,}")
        + kpi("Дней бурь (Kp ≥ 4)", f"{kp['is_storm'].sum()}",
              f"{storm_share:.0f}% от периода")
        + kpi("Просадка оборота", f"{drop_pct:.1f}%",
              "в дни бурь vs спокойные", "down")
        + kpi("Клиентов с proba ≥ 75%", f"{n_high:,}",
              "высокая уверенность модели", "up")
        + '</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns([3, 1])

    with c1:
        ctitle("Геомагнитная активность (Kp-индекс) за период исследования")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=kp["date"], y=kp["kp_max"], name="Kp max",
            fill="tozeroy", fillcolor="rgba(37,99,235,0.07)",
            line=dict(color="#2563eb", width=1.5)))
        fig.add_trace(go.Scatter(
            x=kp[kp["is_storm"]]["date"], y=kp[kp["is_storm"]]["kp_max"],
            mode="markers", marker=dict(color="#ef4444", size=5), name="Буря (Kp ≥ 4)"))
        fig.add_hline(y=4, line_dash="dash", line_color="#ef4444", opacity=0.35,
                      annotation_text="  Kp = 4", annotation_font_color="#ef4444",
                      annotation_position="right")
        ch(fig, h=290, mb=30)
        st.plotly_chart(fig, width="stretch")

    with c2:
        ctitle("Сегменты")
        counts = pred["predicted_segment"].value_counts().reset_index()
        counts.columns = ["seg","n"]
        fig = px.pie(counts, values="n", names="seg",
                     color="seg", color_discrete_map=SEG_C, hole=0.52)
        fig.update_traces(textinfo="percent+label", textfont_size=11,
                          marker=dict(line=dict(color="white", width=2)))
        fig.update_layout(paper_bgcolor="white", height=290,
                          margin=dict(t=10, b=10, l=0, r=0),
                          showlegend=False,
                          font=dict(family="sans-serif", size=11))
        st.plotly_chart(fig, width="stretch")

    # Сводная таблица сегментов
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    for col, s, emoji in zip([c1,c2,c3],
        ["Метеозависимые","Сберегающие","Нейтральные"],
        ["📈","📉","➡️"]):
        d = seg_stats[s]["d"]
        col.metric(
            f"{emoji} {s}",
            f"{seg_stats[s]['n']:,} клиентов",
            f"{d:+.1f}% оборота в бури",
            delta_color="normal" if d > 0 else "inverse" if d < -0.5 else "off")

    st.markdown("<br>", unsafe_allow_html=True)
    ctitle("Изменение среднего чека по категориям в дни бурь (статистически значимые, p < 0.05)")
    top = pd.concat([mcc.head(12), mcc.tail(12)]).drop_duplicates().sort_values("pct_change")
    top["label"] = top["mcc_name"].str[:50]
    fig = go.Figure(go.Bar(
        x=top["pct_change"], y=top["label"], orientation="h",
        marker_color=["#ef4444" if x < 0 else "#16a34a" for x in top["pct_change"]],
        marker_line_width=0,
        text=top["pct_change"].apply(lambda x: f"{x:+.1f}%"),
        textposition="outside", textfont_size=10))
    fig.add_vline(x=0, line_color="#d1d5db", line_width=1)
    ch(fig, h=480, mr=70, mb=10)
    st.plotly_chart(fig, width="stretch")

    insight(
        f"В период исследования зафиксировано <b>{kp['is_storm'].sum()} дней</b> геомагнитных бурь "
        f"({storm_share:.0f}% от всего периода). "
        f"В эти дни средний оборот на клиента снижается на <b>{abs(drop_pct):.1f}%</b> "
        f"(с {avg_q:,.0f}₽ до {avg_s:,.0f}₽). "
        f"Сильнее всего реагируют <b>Сберегающие</b> "
        f"({seg_stats['Сберегающие']['d']:+.1f}%) — "
        f"они избегают физических выходов и снижают все расходы. "
        f"<b>Метеозависимые</b> напротив увеличивают трату "
        f"({seg_stats['Метеозависимые']['d']:+.1f}%) через компенсаторное потребление."
    )


# ══════════════════════════════════════════════
# СЕГМЕНТЫ
# ══════════════════════════════════════════════
elif page == "🗂 Сегменты":
    st.markdown("## 🗂 Анализ сегментов")
    st.caption("Поведенческие профили, кластеры и важность признаков")

    c1, c2 = st.columns(2)
    with c1:
        ctitle("Поведенческие профили (среднее изменение в дни бурь)")
        fig = go.Figure()
        for s, color in SEG_C.items():
            if s not in profiles.index: continue
            fig.add_bar(name=s, x=plabels, y=profiles.loc[s].values,
                        marker_color=color, marker_line_width=0, opacity=0.85)
        fig.add_hline(y=0, line_color="#9ca3af", line_width=1)
        ch(fig, h=340)
        fig.update_layout(barmode="group", xaxis_tickangle=-10)
        st.plotly_chart(fig, width="stretch")

    with c2:
        ctitle("Радар-диаграмма профилей")
        fig = go.Figure()
        for s, color in SEG_C.items():
            if s not in profiles.index: continue
            vals = profiles.loc[s].values.tolist()
            fig.add_trace(go.Scatterpolar(
                r=vals+[vals[0]], theta=plabels+[plabels[0]],
                fill="toself", name=s,
                line=dict(color=color, width=2),
                fillcolor=color, opacity=0.12))
        fig.update_layout(
            polar=dict(bgcolor="#fafafa",
                       radialaxis=dict(visible=True, gridcolor="#e5e7eb", color="#9ca3af"),
                       angularaxis=dict(gridcolor="#e5e7eb")),
            paper_bgcolor="white", height=340,
            margin=dict(t=20, b=50, l=20, r=20),
            legend=dict(orientation="h", y=-0.18, font_size=11),
            font=dict(family="sans-serif", size=11, color="#374151"))
        st.plotly_chart(fig, width="stretch")

    insight(
        f"Три сегмента различаются по реакции на бури. "
        f"<b>Сберегающие ({seg_stats['Сберегающие']['n']:,} чел.)</b> — "
        f"самое сильное снижение транзакций и суммы трат, "
        f"растёт доля товаров первой необходимости (запасы дома). "
        f"<b>Метеозависимые ({seg_stats['Метеозависимые']['n']:,} чел.)</b> — "
        f"транзакции растут, смещение в сторону еды и развлечений: "
        f"компенсируют стресс покупками. "
        f"<b>Нейтральные ({seg_stats['Нейтральные']['n']:,} чел.)</b> — "
        f"изменения близки к нулю по всем признакам."
    )

    ctitle("Кластеры клиентов в пространстве PCA (2D проекция 7 признаков)")
    fig = px.scatter(pca_df, x="PC1", y="PC2", color="segment",
                     color_discrete_map=SEG_C, opacity=0.45,
                     labels={"PC1":"Компонента 1","PC2":"Компонента 2","segment":"Сегмент"})
    fig.update_traces(marker=dict(size=4, line_width=0))
    ch(fig, h=400, mb=30)
    st.plotly_chart(fig, width="stretch")

    ctitle("Важность признаков (XGBoost Classifier)")
    try:
        model = joblib.load(DATA / "storm_model.pkl")
        names = joblib.load(DATA / "feature_names.pkl")
        imp = (pd.DataFrame({"Признак": names, "Важность": model.feature_importances_})
               .sort_values("Важность").tail(12))
        top_feat = imp.iloc[-1]["Признак"]
        fig = go.Figure(go.Bar(
            x=imp["Важность"], y=imp["Признак"], orientation="h",
            marker_color="#2563eb", marker_line_width=0, opacity=0.8,
            text=imp["Важность"].apply(lambda x: f"{x:.3f}"), textposition="outside"))
        ch(fig, h=360, mr=60, mb=10)
        st.plotly_chart(fig, width="stretch")
        insight(
            f"Наиболее важный признак для классификации сегмента — "
            f"<b>{top_feat}</b>. "
            f"Модель XGBoost обучена на 46 признаках (track record прошлых бурь, "
            f"тихие дни, пре-штормовые окна). CV F1-macro ≈ 0.65 — "
            f"честный результат при Silhouette=0.21 (перекрывающиеся кластеры)."
        )
    except Exception as e:
        st.warning(f"Модель не загружена: {e}")


# ══════════════════════════════════════════════
# ВЛИЯНИЕ БУРЬ
# ══════════════════════════════════════════════
elif page == "⚡ Влияние бурь":
    st.markdown("## ⚡ Влияние геомагнитных бурь на поведение")
    st.caption("Сравнительный анализ транзакционной активности в дни бурь и спокойные дни")

    # KPI
    st.markdown(
        f'<div class="kpi-row">'
        + kpi("Ср. оборот (спокойно)", f"{avg_q:,.0f} ₽")
        + kpi("Ср. оборот (буря)", f"{avg_s:,.0f} ₽", f"−{abs(drop_pct):.1f}%", "down")
        + kpi("Просадка на клиента", f"{drop_abs:,.0f} ₽", "в среднем за день бури", "down")
        + kpi("Дней бурь в периоде", f"{kp['is_storm'].sum()}",
              f"из {kp['date'].nunique()} дней наблюдений")
        + '</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        ctitle("Оборот по сегментам: спокойные дни vs дни бурь")
        df_bar = pd.DataFrame([
            {"Сегмент": s, "Период": "Спокойно", "Оборот": seg_stats[s]["q"]}
            for s in ["Метеозависимые","Нейтральные","Сберегающие"]
        ] + [
            {"Сегмент": s, "Период": "Буря", "Оборот": seg_stats[s]["st"]}
            for s in ["Метеозависимые","Нейтральные","Сберегающие"]
        ])
        fig = px.bar(df_bar, x="Сегмент", y="Оборот", color="Период", barmode="group",
                     color_discrete_map={"Спокойно":"#2563eb","Буря":"#ef4444"})
        fig.update_traces(marker_line_width=0)
        ch(fig, h=330, mb=50)
        fig.update_layout(yaxis_title="Ср. оборот (₽)")
        st.plotly_chart(fig, width="stretch")

    with c2:
        ctitle("Изменение оборота в дни бурь по сегментам (%)")
        segs = ["Метеозависимые","Нейтральные","Сберегающие"]
        diffs = [seg_stats[s]["d"] for s in segs]
        colors = ["#16a34a" if d > 0 else "#ef4444" for d in diffs]
        fig = go.Figure(go.Bar(
            x=segs, y=diffs, marker_color=colors, marker_line_width=0,
            text=[f"{d:+.1f}%" for d in diffs], textposition="outside"))
        fig.add_hline(y=0, line_color="#d1d5db", line_width=1)
        ch(fig, h=330, mb=50, legend=False)
        fig.update_layout(yaxis_title="Изменение (%)")
        st.plotly_chart(fig, width="stretch")

    insight(
        f"<b>Сберегающие</b> снижают оборот на <b>{abs(seg_stats['Сберегающие']['d']):.1f}%</b> "
        f"в дни бурь — наибольшая реакция среди всех сегментов. "
        f"Это согласуется с гипотезой энергосбережения: клиенты избегают "
        f"физических действий и выходов из дома, сокращая все расходы. "
        f"<b>Метеозависимые</b> увеличивают оборот на "
        f"<b>{seg_stats['Метеозависимые']['d']:+.1f}%</b> — "
        f"тревожное или компенсаторное потребление (запасы, доставка, аптеки). "
        f"<b>Нейтральные</b> практически не реагируют ({seg_stats['Нейтральные']['d']:+.1f}%)."
    )

    # Дневная динамика
    ctitle("Средний дневной оборот: динамика и дни бурь")
    daily = cd.groupby("date").agg(
        avg_sum=("tx_sum","mean"), is_storm=("is_storm","first")).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily["date"], y=daily["avg_sum"].rolling(7, center=True).mean(),
        name="7-дн. скользящее среднее",
        line=dict(color="#2563eb", width=2)))
    fig.add_trace(go.Scatter(
        x=daily["date"], y=daily["avg_sum"],
        name="Дневной оборот", mode="lines",
        line=dict(color="#cbd5e1", width=0.8), opacity=0.6))
    storm_pts = daily[daily["is_storm"]]
    fig.add_trace(go.Scatter(
        x=storm_pts["date"], y=storm_pts["avg_sum"],
        mode="markers", marker=dict(color="#ef4444", size=5, opacity=0.7),
        name="День бури"))
    ch(fig, h=300, mb=40)
    fig.update_layout(yaxis_title="Ср. оборот (₽)")
    st.plotly_chart(fig, width="stretch")

    ctitle("Топ MCC-категорий: падение и рост чека в дни бурь")
    c1, c2 = st.columns(2)
    with c1:
        bot = mcc.head(10).sort_values("pct_change")
        fig = go.Figure(go.Bar(
            x=bot["pct_change"], y=bot["mcc_name"].str[:40],
            orientation="h", marker_color="#ef4444", marker_line_width=0,
            text=bot["pct_change"].apply(lambda x: f"{x:.1f}%"), textposition="outside"))
        fig.add_vline(x=0, line_color="#d1d5db")
        ch(fig, h=320, mr=50, mb=10, legend=False)
        fig.update_layout(title_text="📉 Снижение чека", title_font_size=13)
        st.plotly_chart(fig, width="stretch")
    with c2:
        top10 = mcc.tail(10).sort_values("pct_change")
        fig = go.Figure(go.Bar(
            x=top10["pct_change"], y=top10["mcc_name"].str[:40],
            orientation="h", marker_color="#16a34a", marker_line_width=0,
            text=top10["pct_change"].apply(lambda x: f"+{x:.1f}%"), textposition="outside"))
        fig.add_vline(x=0, line_color="#d1d5db")
        ch(fig, h=320, mr=60, mb=10, legend=False)
        fig.update_layout(title_text="📈 Рост чека", title_font_size=13)
        st.plotly_chart(fig, width="stretch")


# ══════════════════════════════════════════════
# КЛИЕНТ
# ══════════════════════════════════════════════
elif page == "👤 Клиент":
    st.markdown("## 👤 Профиль клиента")
    st.caption("Предсказанный сегмент, поведение и персональные офферы")

    all_clients = sorted(pred["client"].unique())
    col_s, col_b = st.columns([5,1])
    with col_s:
        client_id = st.selectbox("Выберите клиента:", all_clients)
    with col_b:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🎲 Случайный", width="stretch"):
            st.session_state["rnd"] = int(np.random.choice(all_clients))
            st.rerun()
    if "rnd" in st.session_state:
        client_id = st.session_state["rnd"]

    row = pred[pred["client"] == client_id].iloc[0]
    seg_name = row["predicted_segment"]
    prob     = row["storm_proba"]

    # KPI клиента
    cd_c = cd[cd["client"] == client_id]
    tx_q = cd_c[~cd_c["is_storm"]]["tx_count"].mean()
    tx_s = cd_c[cd_c["is_storm"]]["tx_count"].mean()
    sum_q = cd_c[~cd_c["is_storm"]]["tx_sum"].mean()
    sum_s = cd_c[cd_c["is_storm"]]["tx_sum"].mean()

    st.markdown(
        f'<div class="kpi-row">'
        + kpi("Сегмент", seg_name)
        + kpi("Уверенность модели", f"{prob:.0%}",
              "высокая" if prob >= 0.75 else "средняя",
              "up" if prob >= 0.75 else "")
        + kpi("Транзакций/день (спокойно)", f"{tx_q:.1f}")
        + kpi("Транзакций/день (буря)", f"{tx_s:.1f}",
              f"{(tx_s-tx_q)/tx_q*100:+.1f}% изменение",
              "up" if tx_s > tx_q else "down")
        + '</div>', unsafe_allow_html=True)

    desc = {
        "Метеозависимые": (
            f"Клиент <b>увеличивает</b> активность в дни бурь. "
            f"Характерно компенсаторное потребление — больше трат на еду, аптеки, онлайн-сервисы. "
            f"Уверенность модели: <b>{prob:.0%}</b>."),
        "Сберегающие": (
            f"Клиент <b>сокращает</b> расходы в дни бурь — избегает выходов из дома, "
            f"откладывает несрочные покупки. "
            f"Оборот клиента в дни бурь: <b>{sum_s:,.0f}₽</b> vs {sum_q:,.0f}₽ в спокойные дни. "
            f"Уверенность модели: <b>{prob:.0%}</b>."),
        "Нейтральные": (
            f"Поведение клиента <b>не меняется</b> в дни бурь. "
            f"Транзакционная активность стабильна вне зависимости от Kp-индекса. "
            f"Уверенность модели: <b>{prob:.0%}</b>."),
    }
    insight(desc[seg_name])

    c1, c2 = st.columns(2)
    with c1:
        ctitle("Активность клиента: спокойные дни vs дни бурь")
        metrics_df = pd.DataFrame({
            "Метрика": ["Транзакций/день","Сумма/день (₽)"],
            "Спокойно": [tx_q, sum_q],
            "Буря": [tx_s, sum_s],
        })
        fig = go.Figure()
        fig.add_bar(name="Спокойно", x=metrics_df["Метрика"], y=metrics_df["Спокойно"],
                    marker_color="#2563eb", marker_line_width=0)
        fig.add_bar(name="Буря", x=metrics_df["Метрика"], y=metrics_df["Буря"],
                    marker_color="#ef4444", marker_line_width=0)
        ch(fig, h=280, mb=40)
        fig.update_layout(barmode="group")
        st.plotly_chart(fig, width="stretch")

        ctitle("История транзакций")
        cd_s = cd_c.sort_values("date")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=cd_s["date"], y=cd_s["tx_count"],
            mode="lines", line=dict(color="#94a3b8", width=1), name="Транзакций"))
        bp = cd_s[cd_s["is_storm"]]
        fig2.add_trace(go.Scatter(
            x=bp["date"], y=bp["tx_count"],
            mode="markers", marker=dict(color="#ef4444", size=6), name="Буря"))
        ch(fig2, h=220, mb=30)
        st.plotly_chart(fig2, width="stretch")

    with c2:
        ctitle("Персональные офферы")
        cp = push[push["client"] == client_id]
        if len(cp):
            disp = (cp.drop_duplicates("category")
                    .sort_values("cashback_pct", ascending=False)
                    [["category","cashback_pct","days_before","rationale"]]
                    .rename(columns={
                        "category":"Категория",
                        "cashback_pct":"Кэшбэк %",
                        "days_before":"Дней до бури",
                        "rationale":"Логика"}))
            st.dataframe(disp, width="stretch", hide_index=True, height=220)

            avg_cb = cp["cashback_pct"].mean()
            max_cb = cp["cashback_pct"].max()

            ctitle("Средний персональный кэшбэк")
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=avg_cb,
                title={"text": f"Сегмент: {seg_name}", "font": {"size": 13, "color":"#374151"}},
                gauge={
                    "axis": {"range": [0, 20], "tickcolor":"#9ca3af",
                             "tickfont": {"size":10}},
                    "bar":  {"color": SEG_C[seg_name], "thickness": 0.22},
                    "bgcolor": "#f9fafb",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 7],  "color": "#f3f4f6"},
                        {"range": [7, 13], "color": "#eff6ff"},
                        {"range": [13, 20],"color": "#dbeafe"},
                    ],
                    "threshold": {"line": {"color": "#374151", "width": 2},
                                  "thickness": 0.75, "value": max_cb},
                },
                number={"suffix":"%", "font":{"size":32, "color":"#111827"}},
            ))
            fig.update_layout(height=260, paper_bgcolor="white",
                              margin=dict(t=30, b=10, l=20, r=20),
                              font=dict(family="sans-serif"))
            st.plotly_chart(fig, width="stretch")

            insight(
                f"Клиент получит <b>{len(disp)} оффера</b> перед каждой прогнозируемой бурей. "
                f"Средний кэшбэк: <b>{avg_cb:.1f}%</b>, максимальный: <b>{max_cb}%</b>. "
                f"Категории подобраны под сегмент — все офферы на онлайн-сервисы и доставку, "
                f"без необходимости физических выходов из дома."
            )
        else:
            st.info("Нейтральный сегмент — стандартные офферы без привязки к бурям.")


# ══════════════════════════════════════════════
# РЕКОМЕНДАЦИИ
# ══════════════════════════════════════════════
elif page == "💡 Рекомендации":
    st.markdown("## 💡 Система рекомендаций")
    st.caption("Персональный кэшбэк перед бурями и оценка бизнес-эффекта")

    rec   = drop_abs * 0.35
    n_act = pred[pred["predicted_segment"] != "Нейтральные"]["client"].nunique()

    st.markdown(
        f'<div class="kpi-row">'
        + kpi("Офферов в расписании", f"{len(push):,}")
        + kpi("Активных клиентов", f"{n_act:,}", "Метеозависимые + Сберегающие")
        + kpi("Просадка на клиента", f"{drop_abs:,.0f} ₽", "в день бури", "down")
        + kpi("Потенциал удержания", f"{rec:,.0f} ₽", "при конверсии офферов 35%", "up")
        + '</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        ctitle("Waterfall: механика удержания оборота")
        fig = go.Figure(go.Waterfall(
            orientation="v",
            x=["Спокойный\nдень","Просадка\nв бурю","Удержание\nофферами","Итог"],
            y=[avg_q, -drop_abs, rec, 0],
            measure=["absolute","relative","relative","total"],
            connector={"line":{"color":"#d1d5db","dash":"dot"}},
            increasing={"marker":{"color":"#16a34a"}},
            decreasing={"marker":{"color":"#ef4444"}},
            totals={"marker":{"color":"#2563eb"}},
            text=[f"{avg_q:,.0f}₽",f"−{drop_abs:,.0f}₽",
                  f"+{rec:,.0f}₽",f"{avg_s+rec:,.0f}₽"],
            textposition="outside",
            textfont=dict(size=11, color="#374151")))
        ch(fig, h=340, mb=50, legend=False)
        fig.update_layout(yaxis_title="Оборот (₽)")
        st.plotly_chart(fig, width="stretch")

    with c2:
        ctitle("Кэшбэк по категориям (средний %)")
        cat_avg = (push[push["segment"]!="Нейтральные"]
                   .groupby("category")["cashback_pct"].mean()
                   .sort_values().reset_index())
        fig = go.Figure(go.Bar(
            x=cat_avg["cashback_pct"], y=cat_avg["category"],
            orientation="h", marker_color="#2563eb",
            marker_line_width=0, opacity=0.85,
            text=cat_avg["cashback_pct"].apply(lambda x: f"{x:.1f}%"),
            textposition="outside"))
        ch(fig, h=340, mr=55, mb=10, legend=False)
        fig.update_layout(xaxis_title="Кэшбэк (%)")
        st.plotly_chart(fig, width="stretch")

    # Фильтр сегментов
    st.divider()
    seg_filter = st.multiselect("Сегменты для анализа",
                                list(SEG_C.keys()),
                                default=["Метеозависимые","Сберегающие"])
    push_f = push[push["segment"].isin(seg_filter)]

    c1, c2 = st.columns(2)
    with c1:
        ctitle("Распределение персонального кэшбэка")
        fig = go.Figure()
        for s in seg_filter:
            if s == "Нейтральные": continue
            sub = push_f[push_f["segment"]==s]["cashback_pct"]
            fig.add_trace(go.Histogram(
                x=sub, name=s, marker_color=SEG_C[s],
                nbinsx=16, opacity=0.75, marker_line_width=0))
        ch(fig, h=300, mb=50)
        fig.update_layout(barmode="overlay",
                          xaxis_title="Кэшбэк (%)", yaxis_title="Офферов")
        st.plotly_chart(fig, width="stretch")

    with c2:
        ctitle("Календарь отправки пушей")
        daily_p = (push_f.groupby(["send_date","segment"])["client"]
                   .nunique().reset_index().rename(columns={"client":"n"}))
        fig = px.bar(daily_p, x="send_date", y="n", color="segment",
                     color_discrete_map=SEG_C, barmode="stack",
                     labels={"send_date":"Дата","n":"Клиентов","segment":"Сегмент"})
        fig.update_traces(marker_line_width=0)
        ch(fig, h=300, mb=50)
        fig.update_layout(yaxis_title="Клиентов")
        st.plotly_chart(fig, width="stretch")

    ctitle("Примеры офферов по сегментам")
    disp = (push_f.drop_duplicates(["segment","category"])
            .sort_values(["segment","cashback_pct"], ascending=[True,False])
            [["segment","category","cashback_pct","days_before","rationale"]]
            .rename(columns={
                "segment":"Сегмент","category":"Категория",
                "cashback_pct":"Кэшбэк %","days_before":"Дней до бури",
                "rationale":"Логика"}))
    st.dataframe(disp, width="stretch", hide_index=True)

    insight(
        f"Система отправляет офферы за 1–3 дня до прогнозируемой бури. "
        f"Кэшбэк <b>персонализирован</b>: интерполируется от базового до максимума "
        f"в зависимости от уверенности модели (storm_proba). "
        f"При конверсии 35% банк удерживает около <b>{rec:,.0f}₽</b> оборота с клиента в день бури. "
        f"Все офферы направлены на онлайн-каналы и доставку — "
        f"снижение барьера для клиентов, избегающих физических действий."
    )
