"""ChurnRadar: Streamlit dashboard for the churn prediction model.

Run:  streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.style import (CSS, CYAN, DANGER, MUTED, RISK_COLORS, SAFE, VIOLET, WARN,
                       hero, kpi, section, style_fig)
from src.config import ID_COL, MODEL_PATH, ROOT, TARGET
from src.data import clean, load_raw, validate_columns
from src.features import CATEGORY_OPTIONS, INTERNET_ADDONS, add_features
from src.pdf_tables import read_customer_file
from src.predict import ChurnPredictor

SAMPLE_PDF = ROOT / "data" / "sample" / "customers_sample.pdf"

st.set_page_config(page_title="ChurnRadar", page_icon="📡", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

PRETTY = {
    "tenure": "Tenure (months)", "MonthlyCharges": "Monthly charges", "TotalCharges": "Total charges",
    "avg_monthly_spend": "Avg monthly spend", "charge_ratio": "Bill vs. own average",
    "num_services": "No. of services", "tenure_bucket": "Tenure band", "auto_pay": "Auto-pay",
    "SeniorCitizen": "Senior citizen", "PaymentMethod": "Payment", "InternetService": "Internet",
    "PaperlessBilling": "Paperless billing", "OnlineSecurity": "Online security",
    "OnlineBackup": "Online backup", "DeviceProtection": "Device protection",
    "TechSupport": "Tech support", "StreamingTV": "Streaming TV", "StreamingMovies": "Streaming movies",
    "MultipleLines": "Multiple lines", "PhoneService": "Phone service",
}
YES_NO_TOGGLES = ["Partner", "Dependents", "PaperlessBilling", "PhoneService"]


def pretty(name: str) -> str:
    return PRETTY.get(name, name)


# ---------------------------------------------------------------- data & model
@st.cache_resource
def get_predictor() -> ChurnPredictor:
    return ChurnPredictor()


@st.cache_data
def get_dataset() -> pd.DataFrame:
    return clean(load_raw())


@st.cache_data(show_spinner="Scanning customers...")
def score(df: pd.DataFrame) -> pd.DataFrame:
    scored = clean(df).join(get_predictor().predict(df))
    if ID_COL not in scored:
        scored.insert(0, ID_COL, [f"row-{i + 1}" for i in range(len(scored))])
    scored["tenure_bucket"] = add_features(scored)["tenure_bucket"]
    return scored


if not MODEL_PATH.exists():
    st.error("No trained model found. Run `python -m src.train` first.")
    st.stop()

predictor = get_predictor()
report = predictor.report
st.markdown(hero(report["model_name"], report["metrics"]["roc_auc"]), unsafe_allow_html=True)


# ---------------------------------------------------------------- customer state
PRESETS = {
    "loyal": dict(gender="Female", SeniorCitizen=0, Partner="Yes", Dependents="Yes", tenure=64,
                  PhoneService="Yes", MultipleLines="Yes", InternetService="DSL", OnlineSecurity="Yes",
                  OnlineBackup="Yes", DeviceProtection="Yes", TechSupport="Yes", StreamingTV="No",
                  StreamingMovies="Yes", Contract="Two year", PaperlessBilling="No",
                  PaymentMethod="Credit card (automatic)", MonthlyCharges=74.5),
    "risky": dict(gender="Male", SeniorCitizen=1, Partner="No", Dependents="No", tenure=3,
                  PhoneService="Yes", MultipleLines="No", InternetService="Fiber optic",
                  OnlineSecurity="No", OnlineBackup="No", DeviceProtection="No", TechSupport="No",
                  StreamingTV="Yes", StreamingMovies="No", Contract="Month-to-month",
                  PaperlessBilling="Yes", PaymentMethod="Electronic check", MonthlyCharges=89.9),
}


def load_profile(p: dict, label: str, auto_total: bool = True) -> None:
    ss = st.session_state
    ss.gender = p["gender"]
    ss.senior = bool(int(p["SeniorCitizen"]))
    for k in YES_NO_TOGGLES:
        ss[k] = p[k] == "Yes"
    ss.MultipleLines = p["MultipleLines"] == "Yes"
    ss.InternetService = p["InternetService"]
    for a in INTERNET_ADDONS:
        ss[a] = p[a] == "Yes"
    ss.tenure = int(p["tenure"])
    ss.Contract = p["Contract"]
    ss.PaymentMethod = p["PaymentMethod"]
    ss.MonthlyCharges = float(min(max(p["MonthlyCharges"], 18.0), 120.0))
    ss.auto_total = auto_total
    ss.TotalCharges = float(p.get("TotalCharges", p["tenure"] * p["MonthlyCharges"]))
    ss.profile_label = label


def load_random_customer() -> None:
    row = get_dataset().sample(1).iloc[0].to_dict()
    truth = "churned 💔" if row[TARGET] == 1 else "stayed 💚"
    load_profile(row, f"Real customer **{row[ID_COL]}**, who actually *{truth}*", auto_total=False)


def load_customer_by_id(customer_id: str) -> None:
    df = st.session_state.get("batch_scored")
    row = df.loc[df[ID_COL] == customer_id].iloc[0].to_dict()
    load_profile(row, f"Customer **{customer_id}** from batch scan", auto_total=False)
    st.toast(f"Loaded {customer_id} into Customer Lab", icon="🔬")


if "gender" not in st.session_state:
    load_profile(PRESETS["risky"], "Preset: 🔥 flight risk")


def collect_customer() -> dict:
    ss = st.session_state
    yn = lambda b: "Yes" if b else "No"
    internet = ss.InternetService or "No"
    tenure, monthly = ss.tenure, ss.MonthlyCharges
    customer = {
        "gender": ss.gender or "Female",
        "SeniorCitizen": int(ss.senior),
        **{k: yn(ss[k]) for k in YES_NO_TOGGLES},
        "MultipleLines": yn(ss.MultipleLines) if ss.PhoneService else "No phone service",
        "InternetService": internet,
        **{a: (yn(ss[a]) if internet != "No" else "No internet service") for a in INTERNET_ADDONS},
        "tenure": tenure,
        "Contract": ss.Contract or "Month-to-month",
        "PaymentMethod": ss.PaymentMethod,
        "MonthlyCharges": monthly,
        "TotalCharges": round(tenure * monthly, 2) if ss.auto_total else ss.TotalCharges,
    }
    return customer


# ---------------------------------------------------------------- charts
def gauge(p: float, threshold: float) -> go.Figure:
    color = RISK_COLORS["High"] if p >= 0.6 else RISK_COLORS["Medium"] if p >= 0.3 else RISK_COLORS["Low"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=p * 100,
        number={"suffix": "%", "font": {"size": 54, "family": "Space Grotesk", "color": "#fff"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": MUTED, "tickwidth": 1, "ticksuffix": "%"},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "rgba(255,255,255,0.04)", "borderwidth": 0,
            "steps": [
                {"range": [0, 30], "color": "rgba(46,229,157,0.13)"},
                {"range": [30, 60], "color": "rgba(255,176,32,0.13)"},
                {"range": [60, 100], "color": "rgba(255,77,109,0.13)"},
            ],
            "threshold": {"line": {"color": "#fff", "width": 3}, "thickness": 0.9, "value": threshold * 100},
        },
    ))
    fig = style_fig(fig, height=250)
    fig.update_layout(margin=dict(l=45, r=45, t=25, b=0))
    return fig


def reasons_chart(reasons: list[dict], top: int = 8) -> go.Figure:
    rows = reasons[:top][::-1]
    labels = [f"{pretty(r['feature'])} = {r['value']}" for r in rows]
    impacts = [r["impact"] for r in rows]
    fig = go.Figure(go.Bar(
        x=impacts, y=labels, orientation="h",
        marker=dict(color=[DANGER if v > 0 else SAFE for v in impacts], cornerradius=6),
        hovertemplate="%{y}<br>impact %{x:+.3f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_color="rgba(255,255,255,0.35)")
    fig = style_fig(fig, height=330)
    fig.update_layout(
        xaxis_title="◀ keeps them      |      pushes toward churn ▶",
        xaxis_title_font=dict(size=12, color=MUTED), bargap=0.35,
    )
    return fig


# ================================================================ layout
tab_lab, tab_batch, tab_model, tab_data = st.tabs(
    ["🔬  Customer Lab", "📡  Batch Radar", "🧠  Model Insights", "🗺️  Data Explorer"])

# ---------------------------------------------------------------- Customer Lab
with tab_lab:
    b1, b2, b3, info = st.columns([1, 1, 1, 2.2], vertical_alignment="center")
    b1.button("🎲 Random real customer", on_click=load_random_customer, width="stretch")
    b2.button("💚 Loyal veteran", on_click=load_profile,
              args=(PRESETS["loyal"], "Preset: 💚 loyal veteran"), width="stretch")
    b3.button("🔥 Flight risk", on_click=load_profile,
              args=(PRESETS["risky"], "Preset: 🔥 flight risk"), width="stretch")
    info.caption(f"Loaded: {st.session_state.profile_label}. Tweak anything; the score updates live.")

    form_col, result_col = st.columns([1.1, 1], gap="large")

    with form_col:
        with st.container(border=True):
            st.markdown(section("👤 Profile"), unsafe_allow_html=True)
            st.segmented_control("Gender", CATEGORY_OPTIONS["gender"], key="gender")
            c1, c2, c3 = st.columns(3)
            c1.toggle("Senior citizen", key="senior")
            c2.toggle("Has partner", key="Partner")
            c3.toggle("Has dependents", key="Dependents")

        with st.container(border=True):
            st.markdown(section("💳 Account"), unsafe_allow_html=True)
            st.slider("Tenure (months with us)", 0, 72, key="tenure")
            st.segmented_control("Contract", CATEGORY_OPTIONS["Contract"], key="Contract")
            c1, c2 = st.columns([1.6, 1], vertical_alignment="bottom")
            c1.selectbox("Payment method", CATEGORY_OPTIONS["PaymentMethod"], key="PaymentMethod")
            c2.toggle("Paperless billing", key="PaperlessBilling")
            st.slider("Monthly charges ($)", 18.0, 120.0, step=0.5, key="MonthlyCharges")
            c1, c2 = st.columns([1, 1.4], vertical_alignment="bottom")
            c1.toggle("Auto-estimate total", key="auto_total",
                      help="Total charges = tenure × monthly charges")
            c2.number_input("Total charges to date ($)", min_value=0.0, step=10.0, key="TotalCharges",
                            disabled=st.session_state.auto_total)

        with st.container(border=True):
            st.markdown(section("📶 Services"), unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            c1.toggle("Phone service", key="PhoneService")
            c2.toggle("Multiple lines", key="MultipleLines", disabled=not st.session_state.PhoneService)
            st.segmented_control("Internet", CATEGORY_OPTIONS["InternetService"], key="InternetService")
            no_internet = (st.session_state.InternetService or "No") == "No"
            cols = st.columns(3)
            for i, addon in enumerate(INTERNET_ADDONS):
                cols[i % 3].toggle(pretty(addon), key=addon, disabled=no_internet)

    customer = collect_customer()
    result = predictor.analyze(customer)
    p, thr = result["churn_probability"], result["threshold"]

    with result_col:
        with st.container(border=True):
            st.markdown(section("🎯 Churn risk"), unsafe_allow_html=True)
            st.plotly_chart(gauge(p, thr), width="stretch", config={"displayModeBar": False})
            color = RISK_COLORS[result["risk_level"]]
            headline = "🚨 Likely to churn" if result["will_churn"] else "🛡️ Likely to stay"
            st.markdown(
                f'<div class="verdict" style="--c:{color}"><div class="verdict-big">{headline}</div>'
                f'<div class="verdict-small">{result["risk_level"]} risk · {p:.0%} churn probability · '
                f'decision line at {thr:.0%} (white marker)</div></div>',
                unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown(section("🧩 Why this score"), unsafe_allow_html=True)
            st.plotly_chart(reasons_chart(result["reasons"]), width="stretch",
                            config={"displayModeBar": False})

        with st.container(border=True):
            st.markdown(section("🛟 Retention playbook"), unsafe_allow_html=True)
            html = "".join(
                f'<div class="action"><div><div class="action-title">{a["title"]}</div>'
                f'<div class="action-detail">{a["detail"]}</div></div>'
                + (f'<span class="action-driver">driver: {pretty(a["driver"])}</span>' if "driver" in a else "")
                + "</div>"
                for a in result["recommended_actions"])
            st.markdown(html, unsafe_allow_html=True)

# ---------------------------------------------------------------- Batch Radar
with tab_batch:
    c1, c2 = st.columns([2, 1], vertical_alignment="bottom")
    upload = c1.file_uploader("Upload customers: a PDF with a customer table, or a CSV",
                              type=["pdf", "csv"])
    with c2:
        if st.button("✨ Scan the demo dataset (7,043 customers)", width="stretch"):
            st.session_state.batch_raw = load_raw()
            st.session_state.batch_source = None
        st.download_button("📄 Get a sample PDF to try", SAMPLE_PDF.read_bytes(),
                           "customers_sample.pdf", "application/pdf", width="stretch")

    batch_error = None
    if upload is not None and st.session_state.get("batch_source") != upload.file_id:
        try:
            with st.spinner(f"Reading {upload.name}..."):
                st.session_state.batch_raw = read_customer_file(upload.name, upload.getvalue())
            st.session_state.batch_source = upload.file_id
            st.toast(f"Read {len(st.session_state.batch_raw):,} customers from {upload.name}", icon="📄")
        except Exception as e:  # unreadable file: show it instead of crashing the app
            batch_error = f"Couldn't read {upload.name}: {e}"

    raw = st.session_state.get("batch_raw")
    if raw is not None and not batch_error:
        try:
            validate_columns(raw)
        except ValueError as e:
            batch_error = str(e)
    if batch_error:
        st.error(batch_error)
    elif raw is None:
        st.info("Upload a PDF or CSV, or scan the demo dataset, to light up the radar.", icon="📡")
    else:
        df = score(raw)
        st.session_state.batch_scored = df
        churners = df[df["will_churn"]]
        k1, k2, k3, k4 = st.columns(4)
        k1.markdown(kpi("Customers scanned", f"{len(df):,}", "rows scored", CYAN), unsafe_allow_html=True)
        k2.markdown(kpi("Predicted churners", f"{len(churners):,}",
                        f"{len(churners) / len(df):.1%} of customers", DANGER), unsafe_allow_html=True)
        k3.markdown(kpi("High risk", f"{(df['risk_level'] == 'High').sum():,}",
                        "probability ≥ 60%", WARN), unsafe_allow_html=True)
        k4.markdown(kpi("Revenue at risk", f"${churners['MonthlyCharges'].sum():,.0f}",
                        "per month, from predicted churners", VIOLET), unsafe_allow_html=True)
        st.write("")

        g1, g2, g3 = st.columns([1.4, 1, 1.3])
        with g1, st.container(border=True):
            st.markdown(section("Risk distribution"), unsafe_allow_html=True)
            fig = px.histogram(df, x="churn_probability", color="risk_level", nbins=40,
                               color_discrete_map=RISK_COLORS,
                               category_orders={"risk_level": ["Low", "Medium", "High"]})
            fig.add_vline(x=predictor.threshold, line_dash="dash", line_color="#fff",
                          annotation_text="decision line", annotation_position="top left",
                          annotation_font_color="#fff")
            fig = style_fig(fig, 300)
            fig.update_layout(xaxis_title="churn probability", yaxis_title="customers", bargap=0.05,
                              legend_title_text="", legend_y=1.18, margin_t=50)
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        with g2, st.container(border=True):
            st.markdown(section("Risk mix"), unsafe_allow_html=True)
            counts = df["risk_level"].value_counts().reindex(["High", "Medium", "Low"]).fillna(0)
            fig = go.Figure(go.Pie(labels=counts.index, values=counts.values, hole=0.68, sort=False,
                                   marker=dict(colors=[RISK_COLORS[r] for r in counts.index],
                                               line=dict(color="#070B16", width=3))))
            fig.add_annotation(text=f"<b>{df['churn_probability'].mean():.0%}</b><br>avg risk",
                               showarrow=False, font=dict(size=18, color="#fff"))
            st.plotly_chart(style_fig(fig, 300), width="stretch", config={"displayModeBar": False})
        with g3, st.container(border=True):
            st.markdown(section("Heat map: contract × internet"), unsafe_allow_html=True)
            heat = df.pivot_table(index="Contract", columns="InternetService",
                                  values="churn_probability", aggfunc="mean")
            fig = px.imshow(heat, text_auto=".0%", aspect="auto",
                            color_continuous_scale=[[0, "#0F2A3A"], [0.5, VIOLET], [1, DANGER]])
            fig.update_layout(coloraxis_showscale=False, xaxis_title="", yaxis_title="")
            st.plotly_chart(style_fig(fig, 300), width="stretch", config={"displayModeBar": False})

        with st.container(border=True):
            st.markdown(section("🎯 Watchlist"), unsafe_allow_html=True)
            f1, f2, f3 = st.columns([1.3, 1, 1.4], vertical_alignment="bottom")
            levels = f1.pills("Risk level", ["High", "Medium", "Low"], selection_mode="multi",
                              default=["High"])
            query = f2.text_input("Search customer ID", placeholder="e.g. 7590-VHVEG")
            view = df[df["risk_level"].isin(levels or ["High", "Medium", "Low"])]
            if query:
                view = view[view[ID_COL].astype(str).str.contains(query, case=False)]
            view = view.sort_values("churn_probability", ascending=False)
            pick = f3.selectbox("Deep-dive a customer", view[ID_COL].head(200), index=None,
                                placeholder="choose to open in Customer Lab")
            if pick:
                st.button(f"🔬 Open {pick} in Customer Lab", on_click=load_customer_by_id, args=(pick,))
            cols = [ID_COL, "churn_probability", "risk_level", "Contract", "tenure", "MonthlyCharges",
                    "InternetService", "PaymentMethod"]
            st.dataframe(
                view[cols], hide_index=True, height=380, width="stretch",
                column_config={
                    "churn_probability": st.column_config.ProgressColumn(
                        "Churn risk", min_value=0.0, max_value=1.0, format="percent"),
                    "MonthlyCharges": st.column_config.NumberColumn("Monthly $", format="$%.2f"),
                    "tenure": st.column_config.NumberColumn("Tenure (mo)"),
                    "risk_level": "Risk", ID_COL: "Customer",
                },
            )
            st.download_button("⬇️ Download scored CSV", df.to_csv(index=False).encode(),
                               "churn_scores.csv", "text/csv")

# ---------------------------------------------------------------- Model Insights
with tab_model:
    m = report["metrics"]
    cols = st.columns(5)
    cards = [("ROC-AUC", m["roc_auc"], "ranking quality", CYAN),
             ("Recall", m["recall"], "churners caught", DANGER),
             ("Precision", m["precision"], "alerts that are right", WARN),
             ("F1", m["f1"], "recall/precision balance", VIOLET),
             ("PR-AUC", m["pr_auc"], "on imbalanced data", SAFE)]
    for col, (label, val, hint, acc) in zip(cols, cards):
        col.markdown(kpi(label, f"{val:.3f}", hint, acc), unsafe_allow_html=True)
    st.caption(f"Held-out test set of {report['n_test']:,} customers · model **{report['model_name']}** · "
               f"threshold {report['threshold']:.2f} (picked by max F1 on cross-validated training "
               f"predictions) · trained {report['trained_at']}")

    r1, r2 = st.columns(2)
    with r1, st.container(border=True):
        st.markdown(section("ROC curve"), unsafe_allow_html=True)
        roc = report["roc_curve"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], fill="tozeroy", mode="lines",
                                 line=dict(color=CYAN, width=3), fillcolor="rgba(34,211,238,0.15)",
                                 name=f"model (AUC {m['roc_auc']:.3f})"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="random guess",
                                 line=dict(color=MUTED, dash="dot")))
        fig.update_layout(xaxis_title="False positive rate", yaxis_title="True positive rate")
        st.plotly_chart(style_fig(fig, 340), width="stretch", config={"displayModeBar": False})
    with r2, st.container(border=True):
        st.markdown(section("Confusion matrix"), unsafe_allow_html=True)
        cm = report["confusion_matrix"]
        z = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
        text = [[f"✅ {cm['tn']}<br>stayed, predicted stay", f"⚠️ {cm['fp']}<br>false alarm"],
                [f"💔 {cm['fn']}<br>missed churner", f"🎯 {cm['tp']}<br>caught churner"]]
        fig = go.Figure(go.Heatmap(z=z, x=["Predicted stay", "Predicted churn"],
                                   y=["Actually stayed", "Actually churned"], text=text,
                                   texttemplate="%{text}", textfont=dict(size=14),
                                   colorscale=[[0, "#121A33"], [1, VIOLET]], showscale=False))
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(style_fig(fig, 340), width="stretch", config={"displayModeBar": False})

    r1, r2 = st.columns([1.3, 1])
    with r1, st.container(border=True):
        st.markdown(section("What drives churn overall (mean |SHAP|)"), unsafe_allow_html=True)
        imp = pd.Series(report["feature_importance"]).head(12)[::-1]
        fig = go.Figure(go.Bar(x=imp.values, y=[pretty(i) for i in imp.index], orientation="h",
                               marker=dict(color=imp.values, colorscale=[[0, CYAN], [1, VIOLET]],
                                           cornerradius=6)))
        st.plotly_chart(style_fig(fig, 400), width="stretch", config={"displayModeBar": False})
    with r2, st.container(border=True):
        st.markdown(section("Model leaderboard (5-fold CV ROC-AUC)"), unsafe_allow_html=True)
        lb = pd.Series({k: v["cv_roc_auc"] for k, v in report["leaderboard"].items()}).sort_values()
        fig = go.Figure(go.Bar(
            x=lb.values, y=lb.index, orientation="h", text=[f"{v:.4f}" for v in lb.values],
            textposition="outside",
            marker=dict(color=[VIOLET if k == report["model_name"] else "rgba(255,255,255,0.18)"
                               for k in lb.index], cornerradius=6)))
        fig.update_xaxes(range=[lb.min() - 0.02, lb.max() + 0.01])
        st.plotly_chart(style_fig(fig, 400), width="stretch", config={"displayModeBar": False})

# ---------------------------------------------------------------- Data Explorer
with tab_data:
    data = get_dataset()
    data = data.assign(tenure_bucket=add_features(data)["tenure_bucket"])
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(kpi("Customers", f"{len(data):,}", "Telco dataset", CYAN), unsafe_allow_html=True)
    k2.markdown(kpi("Churn rate", f"{data[TARGET].mean():.1%}", "class imbalance ≈ 1 : 2.8", DANGER),
                unsafe_allow_html=True)
    k3.markdown(kpi("Avg tenure", f"{data['tenure'].mean():.0f} mo", "months with the company", VIOLET),
                unsafe_allow_html=True)
    k4.markdown(kpi("Avg bill", f"${data['MonthlyCharges'].mean():.0f}", "per month", SAFE),
                unsafe_allow_html=True)
    st.write("")

    c1, c2 = st.columns(2)
    with c1, st.container(border=True):
        st.markdown(section("Churn rate by segment"), unsafe_allow_html=True)
        seg_options = ["Contract", "tenure_bucket", "InternetService", "PaymentMethod", "TechSupport",
                       "OnlineSecurity", "PaperlessBilling", "SeniorCitizen", "Partner", "Dependents"]
        seg = st.selectbox("Segment by", seg_options, format_func=pretty)
        rates = data.groupby(seg)[TARGET].agg(["mean", "size"]).reset_index().sort_values("mean")
        fig = go.Figure(go.Bar(
            x=rates["mean"], y=rates[seg].astype(str), orientation="h",
            text=[f"{r:.0%}  ·  n={n:,}" for r, n in zip(rates["mean"], rates["size"])],
            textposition="auto",
            marker=dict(color=rates["mean"], colorscale=[[0, SAFE], [0.5, WARN], [1, DANGER]],
                        cmin=0, cmax=0.5, cornerradius=6)))
        fig.add_vline(x=data[TARGET].mean(), line_dash="dot", line_color="#fff",
                      annotation_text="overall", annotation_font_color="#fff")
        fig.update_xaxes(tickformat=".0%")
        st.plotly_chart(style_fig(fig, 360), width="stretch", config={"displayModeBar": False})
    with c2, st.container(border=True):
        st.markdown(section("Distribution: churned vs stayed"), unsafe_allow_html=True)
        num = st.selectbox("Numeric feature", ["tenure", "MonthlyCharges", "TotalCharges"], format_func=pretty)
        plot_df = data.assign(Outcome=data[TARGET].map({1: "Churned", 0: "Stayed"}))
        fig = px.histogram(plot_df, x=num, color="Outcome", barmode="overlay", nbins=40,
                           histnorm="percent", opacity=0.7,
                           color_discrete_map={"Churned": DANGER, "Stayed": CYAN})
        fig.update_layout(xaxis_title=pretty(num), yaxis_title="% of group")
        st.plotly_chart(style_fig(fig, 360), width="stretch", config={"displayModeBar": False})
