import json
import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from features import engineer, CITIES

st.set_page_config(page_title="MMR Property Valuation", layout="wide")

# ------------------------------------------------------------------ theme
THEMES = {
    "light": dict(bg="#f4f5f7", card="#ffffff", field="#ffffff", ink="#18212b", muted="#5d6978", line="#dde1e7",
                  accent="#1d4e5f", accent_hi="#163c49", on_accent="#ffffff", soft="#e3edf0", head="#fafbfc",
                  row="#eef0f3", bar="#b9c2cc", grid="#e8ebef", hover="#eef3f5"),
    "dark": dict(bg="#0e1318", card="#151c24", field="#1b242e", ink="#e8ecf1", muted="#9aa6b5", line="#273341",
                 accent="#6fb7cb", accent_hi="#8ac7d8", on_accent="#0b1a20", soft="#1d2e36", head="#18212b",
                 row="#1e2833", bar="#3b4856", grid="#222d39", hover="#212c38"),
}
if "theme" not in st.session_state:
    st.session_state["theme"] = "light"
T = THEMES[st.session_state["theme"]]

css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Source+Serif+4:wght@500;600&display=swap');
:root{--bg:%(bg)s;--card:%(card)s;--field:%(field)s;--ink:%(ink)s;--muted:%(muted)s;--line:%(line)s;--accent:%(accent)s;
--accent-hi:%(accent_hi)s;--on-accent:%(on_accent)s;--soft:%(soft)s;--head:%(head)s;--row:%(row)s;--hover:%(hover)s}
html,body,.stApp,[data-testid="stAppViewContainer"],[data-testid="stMain"]{background:var(--bg)!important;color:var(--ink);font-family:'Inter',system-ui,sans-serif}
#MainMenu,footer,header[data-testid="stHeader"],[data-testid="stToolbar"],[data-testid="stDecoration"]{display:none!important}
.block-container{max-width:1180px;padding:2rem 2rem 3rem}
p,span,label,li,div{color:inherit}
.topbar{border-bottom:1px solid var(--line);padding-bottom:1.1rem;margin-bottom:1.8rem;display:flex;justify-content:space-between;align-items:flex-end;gap:2rem}
.eyebrow{font-size:.72rem;font-weight:600;letter-spacing:.12em;text-transform:uppercase;color:var(--accent)}
.title{font-family:'Source Serif 4',Georgia,serif;font-size:2rem;font-weight:600;line-height:1.15;margin:.25rem 0 0;color:var(--ink)}
.sub{color:var(--muted);font-size:.9rem;max-width:32rem;text-align:right}
.panel-h{font-size:.72rem;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);margin:0 0 .6rem}
div[data-testid="stForm"]{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:1.3rem;box-shadow:none}
[data-testid="stWidgetLabel"] p,[data-testid="stWidgetLabel"] label{font-size:.82rem!important;font-weight:500!important;color:var(--ink)!important}
div[data-baseweb="select"]>div,div[data-baseweb="input"],div[data-baseweb="base-input"]{background:var(--field)!important;border:1px solid var(--line)!important;border-radius:4px!important}
div[data-baseweb="select"] *,div[data-baseweb="input"] input{color:var(--ink)!important;-webkit-text-fill-color:var(--ink)!important}
div[data-baseweb="select"] *,div[data-baseweb="input"] input,div[data-baseweb="base-input"] input{background-color:transparent!important}
div[data-baseweb="select"] svg{fill:var(--muted)!important}
[data-testid="stNumberInput"] button{background:var(--field)!important;color:var(--ink)!important;border:0!important}
[data-testid="stNumberInput"] button svg{fill:var(--ink)!important}
div[data-baseweb="popover"] ul,div[data-baseweb="menu"]{background:var(--card)!important;border:1px solid var(--line)}
div[data-baseweb="popover"] li{background:var(--card)!important;color:var(--ink)!important}
div[data-baseweb="popover"] li:hover,div[data-baseweb="popover"] li[aria-selected="true"]{background:var(--hover)!important}
[data-testid="stFormSubmitButton"] button,[data-testid="stButton"] button{border-radius:4px;font-weight:600;font-size:.9rem}
[data-testid="stFormSubmitButton"] button{width:100%%;background:var(--accent)!important;border:0!important;padding:.65rem 1rem}
[data-testid="stFormSubmitButton"] button p{color:var(--on-accent)!important}
[data-testid="stFormSubmitButton"] button:hover{background:var(--accent-hi)!important}
[data-testid="stButton"] button{background:transparent!important;border:1px solid var(--line)!important;padding:.4rem .9rem}
[data-testid="stButton"] button p{color:var(--ink)!important}
[data-testid="stButton"] button:hover{border-color:var(--accent)!important}
.card{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:1.5rem 1.6rem;margin-bottom:1rem}
.est-label{font-size:.78rem;color:var(--muted);font-weight:500}
.est{font-family:'Source Serif 4',Georgia,serif;font-size:3.1rem;font-weight:600;line-height:1.1;margin:.2rem 0 .1rem;color:var(--ink);font-variant-numeric:tabular-nums}
.est-ctx{color:var(--muted);font-size:.88rem}
.range{margin:1.5rem 0 .5rem;position:relative;height:6px;background:var(--soft);border-radius:3px}
.range i{position:absolute;left:12%%;right:12%%;top:0;bottom:0;background:var(--accent);opacity:.45;border-radius:3px}
.range b{position:absolute;left:50%%;top:-4px;width:2px;height:14px;background:var(--accent)}
.rlabels{display:flex;justify-content:space-between;font-size:.8rem;color:var(--muted);font-variant-numeric:tabular-nums}
.stats{display:grid;grid-template-columns:repeat(3,1fr);border:1px solid var(--line);border-radius:6px;background:var(--card);margin-bottom:1rem}
.stat{padding:1rem 1.2rem;border-right:1px solid var(--line)}.stat:last-child{border-right:0}
.stat .k{font-size:.74rem;color:var(--muted);font-weight:500}
.stat .v{font-size:1.2rem;font-weight:600;margin-top:.15rem;color:var(--ink);font-variant-numeric:tabular-nums}
.stat .d{font-size:.78rem;color:var(--muted);margin-top:.1rem}
.empty{border:1px dashed var(--line);border-radius:6px;padding:3.2rem 1.6rem;text-align:center;color:var(--muted);background:var(--card)}
.empty b{display:block;color:var(--ink);font-size:1.02rem;margin-bottom:.35rem}
.section{font-family:'Source Serif 4',Georgia,serif;font-size:1.25rem;font-weight:600;margin:2.2rem 0 .2rem;color:var(--ink)}
.note{color:var(--muted);font-size:.85rem;margin-bottom:.8rem}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin:.6rem 0 1rem}
.kpi{border-top:2px solid var(--accent);padding-top:.6rem}
.kpi .v{font-size:1.5rem;font-weight:600;color:var(--ink);font-variant-numeric:tabular-nums}.kpi .k{font-size:.78rem;color:var(--muted)}
.tbl{width:100%%;border-collapse:collapse;font-size:.88rem;background:var(--card);border:1px solid var(--line)}
.tbl th{text-align:right;font-size:.74rem;font-weight:600;color:var(--muted);padding:.65rem 1rem;border-bottom:1px solid var(--line);background:var(--head)}
.tbl td{text-align:right;padding:.6rem 1rem;border-bottom:1px solid var(--row);color:var(--ink);font-variant-numeric:tabular-nums}
.tbl th:first-child,.tbl td:first-child{text-align:left}.tbl tr:last-child td{border-bottom:0}
.disc{border-top:1px solid var(--line);margin-top:2.4rem;padding-top:1rem;color:var(--muted);font-size:.78rem;line-height:1.55}
.disc b{color:var(--ink)}
@media(max-width:800px){.topbar{display:block}.sub{text-align:left;margin-top:.6rem}.stats,.kpis{grid-template-columns:1fr 1fr}.est{font-size:2.4rem}}
</style>
""" % T
st.markdown(css, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading model")
def load():
    return joblib.load("models/model.joblib"), json.load(open("models/metrics.json"))


B, M = load()
meta = B["meta"]
REP = {"Thane": "Thane West", "Navi Mumbai": "Kharghar", "Panvel": "Panvel", "Mira-Bhayandar": "Mira Road East",
       "Kalyan-Dombivli": "Dombivali", "Vasai-Virar": "Virar", "Mumbai - Western Suburbs": "Andheri West",
       "Mumbai - Eastern Suburbs": "Chembur", "Mumbai - South": "Worli"}


def inr(x: float) -> str:
    return f"₹{x / 1e7:.2f} Cr" if x >= 1e7 else f"₹{x / 1e5:.1f} L"


def predict(region, project, bhk, area, typ, status, age) -> float:
    row = pd.DataFrame([{"region": region, "locality": project, "bhk": bhk, "area": area,
                         "type": typ, "status": status, "age": age}])
    X = engineer(row)
    X["locality"] = X["locality"].where(X["locality"].isin(B["keep_loc"]), "Other")
    cols = B["cat"] + B["num"]
    p = sum(w * (B["models"][k].predict(X) if hasattr(B["models"][k], "named_steps")
                 else B["models"][k].predict(X[cols]))[0] for k, w in B["weights"].items())
    return float(np.exp(p))


# ------------------------------------------------------------------ header
hl, hr = st.columns([6, 1])
with hl:
    st.markdown(f"""
<div class="topbar">
  <div><div class="eyebrow">Mumbai Metropolitan Region · Residential</div>
  <div class="title">Property Valuation Estimator</div></div>
  <div class="sub">Machine-learning price estimates for Mumbai, Thane, Navi Mumbai, Panvel and neighbouring
  nodes, trained on {M['n_listings']:,} market listings.</div>
</div>""", unsafe_allow_html=True)
with hr:
    dark = st.session_state["theme"] == "dark"
    if st.button("Light mode" if dark else "Dark mode", key="themebtn", use_container_width=True):
        st.session_state["theme"] = "light" if dark else "dark"
        st.rerun()

left, right = st.columns([5, 7], gap="large")

with left:
    st.markdown('<div class="panel-h">Property details</div>', unsafe_allow_html=True)
    city = st.selectbox("City / zone", [c for c in CITIES if c in meta["regions"]], index=CITIES.index("Thane"))
    with st.form("f", border=True):
        regs = meta["regions"][city]
        region = st.selectbox("Locality", regs, index=regs.index(REP[city]) if REP.get(city) in regs else 0)
        project = st.selectbox("Project (optional)", ["Not listed"] + meta["projects"].get(region, []))
        c1, c2 = st.columns(2)
        bhk = c1.selectbox("Bedrooms (BHK)", [1, 2, 3, 4, 5, 6], index=1)
        area = c2.number_input("Area (sq ft)", 150, 6000, 650, step=10)
        typ = st.selectbox("Property type", ["Apartment", "Studio Apartment", "Villa", "Independent House", "Penthouse"])
        c3, c4 = st.columns(2)
        status = c3.selectbox("Possession", ["Ready to move", "Under Construction"])
        age = c4.selectbox("Listing", ["New", "Resale", "Unknown"])
        go = st.form_submit_button("Calculate estimate")

    if go:
        proj = "Other" if project == "Not listed" else project
        args = (bhk, area, typ, status, age)
        p = predict(region, proj, *args)
        comp = []
        for c in meta["regions"]:
            r = region if c == city else REP.get(c)
            if r in meta["regions"][c]:
                comp.append({"zone": c.replace("Mumbai - ", "Mumbai "),
                             "price": (p if c == city else predict(r, "Other", *args)) / 1e5, "sel": c == city})
        st.session_state["res"] = dict(p=p, city=city, region=region, area=area, bhk=bhk, comp=comp)

with right:
    res = st.session_state.get("res")
    st.markdown('<div class="panel-h">Valuation</div>', unsafe_allow_html=True)
    if not res:
        st.markdown('<div class="empty"><b>No estimate yet</b>'
                    'Enter the property details and select Calculate estimate.</div>', unsafe_allow_html=True)
    else:
        p, area_ = res["p"], res["area"]
        e = M["median_ape_pct"] / 100 * 1.5
        mp = meta["median_ppsf"].get(res["region"])
        vs = f"{(p / area_ / mp - 1) * 100:+.0f}% vs locality median" if mp else ""
        mp_txt = f"₹{mp:,.0f}" if mp else "—"
        st.markdown(f"""
<div class="card">
  <div class="est-label">Estimated market price</div>
  <div class="est">{inr(p)}</div>
  <div class="est-ctx">{res['bhk']} BHK · {area_:,} sq ft · {res['region']}, {res['city']}</div>
  <div class="range"><i></i><b></b></div>
  <div class="rlabels"><span>{inr(p * (1 - e))}</span><span>Likely range</span><span>{inr(p * (1 + e))}</span></div>
</div>
<div class="stats">
  <div class="stat"><div class="k">Price per sq ft</div><div class="v">₹{p / area_:,.0f}</div><div class="d">{vs}</div></div>
  <div class="stat"><div class="k">Locality median</div><div class="v">{mp_txt}</div><div class="d">per sq ft, listings</div></div>
  <div class="stat"><div class="k">Typical model error</div><div class="v">±{M['median_ape_pct']:.0f}%</div><div class="d">median, unseen data</div></div>
</div>""", unsafe_allow_html=True)

        df = pd.DataFrame(res["comp"]).sort_values("price")
        ch = (alt.Chart(df).mark_bar(size=18, cornerRadiusEnd=2)
              .encode(y=alt.Y("zone:N", sort=list(df.zone), title=None,
                              axis=alt.Axis(labelFontSize=12, labelLimit=220, labelColor=T["ink"], domain=False, ticks=False)),
                      x=alt.X("price:Q", title="Estimated price (₹ lakh)",
                              axis=alt.Axis(grid=True, gridColor=T["grid"], labelColor=T["muted"], titleColor=T["muted"], domain=False)),
                      color=alt.condition("datum.sel", alt.value(T["accent"]), alt.value(T["bar"])),
                      tooltip=[alt.Tooltip("zone", title="Zone"), alt.Tooltip("price:Q", title="₹ lakh", format=",.1f")])
              .properties(height=300).configure_view(strokeWidth=0).configure(background="transparent"))
        st.markdown('<div class="panel-h" style="margin-top:1.4rem">Same property across the region</div>',
                    unsafe_allow_html=True)
        st.altair_chart(ch, use_container_width=True)

# ------------------------------------------------------------------ model section
st.markdown('<div class="section">Model performance</div>'
            '<div class="note">Measured on listings held out from training.</div>', unsafe_allow_html=True)
st.markdown(f"""
<div class="kpis">
  <div class="kpi"><div class="v">{M['holdout_r2_log']:.3f}</div><div class="k">R² (log price)</div></div>
  <div class="kpi"><div class="v">{M['median_ape_pct']:.1f}%</div><div class="k">Median absolute % error</div></div>
  <div class="kpi"><div class="v">{M['within_10pct']:.0f}%</div><div class="k">Predictions within ±10%</div></div>
  <div class="kpi"><div class="v">{M['within_20pct']:.0f}%</div><div class="k">Predictions within ±20%</div></div>
</div>""", unsafe_allow_html=True)
rows = "".join(
    f"<tr><td>{c.replace('Mumbai - ', 'Mumbai ')}</td><td>{v['n']:,}</td><td>{v['r2']:.2f}</td><td>{v['mape_pct']:.1f}%</td></tr>"
    for c, v in sorted(M["by_city"].items(), key=lambda kv: -kv[1]["n"]))
st.markdown(f"""<table class="tbl"><thead><tr><th>Zone</th><th>Test listings</th><th>R² (log price)</th><th>Mean error</th></tr></thead>
<tbody>{rows}</tbody></table>""", unsafe_allow_html=True)

st.markdown("""<div class="disc"><b>Method.</b> Weighted ensemble of LightGBM, XGBoost and CatBoost on locality, project, BHK, area,
possession status and listing age. <b>Data.</b> Asking prices from online listings (Makaan.com via Kaggle); duplicates and
extreme price-per-sq-ft outliers removed. Estimates are indicative, not a formal valuation or a registered sale price,
and should not be relied on for lending or legal purposes.</div>""", unsafe_allow_html=True)
