# python -m streamlit run app.py --server.address 127.0.0.1
from __future__ import annotations

import html
import textwrap
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from config import DATA_DIR, DORM_CS_PROFILE
from models.baseline import train_ridge_baseline, predict_baseline
from models.bandit import linucb_init
from pages.admin_page import render_admin_page
from pages.home_page import render_home_page
from pages.messages_page import render_mobile_messages_page
from pages.profile_page import render_profile_page
from pages.tasks_page import render_tasks_page
from services.replay_service import (
    batch_generate_scores_from_sim_user_pool,
    generate_simulated_user_pool_from_base_samples,
    load_user_pool_from_excel,
)
from state.user_progress import init_user_progress_state
from utils.data_utils import (
    fill_hours,
    generate_energy_sample,
    get_dorm_hourly,
    load_energy_csv,
    load_energy_df,
)

st.set_page_config(page_title="节能小助手 v4（LinUCB 完整版）", layout="wide")


# ===================== 通用工具 =====================
def active_date_str() -> str:
    return str(st.session_state.get("business_date_str", datetime.now().strftime("%Y-%m-%d")))


def _normalize_business_date(value):
    if value is None:
        return datetime.now().date()
    try:
        return pd.to_datetime(value).date()
    except Exception:
        return datetime.now().date()


def _shift_business_date_callback(days: int):
    current = _normalize_business_date(st.session_state.get("business_date_picker"))
    st.session_state["business_date_picker"] = current + timedelta(days=int(days))


def _set_today_callback():
    st.session_state["business_date_picker"] = datetime.now().date()


# ===================== 频控 =====================
def freq_guard_allow_daily(dorm_id: str, daily_cap: int, inter_logs: pd.DataFrame) -> tuple[bool, int]:
    """判断某宿舍当天是否还能继续发送干预。"""
    if inter_logs is None or inter_logs.empty:
        return True, 0

    need_cols = {"dorm_id", "date"}
    if not need_cols.issubset(inter_logs.columns):
        return True, 0

    logs_today = inter_logs[
        (inter_logs["dorm_id"].astype(str) == str(dorm_id))
        & (inter_logs["date"].astype(str) == active_date_str())
    ]
    count = int(len(logs_today))
    return count < int(daily_cap), count


def already_decided_today(dorm_id: str, inter_logs: pd.DataFrame) -> bool:
    """判断某宿舍当天是否已经有一条 LinUCB 决策记录。"""
    if inter_logs is None or inter_logs.empty:
        return False

    need_cols = {"dorm_id", "date", "algo_type"}
    if not need_cols.issubset(inter_logs.columns):
        return False

    logs_today = inter_logs[
        (inter_logs["dorm_id"].astype(str) == str(dorm_id))
        & (inter_logs["date"].astype(str) == active_date_str())
        & (inter_logs["algo_type"].astype(str) == "LinUCB")
    ]
    return len(logs_today) > 0


# ===================== 交互日志 =====================
def log_interaction(event_name: str, extra: dict | None = None):
    st.session_state.setdefault("interaction_logs", [])
    st.session_state["interaction_logs"].append(
        {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "event_name": event_name,
            "extra_json": {} if extra is None else extra,
        }
    )


# ===================== session_state 初始化 =====================
def init_session_state():
    defaults = {
        "messages": [],
        "interaction_logs": [],
        "intervention_logs": [],
        "scores": [],
        "cluster_arms_config": None,
        "rl_arm_last_metrics": [],
        "demo_time_mode": True,
        "daily_cap": 1,
        "current_dorm_for_messages": "",
        "user_progress_by_dorm": {},
        "tasks_checkin_log": [],
        "today_checked_in": False,
        "streak_days": 0,
        "weekly_checkins": [],
        "earned_xp": 0,
        "total_xp": 0,
        "unlocked_badges": [],
        # 手机端 Tab 路由（Home / Tasks / Messages / Profile）
        "mobile_tab": "Home",
    }
    for key, default in defaults.items():
        st.session_state.setdefault(key, default)

    if "linucb_params" not in st.session_state:
        st.session_state["linucb_params"] = linucb_init(alpha=0.8)

    if "business_date_picker" not in st.session_state:
        st.session_state["business_date_picker"] = datetime.now().date()

    business_date = _normalize_business_date(st.session_state.get("business_date_picker"))
    st.session_state["business_date"] = business_date
    st.session_state["business_date_str"] = pd.to_datetime(business_date).strftime("%Y-%m-%d")
    init_user_progress_state()


def ensure_default_demo_session():
    """Prepare an in-memory demo pool for mobile cold starts."""
    st.session_state.setdefault("sim_target_n", 48)
    st.session_state.setdefault("sim_noise_scale", 0.03)
    st.session_state.setdefault("sim_pool_seed", 42)
    st.session_state.setdefault("tau_batch_scores", 0.30)

    sim_user_pool_df = st.session_state.get("sim_user_pool_df")
    scores = st.session_state.get("scores", [])
    has_pool = (
        sim_user_pool_df is not None
        and hasattr(sim_user_pool_df, "empty")
        and not sim_user_pool_df.empty
    )
    has_scores = isinstance(scores, list) and len(scores) > 0
    if has_pool and has_scores:
        st.session_state["_default_demo_session_ready"] = True
        return
    if st.session_state.get("_default_demo_session_ready", False):
        return

    try:
        base_user_pool_df = load_user_pool_from_excel(
            excel_path=Path(DATA_DIR) / "user_pool.xlsx",
            sheet_name="Sheet2",
        )
        sim_user_pool_df = generate_simulated_user_pool_from_base_samples(
            base_user_pool_df=base_user_pool_df,
            target_n=int(st.session_state.get("sim_target_n", 48)),
            noise_scale=float(st.session_state.get("sim_noise_scale", 0.03)),
            random_state=int(st.session_state.get("sim_pool_seed", 42)),
            id_prefix="SIM",
        )
        scores_from_sim_pool, _msg = batch_generate_scores_from_sim_user_pool(
            sim_user_pool_df=sim_user_pool_df,
            tau=float(st.session_state.get("tau_batch_scores", 0.30)),
        )
        st.session_state["sim_user_pool_df"] = sim_user_pool_df
        st.session_state["scores"] = scores_from_sim_pool
        st.session_state["_default_demo_session_ready"] = True
    except Exception as exc:
        st.session_state["_default_demo_session_error"] = str(exc)


# ===================== baseline / outcome =====================
def get_model_for_dorm(df_all: pd.DataFrame, dorm_id: str, ridge_alpha: float = 1.0):
    """为单个 dorm 训练 baseline 并返回预测结果。"""
    if df_all is None or df_all.empty:
        return None, pd.DataFrame(), pd.DataFrame(), {}

    dorm_df = get_dorm_hourly(df_all, dorm_id)
    if dorm_df is None or dorm_df.empty:
        return None, pd.DataFrame(), pd.DataFrame(), {}

    train_df = fill_hours(dorm_df, dorm_id)
    if train_df is None or train_df.empty:
        return None, pd.DataFrame(), pd.DataFrame(), {}

    model, _info = train_ridge_baseline(train_df, float(ridge_alpha))
    pred_df = predict_baseline(model, train_df)

    if "baseline_pred" not in pred_df.columns:
        pred_df["baseline_pred"] = 0.0

    dorm_out = pred_df.copy()
    dorm_out["timestamp_hour"] = pd.to_datetime(dorm_out["timestamp"], errors="coerce")
    dorm_out["reward_value"] = (
        pd.to_numeric(dorm_out.get("baseline_pred", 0), errors="coerce").fillna(0.0)
        - pd.to_numeric(dorm_out.get("energy_kwh", 0), errors="coerce").fillna(0.0)
    )

    ts = pd.to_datetime(dorm_out["timestamp_hour"], errors="coerce").dropna()
    if ts.empty:
        weekly_metrics = {}
    else:
        t_max = ts.max()
        t_min = t_max - pd.Timedelta(days=7)
        wk = dorm_out[dorm_out["timestamp_hour"] >= t_min].copy()
        weekly_metrics = {
            "week_start": t_min,
            "week_end": t_max,
            "actual_sum": float(pd.to_numeric(wk.get("energy_kwh", 0), errors="coerce").fillna(0.0).sum()),
            "baseline_sum": float(pd.to_numeric(wk.get("baseline_pred", 0), errors="coerce").fillna(0.0).sum()),
            "reward_sum": float(pd.to_numeric(wk.get("reward_value", 0), errors="coerce").fillna(0.0).sum()),
        }

    return model, train_df, dorm_out, weekly_metrics


def prepare_global_data(df_all: pd.DataFrame, ridge_alpha: float = 1.0):
    """基于导入后的小时级数据，为所有 dorm 准备 outcome。"""
    models_by_dorm: dict[str, object] = {}
    weekly_metrics_by_dorm: dict[str, dict] = {}
    outcome_frames: list[pd.DataFrame] = []

    if df_all is None or df_all.empty or "dorm_id" not in df_all.columns:
        return models_by_dorm, pd.DataFrame(), weekly_metrics_by_dorm

    dorm_ids = sorted(df_all["dorm_id"].dropna().astype(str).unique().tolist())
    for dorm_id in dorm_ids:
        model, _train_df, dorm_out, weekly_metrics = get_model_for_dorm(
            df_all, dorm_id, ridge_alpha=ridge_alpha
        )
        if model is not None:
            models_by_dorm[dorm_id] = model
        if weekly_metrics:
            weekly_metrics_by_dorm[dorm_id] = weekly_metrics
        if dorm_out is not None and not dorm_out.empty:
            outcome_frames.append(dorm_out)

    outcome_logs_df = pd.concat(outcome_frames, ignore_index=True) if outcome_frames else pd.DataFrame()
    return models_by_dorm, outcome_logs_df, weekly_metrics_by_dorm


# ===================== Sidebar 样式与展示 =====================
def inject_sidebar_style():
    st.markdown(
        textwrap.dedent(
            """
            <style>
            section[data-testid="stSidebar"] {
                background:
                    radial-gradient(circle at top left, rgba(44, 117, 76, 0.22), transparent 24%),
                    linear-gradient(180deg, #07110D 0%, #0A1712 42%, #0D1E17 100%);
                border-right: 1px solid rgba(124, 163, 139, 0.18);
            }

            section[data-testid="stSidebar"] .block-container {
                padding-top: 0.85rem;
                padding-bottom: 1.25rem;
                padding-left: 0.85rem;
                padding-right: 0.85rem;
            }

            section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
            section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p,
            section[data-testid="stSidebar"] label,
            section[data-testid="stSidebar"] .st-emotion-cache-10trblm,
            section[data-testid="stSidebar"] .st-emotion-cache-16idsys {
                color: #D7E5DD !important;
            }

            section[data-testid="stSidebar"] [data-testid="stRadio"] > div {
                gap: 0.44rem;
            }

            section[data-testid="stSidebar"] [data-testid="stRadio"] label {
                background: linear-gradient(180deg, rgba(255,255,255,0.06) 0%, rgba(255,255,255,0.035) 100%);
                border: 1px solid rgba(164, 197, 177, 0.16);
                border-radius: 16px;
                padding: 10px 12px;
                transition: all .16s ease;
                box-shadow: inset 0 1px 0 rgba(255,255,255,0.02);
            }

            section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
                border-color: rgba(138, 213, 156, 0.34);
                background: rgba(255,255,255,0.08);
                transform: translateY(-1px);
            }

            section[data-testid="stSidebar"] [data-testid="stRadio"] label p {
                color: #EAF4EE !important;
                font-weight: 600 !important;
            }

            section[data-testid="stSidebar"] [data-baseweb="input"] > div,
            section[data-testid="stSidebar"] [data-baseweb="select"] > div,
            section[data-testid="stSidebar"] [data-baseweb="input"] input,
            section[data-testid="stSidebar"] [data-baseweb="select"] input,
            section[data-testid="stSidebar"] [data-testid="stDateInput"] input,
            section[data-testid="stSidebar"] [data-testid="stNumberInput"] input,
            section[data-testid="stSidebar"] [data-testid="stTextInput"] input,
            section[data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] {
                background: rgba(255,255,255,0.08) !important;
                border: 1px solid rgba(164, 197, 177, 0.18) !important;
                border-radius: 14px !important;
                color: #F4FBF7 !important;
                box-shadow: none !important;
            }

            section[data-testid="stSidebar"] input,
            section[data-testid="stSidebar"] textarea,
            section[data-testid="stSidebar"] [data-baseweb="select"] * {
                color: #F4FBF7 !important;
                -webkit-text-fill-color: #F4FBF7 !important;
            }

            section[data-testid="stSidebar"] input::placeholder,
            section[data-testid="stSidebar"] textarea::placeholder {
                color: #A8B9B0 !important;
                -webkit-text-fill-color: #A8B9B0 !important;
            }

            section[data-testid="stSidebar"] [data-testid="stNumberInput"] button,
            section[data-testid="stSidebar"] [data-testid="stDateInput"] button,
            section[data-testid="stSidebar"] [data-testid="stSelectbox"] button {
                background: rgba(255,255,255,0.08) !important;
                color: #EAF7EF !important;
                border: 1px solid rgba(164, 197, 177, 0.14) !important;
            }

            section[data-testid="stSidebar"] [data-testid="stFileUploader"] {
                background: rgba(255,255,255,0.04);
                border: 1px dashed rgba(164, 197, 177, 0.24);
                border-radius: 16px;
                padding: 0.38rem 0.6rem;
            }

            section[data-testid="stSidebar"] [data-testid="stButton"] button {
                border-radius: 999px !important;
                font-weight: 700 !important;
                border: 1px solid rgba(152, 211, 162, 0.22) !important;
                background: linear-gradient(180deg, rgba(110,231,146,0.14) 0%, rgba(62,155,93,0.14) 100%) !important;
                color: #EAF7EF !important;
                box-shadow: none !important;
            }

            section[data-testid="stSidebar"] [data-testid="stButton"] button:hover {
                border-color: rgba(152, 211, 162, 0.42) !important;
                background: linear-gradient(180deg, rgba(110,231,146,0.20) 0%, rgba(62,155,93,0.20) 100%) !important;
                color: #FFFFFF !important;
            }

            section[data-testid="stSidebar"] [data-testid="stToggle"] label[data-testid="stWidgetLabel"] p {
                font-weight: 650 !important;
                color: #E4F2E9 !important;
            }

            section[data-testid="stSidebar"] [data-testid="stExpander"] details {
                background: linear-gradient(180deg, rgba(255,255,255,0.045) 0%, rgba(255,255,255,0.03) 100%);
                border: 1px solid rgba(164, 197, 177, 0.14);
                border-radius: 18px;
                overflow: hidden;
                margin-bottom: 0.6rem;
            }

            section[data-testid="stSidebar"] [data-testid="stExpander"] summary {
                font-weight: 800;
                color: #173126 !important;
                background: rgba(255,255,255,0.04) !important;
                padding-top: 0.45rem;
                padding-bottom: 0.45rem;
                padding-left: 0.7rem;
                padding-right: 0.7rem;
                border-radius: 14px 14px 0 0;
            }

            section[data-testid="stSidebar"] [data-testid="stExpander"] details[open] {
                border-color: rgba(126, 226, 154, 0.24);
                box-shadow: inset 0 1px 0 rgba(255,255,255,0.02), 0 10px 24px rgba(2,7,5,0.18);
            }

            section[data-testid="stSidebar"] [data-testid="stExpander"] details > div {
                padding-top: 0.3rem;
            }

            .sb-soft-divider {
                height: 1px;
                background: linear-gradient(90deg, rgba(93,126,108,0) 0%, rgba(93,126,108,.58) 18%, rgba(93,126,108,.58) 82%, rgba(93,126,108,0) 100%);
                margin: 0.7rem 0 0.95rem;
            }

            .sb-sec {
                margin: 0.12rem 0 0.52rem;
            }
            .sb-sec-kicker {
                font-size: 10px;
                font-weight: 800;
                letter-spacing: 0.14em;
                text-transform: uppercase;
                color: #7EE29A;
                margin-bottom: 0.16rem;
            }
            .sb-sec-title {
                font-size: 15px;
                font-weight: 800;
                line-height: 1.1;
                color: #F2FBF6;
                margin-bottom: 0.12rem;
                letter-spacing: -0.02em;
            }
            .sb-sec-desc {
                font-size: 11px;
                line-height: 1.55;
                color: #8FA79A;
            }
            </style>
            """
        ),
        unsafe_allow_html=True,
    )


def _latest_score_for_sidebar(dorm_id: str):
    scores = st.session_state.get("scores", [])
    latest = None
    for rec in scores:
        if not isinstance(rec, dict):
            continue
        if str(rec.get("dorm_id", "")) != str(dorm_id):
            continue
        if latest is None or str(rec.get("ts", "")) > str(latest.get("ts", "")):
            latest = rec
    return latest


def _today_sent_count_for_sidebar(dorm_id: str) -> int:
    logs = pd.DataFrame(st.session_state.get("intervention_logs", []))
    if logs.empty or "dorm_id" not in logs.columns or "date" not in logs.columns:
        return 0
    today = active_date_str()
    sub = logs[
        (logs["dorm_id"].astype(str) == str(dorm_id))
        & (logs["date"].astype(str) == today)
    ]
    return int(len(sub))


def _latest_arm_for_sidebar(dorm_id: str) -> str:
    logs = st.session_state.get("intervention_logs", [])
    latest = None
    for rec in logs:
        if not isinstance(rec, dict):
            continue
        if str(rec.get("dorm_id", "")) != str(dorm_id):
            continue
        ts = str(rec.get("timestamp", "") or rec.get("decision_time", ""))
        if latest is None or ts > latest[0]:
            latest = (ts, str(rec.get("arm_id", "")))
    return latest[1] if latest else "—"


def _sidebar_cluster_text(cluster_type: str) -> str:
    mapping = {"A": "A 类", "B": "B 类", "C": "C 类", "D": "D 类", "E": "E 类"}
    return mapping.get(str(cluster_type).upper(), "未识别")


def _sidebar_section_title(title: str, desc: str | None = None, kicker: str | None = None):
    kicker_html = f'<div class="sb-sec-kicker">{html.escape(kicker)}</div>' if kicker else ''
    desc_html = f'<div class="sb-sec-desc">{html.escape(desc)}</div>' if desc else ''
    st.markdown(
        f"""
        <div class="sb-sec">
            {kicker_html}
            <div class="sb-sec-title">{html.escape(title)}</div>
            {desc_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_status_card():
    daily_cap = int(st.session_state.get("daily_cap", 1))
    dorm_id = str(st.session_state.get("current_dorm_for_messages", "")).strip()

    if not dorm_id:
        dorm_value = "未选择"
        cluster_value = "—"
        sent_count = 0
        latest_arm = "—"
    else:
        dorm_value = dorm_id
        score = _latest_score_for_sidebar(dorm_id)
        cluster_value = _sidebar_cluster_text(score.get("cluster_type", "")) if score else "—"
        sent_count = _today_sent_count_for_sidebar(dorm_id)
        latest_arm = _latest_arm_for_sidebar(dorm_id)

    business_date_str = str(st.session_state.get("business_date_str", datetime.now().strftime("%Y-%m-%d")))

    hero_html = f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8">
      <style>
        * {{ box-sizing: border-box; }}
        html, body {{ margin: 0; padding: 0; background: transparent; font-family: Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }}
        .hero {{
          background:
            radial-gradient(circle at top left, rgba(126,226,154,.20), transparent 26%),
            linear-gradient(145deg, #08130F 0%, #0B1813 38%, #10241B 100%);
          color: #fff;
          border-radius: 22px;
          padding: 15px 14px 14px;
          border: 1px solid rgba(255,255,255,.06);
          box-shadow: 0 16px 30px rgba(5,14,10,.30);
          overflow: visible;
        }}
        .brand {{ display:flex; align-items:center; gap:10px; margin-bottom:12px; }}
        .logo {{ width:34px; height:34px; border-radius:11px; background: linear-gradient(135deg, #77F38D 0%, #5FD675 100%); color:#11271D; display:flex; align-items:center; justify-content:center; font-size:18px; font-weight:800; box-shadow: 0 8px 18px rgba(95,214,117,.25); }}
        .title {{ font-size:17px; font-weight:800; line-height:1.05; letter-spacing:-.02em; }}
        .sub {{ font-size:11px; color:rgba(255,255,255,.66); margin-top:2px; }}
        .panel {{ background: linear-gradient(180deg, rgba(255,255,255,.07) 0%, rgba(255,255,255,.045) 100%); border: 1px solid rgba(255,255,255,.08); border-radius: 17px; padding: 11px; backdrop-filter: blur(6px); }}
        .top {{ display:flex; align-items:center; justify-content:space-between; gap:10px; margin-bottom:10px; }}
        .eyebrow {{ font-size:11px; font-weight:700; color:rgba(255,255,255,.84); letter-spacing:.05em; }}
        .date {{ font-size:11px; color:rgba(255,255,255,.58); }}
        .grid {{ display:grid; grid-template-columns: 1fr 1fr; gap:8px; }}
        .item {{ background: rgba(255,255,255,.05); border-radius: 13px; padding: 10px 10px 9px; min-height: 68px; border: 1px solid rgba(255,255,255,.03); }}
        .label {{ font-size:10px; color:rgba(255,255,255,.52); margin-bottom:4px; }}
        .value {{ font-size:12px; line-height:1.35; font-weight:800; color:#F7FEFA; word-break:break-word; }}
        .footer {{ display:flex; gap:6px; flex-wrap:wrap; margin-top:10px; }}
        .pill {{ font-size:10px; font-weight:700; padding:4px 8px; border-radius:999px; background:rgba(114,224,125,.12); color:#BDF5C4; border:1px solid rgba(114,224,125,.16); }}
      </style>
    </head>
    <body>
      <div class="hero">
        <div class="brand">
          <div class="logo">⚡</div>
          <div>
            <div class="title">智控绿舍</div>
            <div class="sub">Dorm AC Intervention</div>
          </div>
        </div>
        <div class="panel">
          <div class="top">
            <div class="eyebrow">实验状态</div>
            <div class="date">{html.escape(business_date_str)}</div>
          </div>
          <div class="grid">
            <div class="item"><div class="label">当前宿舍</div><div class="value">{html.escape(str(dorm_value))}</div></div>
            <div class="item"><div class="label">当前群体</div><div class="value">{html.escape(str(cluster_value))}</div></div>
            <div class="item"><div class="label">今日频控</div><div class="value">{sent_count}/{daily_cap}</div></div>
            <div class="item"><div class="label">最新 arm</div><div class="value">{html.escape(str(latest_arm))}</div></div>
          </div>
          <div class="footer">
            <span class="pill">LinUCB</span>
            <span class="pill">Baseline</span>
            <span class="pill">Replay</span>
          </div>
        </div>
      </div>
    </body>
    </html>
    """
    components.html(hero_html, height=320, scrolling=False)


def _sidebar_soft_divider():
    st.markdown('<div class="sb-soft-divider"></div>', unsafe_allow_html=True)

# ===================== 启动初始化 =====================
init_session_state()
ensure_default_demo_session()
inject_sidebar_style()


# ===================== Sidebar 控件 =====================
df_all = None
load_stats = None

with st.sidebar:
    render_sidebar_status_card()

    _sidebar_section_title("导航", "用户端 · 手机视图 / 后台 Admin", kicker="Navigation")
    page = st.radio("页面", ["用户端", "Admin"], index=0, label_visibility="collapsed")

    _sidebar_soft_divider()
    with st.expander("Data｜数据来源", expanded=True):
        _sidebar_section_title("数据来源", "上传真实 CSV，或先用模拟数据驱动界面", kicker="Data")
        data_mode = st.radio(
            "选择数据来源",
            ["上传CSV", "使用模拟数据（>=2周）"],
            index=1,
            label_visibility="collapsed",
        )
        align_to_hour = st.toggle("导入时归整到整点（推荐开）", value=True)
        strict_hour = st.toggle("严格整点（分钟!=00则报错）", value=False)

        if data_mode == "上传CSV":
            uploaded = st.file_uploader("上传能耗CSV", type=["csv"])
            if uploaded is not None:
                df_all, load_stats = load_energy_csv(
                    uploaded,
                    align_to_hour=align_to_hour,
                    strict_hour=strict_hour,
                )
        else:
            sim_days = st.slider("模拟天数（>=14）", 14, 60, 20, 1)
            seed = st.number_input("随机种子", value=42, step=1)

            sim_user_pool_df = st.session_state.get("sim_user_pool_df")
            if (
                sim_user_pool_df is not None
                and not sim_user_pool_df.empty
                and "dorm_id" in sim_user_pool_df.columns
            ):
                sim_dorm_ids = (
                    sim_user_pool_df["dorm_id"].dropna().astype(str).unique().tolist()
                )
            else:
                sim_dorm_ids = list(DORM_CS_PROFILE.keys())

            df_sim = generate_energy_sample(
                n_days=int(sim_days),
                dorms=sim_dorm_ids,
                seed=int(seed),
            )
            df_all, load_stats = load_energy_df(
                df_sim,
                align_to_hour=align_to_hour,
                strict_hour=strict_hour,
            )

    with st.expander("Tuning｜实验参数", expanded=True):
        _sidebar_section_title("实验参数", "调节 bandit 与 baseline 的基础参数", kicker="Tuning")
        linucb_alpha = st.slider("LinUCB alpha", 0.1, 3.0, 1.0, 0.1)
        ridge_alpha = st.number_input("Ridge alpha", min_value=0.01, value=1.0, step=0.1)
        daily_cap = st.number_input(
            "每日最多干预次数",
            min_value=1,
            value=int(st.session_state.get("daily_cap", 1)),
            step=1,
            key="daily_cap",
        )

    with st.expander("Timeline｜实验日期", expanded=True):
        _sidebar_section_title("实验日期", "支持按天回看历史消息与干预记录", kicker="Timeline")
        step_days = st.selectbox("快速跳转步长", options=[1, 3, 7], index=0, key="business_date_step")
        st.date_input(
            "选择日期",
            value=_normalize_business_date(st.session_state.get("business_date_picker")),
            key="business_date_picker",
        )
        b1, b2, b3 = st.columns(3)
        with b1:
            st.button(
                f"前{step_days}天",
                use_container_width=True,
                on_click=_shift_business_date_callback,
                args=(-int(step_days),),
            )
        with b2:
            st.button("今天", use_container_width=True, on_click=_set_today_callback)
        with b3:
            st.button(
                f"后{step_days}天",
                use_container_width=True,
                on_click=_shift_business_date_callback,
                args=(int(step_days),),
            )

        business_date = _normalize_business_date(st.session_state.get("business_date_picker"))
        st.session_state["business_date"] = business_date
        st.session_state["business_date_str"] = pd.to_datetime(business_date).strftime("%Y-%m-%d")
        st.caption(f"当前演示日期：{st.session_state['business_date_str']}")

    with st.expander("Manage｜历史管理", expanded=False):
        _sidebar_section_title("历史管理", "清空消息与日志前请先确认是否需要保留实验痕迹", kicker="Manage")
        if st.button("清除消息历史", use_container_width=True):
            st.session_state["messages"] = []
            st.session_state["interaction_logs"] = []
            st.session_state["intervention_logs"] = []
            st.session_state["tasks_checkin_log"] = []
            st.session_state["user_progress_by_dorm"] = {}
            st.session_state["today_checked_in"] = False
            st.session_state["streak_days"] = 0
            st.session_state["weekly_checkins"] = []
            st.session_state["earned_xp"] = 0
            st.session_state["total_xp"] = 0
            st.session_state["unlocked_badges"] = []
            st.success("已清除消息、交互日志、干预日志和打卡进度历史。")
            st.rerun()


# ===================== 公共数据准备 =====================
models_by_dorm, outcome_logs_df, weekly_metrics_by_dorm = prepare_global_data(
    df_all, ridge_alpha=float(ridge_alpha)
)
st.session_state["weekly_metrics_by_dorm"] = weekly_metrics_by_dorm
st.session_state["dorm_outcome_map"] = (
    {dorm_id: sub.copy() for dorm_id, sub in outcome_logs_df.groupby("dorm_id")}
    if outcome_logs_df is not None
    and not outcome_logs_df.empty
    and "dorm_id" in outcome_logs_df.columns
    else {}
)




# ===================== 手机壳公共样式（仅用户端注入） =====================
def _inject_mobile_shell_style():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;700;900&display=swap');

        :root {
            --mobile-shell-max: 420px;
            --mobile-shell-bg: #F1F5F2;
            --mobile-page-bg: #CBD8D0;
            --mobile-radius: 36px;
            --mobile-gap: 12px;
            --mobile-stage-top: 36px;
            --mobile-stage-bottom: 28px;
            --mobile-stage-reserve: 84px;
            --mobile-shell-h: min(844px, calc(100dvh - var(--mobile-stage-reserve)));
            --mobile-edge-inset: 28px;
            --mobile-status-h: 30px;
            --mobile-tab-h: 44px;
            --mobile-tab-shell-h: 64px;
            --mobile-safe-bottom: env(safe-area-inset-bottom, 0px);
        }

        #MainMenu,
        header[data-testid="stHeader"] {
            visibility: hidden !important;
            height: 0 !important;
        }

        html, body, .stApp {
            min-height: 100dvh !important;
            background: var(--mobile-page-bg) !important;
            font-family: 'Noto Sans SC', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        [data-testid="stAppViewContainer"],
        .stAppViewContainer,
        [data-testid="stMain"],
        .stMain {
            min-height: 100dvh !important;
            background: var(--mobile-page-bg) !important;
            overflow: visible !important;
        }

        [data-testid="stMainBlockContainer"]:not(.block-container),
        .stMainBlockContainer:not(.block-container) {
            box-sizing: border-box !important;
            min-height: 100dvh !important;
            height: auto !important;
            padding: var(--mobile-stage-top) 0 var(--mobile-stage-bottom) !important;
            margin: 0 !important;
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
            justify-content: flex-start !important;
            overflow: visible !important;
        }

        .block-container {
            box-sizing: border-box !important;
            width: min(100%, var(--mobile-shell-max)) !important;
            max-width: var(--mobile-shell-max) !important;
            height: var(--mobile-shell-h) !important;
            min-height: 0 !important;
            max-height: var(--mobile-shell-h) !important;
            position: relative !important;
            margin: var(--mobile-stage-top) auto var(--mobile-stage-bottom) !important;
            padding: 0 !important;
            background: var(--mobile-shell-bg) !important;
            border-radius: var(--mobile-radius) !important;
            box-shadow:
                0 0 0 8px #1A1C22,
                0 0 0 10px #3A3C44,
                0 26px 72px rgba(0,0,0,0.36) !important;
            overflow: hidden !important;
        }

        [data-testid="stMainBlockContainer"]:not(.block-container) .block-container,
        .stMainBlockContainer:not(.block-container) .block-container {
            margin: 0 auto !important;
        }

        .block-container .stMainBlockContainer,
        .block-container [data-testid="stMainBlockContainer"] {
            min-height: 0 !important;
            height: 100% !important;
            position: relative !important;
            overflow: hidden !important;
            padding: 0 !important;
            margin: 0 !important;
        }

        .block-container > [data-testid="stVerticalBlock"],
        .block-container > div > [data-testid="stVerticalBlock"] {
            height: 100% !important;
            min-height: 0 !important;
            position: relative !important;
            gap: 0 !important;
            overflow: hidden !important;
        }

        [data-testid="stVerticalBlock"] { min-width: 0 !important; }
        [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
            gap: 8px !important;
            min-width: 0 !important;
        }
        iframe {
            max-width: 100% !important;
            border-radius: 16px !important;
        }
        [data-testid="stButton"] > button {
            min-height: 40px !important;
            border-radius: 14px !important;
            font-size: 13px !important;
            font-weight: 800 !important;
            padding: 8px 10px !important;
            white-space: normal !important;
        }

        .st-key-mobile_status_shell,
        .mobile-status-wrap {
            position: absolute !important;
            top: 0 !important;
            left: var(--mobile-edge-inset) !important;
            right: var(--mobile-edge-inset) !important;
            height: var(--mobile-status-h) !important;
            z-index: 20 !important;
            margin: 0 !important;
            padding: 0 !important;
            line-height: 0 !important;
            overflow: hidden !important;
        }
        .st-key-mobile_status_shell [data-testid="stElementContainer"],
        .st-key-mobile_status_shell [data-testid="stIFrame"],
        .mobile-status-wrap [data-testid="stElementContainer"],
        .mobile-status-wrap [data-testid="stIFrame"] {
            margin: 0 !important;
            padding: 0 !important;
            line-height: 0 !important;
        }
        .st-key-mobile_status_shell iframe,
        .mobile-status-wrap iframe {
            display: block !important;
            border-radius: 0 !important;
        }

        .st-key-mobile_tabbar_shell,
        .mobile-tabbar {
            position: absolute !important;
            left: var(--mobile-edge-inset) !important;
            right: var(--mobile-edge-inset) !important;
            bottom: calc(8px + var(--mobile-safe-bottom)) !important;
            box-sizing: border-box !important;
            width: auto !important;
            max-width: none !important;
            z-index: 30;
            margin: 0 auto !important;
            padding: 6px 7px calc(6px + var(--mobile-safe-bottom)) !important;
            background: rgba(255,255,255,0.94) !important;
            border: 1px solid rgba(220,232,223,0.86) !important;
            border-radius: 28px !important;
            box-shadow: 0 12px 30px rgba(16,24,18,0.13) !important;
            backdrop-filter: blur(14px);
        }
        .st-key-mobile_tabbar_shell [data-testid="stVerticalBlock"],
        .st-key-mobile_tabbar_shell [data-testid="stElementContainer"] {
            gap: 0 !important;
            margin: 0 !important;
            padding: 0 !important;
        }
        .st-key-mobile_tabbar_shell [data-testid="stHorizontalBlock"],
        .mobile-tabbar [data-testid="stHorizontalBlock"] {
            flex-wrap: nowrap !important;
            gap: 2px !important;
            margin: 0 !important;
            padding: 0 !important;
        }
        .st-key-mobile_tabbar_shell [data-testid="column"] {
            flex: 1 1 0 !important;
            min-width: 0 !important;
            padding: 0 !important;
        }
        .st-key-mobile_tabbar_shell [data-testid="stButton"],
        .mobile-tabbar [data-testid="stButton"] {
            margin: 0 !important;
            padding: 0 !important;
        }
        .st-key-mobile_tabbar_shell [data-testid="stButton"] > button,
        .mobile-tabbar [data-testid="stButton"] > button {
            min-height: var(--mobile-tab-h) !important;
            height: var(--mobile-tab-h) !important;
            padding: 4px 2px !important;
            line-height: 1.12 !important;
            border: 0 !important;
            border-radius: 18px !important;
            background: transparent !important;
            box-shadow: none !important;
            color: #7B8D83 !important;
            font-size: 12px !important;
            font-weight: 900 !important;
            white-space: pre-line !important;
        }
        .st-key-mobile_tabbar_shell [data-testid="stButton"] > button[kind="primary"],
        .mobile-tabbar [data-testid="stButton"] > button[kind="primary"] {
            background: #EAF7EF !important;
            color: #0D6B38 !important;
            box-shadow: inset 0 0 0 1px rgba(74,222,128,0.20) !important;
        }
        .st-key-mobile_tabbar_shell [data-testid="stButton"] > button:hover,
        .mobile-tabbar [data-testid="stButton"] > button:hover {
            background: #F2F7F4 !important;
            color: #1C6E3D !important;
        }

        .st-key-mobile_page_scroll {
            position: absolute !important;
            top: calc(var(--mobile-status-h) + 6px) !important;
            bottom: calc(var(--mobile-tab-shell-h) + 18px + var(--mobile-safe-bottom)) !important;
            left: var(--mobile-edge-inset) !important;
            right: var(--mobile-edge-inset) !important;
            width: auto !important;
            max-width: none !important;
            min-height: 0 !important;
            overflow-y: auto !important;
            overflow-x: hidden !important;
            padding: 0 0 16px !important;
            margin: 0 !important;
            scrollbar-width: none !important;
            box-sizing: border-box !important;
        }
        .st-key-mobile_page_scroll::-webkit-scrollbar {
            display: none !important;
        }
        .st-key-mobile_page_scroll [data-testid="stElementContainer"] {
            max-width: 100% !important;
        }
        .st-key-mobile_page_scroll iframe {
            max-width: 100% !important;
        }

        @media (max-width: 520px) {
            :root {
                --mobile-shell-max: 100vw;
                --mobile-shell-h: 100dvh;
                --mobile-radius: 0px;
                --mobile-stage-top: 0px;
                --mobile-stage-bottom: 0px;
                --mobile-stage-reserve: 0px;
                --mobile-edge-inset: 14px;
                --mobile-tab-shell-h: 72px;
            }
            html, body, .stApp,
            [data-testid="stAppViewContainer"],
            .stAppViewContainer,
            [data-testid="stMain"],
            .stMain {
                width: 100vw !important;
                max-width: 100vw !important;
                min-height: 100dvh !important;
                overflow-x: hidden !important;
                background: var(--mobile-shell-bg) !important;
            }
            .block-container {
                box-sizing: border-box !important;
                width: 100vw !important;
                max-width: 100vw !important;
                height: 100dvh !important;
                min-height: 0 !important;
                max-height: 100dvh !important;
                margin: 0 !important;
                padding-left: 0 !important;
                padding-right: 0 !important;
                border-radius: 0 !important;
                box-shadow: none !important;
                overflow: hidden !important;
            }
            [data-testid="stMainBlockContainer"]:not(.block-container),
            .stMainBlockContainer:not(.block-container) {
                min-height: 100dvh !important;
                height: 100dvh !important;
                padding: 0 !important;
                overflow: hidden !important;
            }
            [data-testid="stMainBlockContainer"]:not(.block-container) .block-container,
            .stMainBlockContainer:not(.block-container) .block-container {
                height: 100dvh !important;
                max-height: 100dvh !important;
                margin: 0 !important;
            }
            .st-key-mobile_status_shell,
            .mobile-status-wrap {
                top: env(safe-area-inset-top, 0px) !important;
                left: var(--mobile-edge-inset) !important;
                right: var(--mobile-edge-inset) !important;
            }
            .st-key-mobile_page_scroll {
                top: calc(env(safe-area-inset-top, 0px) + var(--mobile-status-h) + 8px) !important;
                bottom: calc(var(--mobile-tab-shell-h) + 10px + var(--mobile-safe-bottom)) !important;
                left: var(--mobile-edge-inset) !important;
                right: var(--mobile-edge-inset) !important;
                padding-bottom: 18px !important;
                overflow-x: hidden !important;
            }
            .st-key-mobile_tabbar_shell,
            .mobile-tabbar {
                left: 12px !important;
                right: 12px !important;
                bottom: calc(6px + var(--mobile-safe-bottom)) !important;
                border-radius: 24px !important;
                box-shadow: 0 8px 24px rgba(16,24,18,0.16) !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_mobile_tabbar(active_tab: str, unread_msg_count: int = 0):
    tabs_def = [
        ("Home", "首页", "🏠"),
        ("Tasks", "任务", "📋"),
        ("Messages", "消息", "💬"),
        ("Profile", "我的", "👤"),
    ]

    with st.container(key="mobile_tabbar_shell"):
        cols = st.columns(4, gap="small")
        for col, (tab_id, tab_name, icon) in zip(cols, tabs_def):
            active = active_tab == tab_id
            unread = f" {unread_msg_count}" if tab_id == "Messages" and unread_msg_count > 0 else ""
            label = f"{icon}\n{tab_name}{unread}"
            with col:
                if st.button(
                    label,
                    key=f"mobile_tab_btn_{tab_id}",
                    use_container_width=True,
                    type="primary" if active else "secondary",
                ):
                    st.session_state["mobile_tab"] = tab_id
                    st.rerun()


def _inject_mobile_final_overrides():
    """Re-assert the mobile shell after page-level CSS has been injected."""
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"],
        .stAppViewContainer,
        [data-testid="stMain"],
        .stMain {
            min-height: 100dvh !important;
            background: var(--mobile-page-bg) !important;
            overflow: visible !important;
        }
        [data-testid="stMainBlockContainer"]:not(.block-container),
        .stMainBlockContainer:not(.block-container) {
            box-sizing: border-box !important;
            min-height: 100dvh !important;
            height: auto !important;
            padding: var(--mobile-stage-top) 0 var(--mobile-stage-bottom) !important;
            margin: 0 !important;
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
            justify-content: flex-start !important;
            overflow: visible !important;
        }
        .block-container {
            box-sizing: border-box !important;
            width: min(100%, var(--mobile-shell-max)) !important;
            max-width: var(--mobile-shell-max) !important;
            height: var(--mobile-shell-h) !important;
            min-height: 0 !important;
            max-height: var(--mobile-shell-h) !important;
            position: relative !important;
            margin: var(--mobile-stage-top) auto var(--mobile-stage-bottom) !important;
            padding: 0 !important;
            background: var(--mobile-shell-bg) !important;
            border-radius: var(--mobile-radius) !important;
            box-shadow:
                0 0 0 8px #1A1C22,
                0 0 0 10px #3A3C44,
                0 26px 72px rgba(0,0,0,0.36) !important;
            overflow: hidden !important;
        }
        [data-testid="stMainBlockContainer"]:not(.block-container) .block-container,
        .stMainBlockContainer:not(.block-container) .block-container {
            margin: 0 auto !important;
        }
        .block-container .stMainBlockContainer,
        .block-container [data-testid="stMainBlockContainer"] {
            height: 100% !important;
            min-height: 0 !important;
            position: relative !important;
            overflow: hidden !important;
            padding: 0 !important;
            margin: 0 !important;
        }
        @media (max-width: 520px) {
            :root {
                --mobile-shell-max: 100vw;
                --mobile-shell-h: 100dvh;
                --mobile-radius: 0px;
                --mobile-stage-top: 0px;
                --mobile-stage-bottom: 0px;
                --mobile-stage-reserve: 0px;
                --mobile-edge-inset: 14px;
                --mobile-tab-shell-h: 72px;
            }
            html, body, .stApp,
            [data-testid="stAppViewContainer"],
            .stAppViewContainer,
            [data-testid="stMain"],
            .stMain {
                width: 100vw !important;
                max-width: 100vw !important;
                min-height: 100dvh !important;
                overflow-x: hidden !important;
                background: var(--mobile-shell-bg) !important;
            }
            .block-container {
                box-sizing: border-box !important;
                width: 100vw !important;
                max-width: 100vw !important;
                height: 100dvh !important;
                min-height: 0 !important;
                max-height: 100dvh !important;
                margin: 0 !important;
                padding-left: 0 !important;
                padding-right: 0 !important;
                border-radius: 0 !important;
                box-shadow: none !important;
                overflow: hidden !important;
            }
            [data-testid="stMainBlockContainer"]:not(.block-container),
            .stMainBlockContainer:not(.block-container) {
                min-height: 100dvh !important;
                height: 100dvh !important;
                padding: 0 !important;
                overflow: hidden !important;
            }
            [data-testid="stMainBlockContainer"]:not(.block-container) .block-container,
            .stMainBlockContainer:not(.block-container) .block-container {
                height: 100dvh !important;
                max-height: 100dvh !important;
                margin: 0 !important;
            }
            .st-key-mobile_status_shell,
            .mobile-status-wrap {
                top: env(safe-area-inset-top, 0px) !important;
                left: var(--mobile-edge-inset) !important;
                right: var(--mobile-edge-inset) !important;
            }
            .st-key-mobile_page_scroll {
                top: calc(env(safe-area-inset-top, 0px) + var(--mobile-status-h) + 8px) !important;
                bottom: calc(var(--mobile-tab-shell-h) + 10px + var(--mobile-safe-bottom)) !important;
                left: var(--mobile-edge-inset) !important;
                right: var(--mobile-edge-inset) !important;
                padding-bottom: 18px !important;
                overflow-x: hidden !important;
            }
            .st-key-mobile_tabbar_shell,
            .mobile-tabbar {
                left: 12px !important;
                right: 12px !important;
                bottom: calc(6px + var(--mobile-safe-bottom)) !important;
                border-radius: 24px !important;
                box-shadow: 0 8px 24px rgba(16,24,18,0.16) !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# ===================== 页面路由 =====================
if page == "Admin":
    # Admin 保持完全原样，不套手机壳
    st.markdown(
        """
        <style>
        .stApp { background: #F2F4F3 !important; }
        .block-container { max-width: 1120px !important; padding: 1.6rem 2rem 3.5rem !important; margin: 0 auto !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    render_admin_page(
        hourly_df=df_all,
        outcome_logs_df=outcome_logs_df,
        baseline_model=None,
    )

else:
    _inject_mobile_shell_style()

    if st.query_params.get("splash") == "done":
        st.session_state["_splash_shown"] = True

    if not st.session_state.get("_splash_shown", False):
        import time as _time
        import streamlit.components.v1 as _splash_comp

        splash_deadline = st.session_state.setdefault("_splash_deadline", _time.time() + 2.6)

        with st.container(key="splash_shell"):
            st.markdown("""
<style>
*{box-sizing:border-box;margin:0;padding:0;}
.st-key-splash_shell{
  position:absolute !important;
  inset:0 !important;
  width:100% !important;
  height:100% !important;
  z-index:40 !important;
  overflow:hidden !important;
}
.st-key-splash_shell [data-testid="stElementContainer"],
.st-key-splash_shell [data-testid="stMarkdownContainer"]{
  width:100% !important;
  height:100% !important;
  margin:0 !important;
  padding:0 !important;
}
.splash{width:100%;height:100%;
  background:linear-gradient(155deg,#052E16 0%,#1A6E3D 55%,#2D9E5A 100%);
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  border-radius:var(--mobile-radius);padding:72px 18px 92px;position:relative;}
.logo-box{width:88px;height:88px;border-radius:28px;
  background:rgba(255,255,255,0.12);border:1px solid rgba(255,255,255,0.2);
  display:flex;align-items:center;justify-content:center;margin-bottom:18px;}
.app-name{font-size:28px;font-weight:900;color:#fff;letter-spacing:-0.5px;margin-bottom:8px;}
.app-sub{font-size:13px;color:rgba(255,255,255,0.5);letter-spacing:2px;margin-bottom:32px;}
.bar-bg{width:80px;height:4px;background:rgba(255,255,255,0.18);border-radius:99px;overflow:hidden;}
.bar-fg{width:70%;height:100%;background:rgba(255,255,255,0.75);border-radius:99px;}
.ver{font-size:11px;color:rgba(255,255,255,0.3);margin-top:48px;font-family:monospace;}
.splash-hint{position:absolute;left:0;right:0;bottom:38px;text-align:center;
  font-size:12px;font-weight:800;color:rgba(255,255,255,0.72);letter-spacing:1px;}
</style>
<div class="splash">
  <div class="logo-box">
    <svg width="44" height="44" viewBox="0 0 44 44" fill="none">
      <circle cx="22" cy="22" r="18" stroke="rgba(255,255,255,0.2)" stroke-width="2"/>
      <path d="M22 8 L16 22 H22 L19 36 L32 18 H25 Z" fill="#4ADE80"/>
      <circle cx="22" cy="22" r="3.5" fill="white" opacity="0.9"/>
    </svg>
  </div>
  <div class="app-name">智控绿舍</div>
  <div class="app-sub">SMART DORM · ECO SYSTEM</div>
  <div class="bar-bg"><div class="bar-fg"></div></div>
  <div class="ver">v4.0 · LinUCB 完整版</div>
  <div class="splash-hint">正在进入应用</div>
</div>
""", unsafe_allow_html=True)

            # Frontend navigation is only a helper; the server-side timer below is authoritative.
            _splash_comp.html(
                """
                <script>
                setTimeout(function(){
                  try {
                    const url = new URL(window.parent.location.href);
                    url.searchParams.set("splash", "done");
                    window.parent.location.href = url.toString();
                  } catch (e) {}
                }, 2700);
                </script>
                """,
                height=1,
                scrolling=False,
            )

        remaining = float(splash_deadline) - _time.time()
        if remaining > 0:
            _time.sleep(remaining)
        st.query_params["splash"] = "done"
        st.session_state["_splash_shown"] = True
        st.rerun()

    else:
        import streamlit.components.v1 as _comp
        mobile_tab   = st.session_state.get("mobile_tab", "Home")
        from datetime import datetime as _dt
        now_time     = _dt.now().strftime("%H:%M")
        unread_count = len([m for m in st.session_state.get("messages", [])
                            if isinstance(m, dict) and not m.get("read", False)])

        # 顶部状态栏。它现在只是手机容器里的普通内容，不再依赖固定壳高。
        with st.container(key="mobile_status_shell"):
            _comp.html(f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
*{{box-sizing:border-box;margin:0;padding:0;}}
html,body{{width:100%;height:30px;overflow:hidden;background:#F1F5F2;
  font-family:'Noto Sans SC',-apple-system,sans-serif;}}
.bar{{height:30px;display:flex;justify-content:space-between;align-items:center;
  padding:1px 4px 0;font-size:14px;font-weight:800;color:#1A2820;}}
</style></head><body>
<div class="bar">
  <span>{now_time}</span>
  <div style="display:flex;gap:6px;align-items:center;">
    <svg width="17" height="12" viewBox="0 0 17 12" fill="#1A2820">
      <rect x="0" y="6" width="3" height="6" rx="0.8"/>
      <rect x="4.5" y="4" width="3" height="8" rx="0.8"/>
      <rect x="9" y="2" width="3" height="10" rx="0.8"/>
      <rect x="13.5" y="0" width="3" height="12" rx="0.8" opacity="0.3"/>
    </svg>
    <svg width="25" height="12" viewBox="0 0 25 12" fill="#1A2820">
      <rect x="0.5" y="0.5" width="21" height="11" rx="3"
            stroke="#1A2820" stroke-width="1" fill="none"/>
      <rect x="22" y="4" width="3" height="4" rx="1.5"/>
      <rect x="2" y="2" width="15" height="8" rx="2"/>
    </svg>
  </div>
</div>
</body></html>""", height=30, scrolling=False)

        # 内容路由。页面内容在固定手机屏幕内部滚动，底部 Tab 留在屏幕内。
        with st.container(key="mobile_page_scroll"):
            if mobile_tab == "Home":
                if df_all is not None and not df_all.empty:
                    render_home_page(
                        hourly_df=df_all, outcome_logs_df=outcome_logs_df,
                        baseline_model=None, load_stats=load_stats,
                        daily_cap=int(daily_cap), alpha_ucb=float(linucb_alpha),
                        log_interaction_func=log_interaction,
                        freq_guard_allow_daily_func=freq_guard_allow_daily,
                        already_decided_today_func=already_decided_today,
                        mobile=True,
                    )
                else:
                    st.info("请先在左侧导入 CSV 或生成模拟数据。")
            elif mobile_tab == "Tasks":
                render_tasks_page()
            elif mobile_tab == "Messages":
                render_mobile_messages_page()
            elif mobile_tab == "Profile":
                render_profile_page()

        _inject_mobile_final_overrides()
        _render_mobile_tabbar(mobile_tab, unread_count)
