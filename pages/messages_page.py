import pandas as pd
import streamlit as st
import json as _json
import streamlit.components.v1 as components
from strategy_library import build_message_from_arm, get_cluster_message_plan
from textwrap import dedent


def _render_html(html: str):
    st.markdown(dedent(html).strip(), unsafe_allow_html=True)


def inject_messages_style():
    st.markdown(
        dedent("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter+Tight:wght@300;400;500;600;700;800;900&display=swap');

        /* ══ 全局 ══ */
        .block-container { 
            padding-top: 1.0rem !important; 
            padding-bottom: 2rem !important; 
            max-width: 100% !important; 
            margin: 0 auto; 
        }
        .stApp { background: #F0F2F1; }

        /* ══ 页두 ══ */
        .msg-page-hero {
            background: linear-gradient(135deg, #FFFFFF 0%, #F7FBF8 100%);
            border: 1px solid #E2E8E4;
            border-radius: 20px;
            padding: 18px 20px 16px;
            margin-bottom: 18px;
            box-shadow: 0 4px 14px rgba(0,0,0,.05);
        }
       .msg-eyebrow {
            font-size: 11px;
            font-weight: 800;
            color: #16A34A;
            letter-spacing: 1.6px;
            text-transform: uppercase;
            display: flex;
            align-items: center;
            gap: 6px;
            margin-bottom: 8px;
        }
        .msg-eyebrow-dot { width: 7px; height: 7px; border-radius: 50%; background: #22C55E; animation: msg-blink 2s infinite; }
        @keyframes msg-blink { 0%,100%{opacity:1;} 50%{opacity:.2;} }
        .msg-page-title {
            font-family: 'Inter Tight', sans-serif;
            font-size: 42px;
            font-weight: 900;
            color: #061408;
            letter-spacing: -1.4px;
            line-height: 1.02;
            margin-bottom: 8px;
        }
        .msg-page-sub {
            font-size: 15px;
            color: #4B5563;
            font-weight: 500;
            line-height: 1.6;
        }
        .msg-page-guide {
            margin-top: 10px;
            font-size: 14px;
            color: #6B7280;
            line-height: 1.7;
        }

        .msg-page-kpi {
            display: inline-flex;
            align-items: center;
            padding: 5px 12px;
            border-radius: 999px;
            background: #F0FDF4;
            border: 1px solid #BBF7D0;
            color: #15803D;
            font-size: 12px;
            font-weight: 700;
        }
        /* ══ 筛选栏 ══ */
        .msg-filter-bar { background: #FFFFFF; border: 1px solid #D1D9D5; border-radius: 14px; padding: 12px 14px 6px; margin-bottom: 22px; box-shadow: 0 1px 4px rgba(0,0,0,.04); }

        /* ══ 分组标题 ══ */
        .msg-group-hd { display: flex; align-items: center; gap: 10px; margin: 24px 0 10px; }
        .msg-group-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
        .msg-group-label { font-family: 'Inter Tight', sans-serif; font-size: 12px; font-weight: 800; letter-spacing: .8px; text-transform: uppercase; }
        .msg-group-line { flex: 1; height: 1px; background: #E2E8E4; }
        .msg-group-pill { font-size: 10px; font-weight: 600; padding: 2px 10px; border-radius: 20px; white-space: nowrap; }

        /* ══ 滚动分组容器 ══ */
        .msg-scroll-zone {
            max-height: 480px;
            overflow-y: auto;
            padding-right: 4px;
            scrollbar-width: thin;
            scrollbar-color: #D1D9D5 transparent;
        }
        .msg-scroll-zone::-webkit-scrollbar { width: 4px; }
        .msg-scroll-zone::-webkit-scrollbar-track { background: transparent; }
        .msg-scroll-zone::-webkit-scrollbar-thumb { background: #C8D0CB; border-radius: 4px; }

        /* ══ Push 消息卡片 (R 类) ══ */
        .push-card {
            background: #FFFFFF;
            border-radius: 16px;
            padding: 16px 18px 14px;
            margin-bottom: 10px;
            position: relative;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,.06), 0 0 0 1px rgba(0,0,0,.06);
            transition: box-shadow .15s;
        }
        .push-card.unread {
            background: #F7FFF9;
            box-shadow: 0 4px 16px rgba(22,163,74,.10), 0 0 0 1.5px #86EFAC;
        }
        .push-card.done {
            background: #FAFAFA;
            box-shadow: 0 1px 4px rgba(0,0,0,.04), 0 0 0 1px #E5E7EB;
            opacity: .75;
        }
        .push-card-bar {
            position: absolute;
            left: 0; top: 0; bottom: 0;
            width: 4px;
            border-radius: 4px 0 0 4px;
        }
        .push-card-meta { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; margin-bottom: 9px; padding-left: 6px; }
        .push-badge { font-size: 10px; font-weight: 700; padding: 2px 9px; border-radius: 999px; letter-spacing: .2px; }
        .push-chip  { font-size: 10px; color: #9CA3AF; background: #F3F4F6; border-radius: 5px; padding: 1px 7px; }
        .push-status-dot { width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; }
        .push-title { font-family: 'Inter Tight', sans-serif; font-size: 15px; font-weight: 700; color: #0A1A0F; line-height: 1.35; margin-bottom: 6px; padding-left: 6px; }
        .push-title.done { color: #9CA3AF; font-weight: 500; text-decoration: line-through; }
        .push-body  { font-size: 13px; color: #4B5563; line-height: 1.75; margin-bottom: 10px; padding-left: 6px; }
        .push-body.done { color: #B0BAB5; }
        .push-foot  { display: flex; align-items: center; justify-content: space-between; gap: 8px; padding: 9px 6px 0; border-top: 1px solid #F0F4F1; }
        .push-foot-meta { font-size: 10px; color: #B0BAB5; font-family: monospace; }
        .push-cta {
            font-size: 11px; font-weight: 700;
            padding: 5px 16px; border-radius: 999px;
            border: none; cursor: pointer;
            background: #16A34A; color: #fff;
            transition: background .15s, transform .1s;
            white-space: nowrap;
        }
        .push-cta:hover { background: #15803D; transform: scale(1.02); }
        .push-cta.done-btn { background: #F3F4F6; color: #9CA3AF; cursor: default; }

        /* ══ 贴士轮播 (P 类 iframe) ══ */
        .tip-carousel-wrap { border-radius: 16px; overflow: hidden; margin-bottom: 8px; }

        /* ══ 右侧面板 ══ */
        .rp-section-label { font-size: 10px; font-weight: 700; color: #9CA3AF; letter-spacing: .8px; text-transform: uppercase; margin-bottom: 8px; }
        .focus-card-native { background: linear-gradient(135deg, #052E12 0%, #166534 100%); border-radius: 16px; padding: 18px 16px 16px; margin-bottom: 6px; }
        .focus-card-pre   { font-size: 9px; font-weight: 700; color: rgba(255,255,255,.45); letter-spacing: 1.2px; text-transform: uppercase; margin-bottom: 8px; }
        .focus-card-title { font-family: 'Inter Tight', sans-serif; font-size: 15px; font-weight: 800; color: #fff; line-height: 1.3; margin-bottom: 8px; }
        .focus-card-body  { font-size: 12px; color: rgba(255,255,255,.70); line-height: 1.65; margin-bottom: 10px; }
        .focus-card-chip  { display: inline-block; font-size: 10px; font-weight: 600; background: rgba(255,255,255,.14); color: rgba(255,255,255,.80); border-radius: 6px; padding: 3px 10px; }

        /* ══ 旧类兼容 ══ */
        .msg-section-title { display: none; }
        .msg-push-wrap { margin-bottom: 0; }
        .msg-tip-wrap  { margin-bottom: 0; }
        .notif-card, .notif-card.unread { border-radius: 16px; }
        /* ══ v3 间距压缩覆盖 ══ */
        .block-container { padding-top: 1.0rem !important; padding-bottom: 2rem !important; }
        .msg-group-hd { margin: 14px 0 8px !important; }
        .msg-scroll-zone { margin-bottom: 0 !important; }
        /* 压缩 st.metric 间距 */
        [data-testid="stMetric"] { padding: 10px 14px 8px !important; }
        [data-testid="stMetricValue"] { font-size: 24px !important; }
        /* 压缩 iframe 外层 margin */
        iframe { margin-bottom: 0 !important; }
        /* 压缩 st.divider */
        hr { margin: 10px 0 !important; }
        /* 压缩 st.columns 间隙 */
        [data-testid="stHorizontalBlock"] { gap: 6px !important; }

        /* bordered container 强制白底，覆盖到更深层，避免底色发灰 */
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: #FFFFFF !important;
            border: 1px solid #E6EDE8 !important;
            border-radius: 16px !important;
            box-shadow: 0 1px 4px rgba(0,0,0,.04) !important;
        }

        [data-testid="stVerticalBlockBorderWrapper"] > div,
        [data-testid="stVerticalBlockBorderWrapper"] > div > div,
        [data-testid="stVerticalBlockBorderWrapper"] > div > div > div,
        [data-testid="stVerticalBlockBorderWrapper"] > div > div > div > div {
            background: #FFFFFF !important;
            border-radius: 16px !important;
        }

        /* push 卡片内部：仅负责内容排版，不再包裹按钮逻辑 */
        
        .push-shell-top {
            display:flex; align-items:center; justify-content:space-between;
            gap:10px; margin-bottom:10px;
        }
        .push-shell-left {
            display:flex; align-items:center; gap:8px; flex-wrap:wrap;
        }
        .push-shell-badge {
            font-size:10px; font-weight:700; padding:3px 10px; border-radius:999px;
            letter-spacing:.2px;
        }
        .push-shell-chip {
            font-size:10px; color:#9CA3AF; background:#F3F4F6; border-radius:999px; padding:3px 10px;
        }
        .push-shell-status {
            font-size:11px; font-weight:700; white-space:nowrap;
        }
        .push-shell-title {
            font-family:'Inter Tight', sans-serif; font-size:18px; font-weight:800;
            color:#061408; line-height:1.35; letter-spacing:-.25px; margin-bottom:8px;
        }
        .push-shell-title.read {
            color:#9CA3AF; text-decoration:line-through;
        }
        .push-shell-body {
            font-size:14px; color:#374151; line-height:1.82; margin-bottom:12px;
        }
        .push-shell-body.read { color:#B0BAB5; }
        .push-shell-foot {
            padding-top:10px; border-top:1px solid #EEF2EE; font-size:11px; color:#9CA3AF;
        }
        /* 右上角状态按钮：弱化成状态胶囊 */
        [data-testid="stButton"] > button[kind="secondary"] {
            border-radius: 999px !important;
        }

        /* 当前页 push 状态按钮弱化 */
        .stButton > button {
            font-size: 12px !important;
            font-weight: 600 !important;
            min-height: 2rem !important;
            padding: 0.2rem 0.7rem !important;
            border-radius: 999px !important;
        }
        /* push 状态按钮缩小，降低存在感 */
        button[kind="secondary"],
        button[kind="secondaryFormSubmit"] {
            min-height: 1.75rem !important;
            height: 1.75rem !important;
            padding: 0.05rem 0.55rem !important;
            font-size: 11px !important;
            font-weight: 600 !important;
            line-height: 1 !important;
            border-radius: 999px !important;
        }

        /* disabled 状态也保持小尺寸 */
        button:disabled {
            min-height: 1.75rem !important;
            height: 1.75rem !important;
            padding: 0.05rem 0.55rem !important;
            font-size: 11px !important;
            border-radius: 999px !important;
        }     
        </style>
        """).strip(),
        unsafe_allow_html=True,
    )


# ===================== 函数定义 =====================

def infer_strategy_type(arm_id: str) -> str:
    if not arm_id:
        return "Unknown"

    arm = str(arm_id).strip().lower()

    if arm.startswith("p"):
        return "P"
    if arm.startswith("r"):
        return "R"
    if arm.startswith("f"):
        return "F"
    if arm.startswith("s"):
        return "S"

    if "prompt" in arm:
        return "P"
    if "remind" in arm or "reminder" in arm:
        return "R"
    if "feedback" in arm:
        return "F"
    if "social" in arm or "compare" in arm or "comparison" in arm:
        return "S"

    return "Unknown"



def infer_channel(msg: dict) -> str:
    channel = str(msg.get("channel", "")).strip().lower()
    if channel in ["tip", "push"]:
        return channel

    arm_id = str(msg.get("arm_id", "")).strip().lower()
    strategy_type = infer_strategy_type(arm_id)

    if strategy_type == "P":
        return "tip"
    if strategy_type == "R":
        return "push"

    if "tip" in arm_id:
        return "tip"
    if "push" in arm_id:
        return "push"

    return "other"



def strategy_label_text(strategy_type: str) -> str:
    mapping = {
        "P": "信息提示",
        "R": "定期提醒",
        "F": "数据反馈",
        "S": "社会比较",
        "Unknown": "未分类",
    }
    return mapping.get(strategy_type, "未分类")



def channel_label_text(channel: str) -> str:
    mapping = {
        "tip": "小贴士",
        "push": "消息推送",
        "other": "其他消息",
    }
    return mapping.get(channel, "其他消息")

def get_strategy_theme(strategy_type: str) -> dict:
    stype = str(strategy_type or "Unknown").upper().strip()

    themes = {
        "P": {
            "code": "P",
            "name": "信息提示",
            "color": "#2563EB",
            "bg": "rgba(219,234,254,0.92)",
            "border": "rgba(147,197,253,0.95)",
            "accent": "linear-gradient(90deg, #60A5FA 0%, #2563EB 100%)",
        },
        "R": {
            "code": "R",
            "name": "定期提醒",
            "color": "#16A34A",
            "bg": "rgba(220,252,231,0.92)",
            "border": "rgba(134,239,172,0.95)",
            "accent": "linear-gradient(90deg, #4ADE80 0%, #16A34A 100%)",
        },
        "F": {
            "code": "F",
            "name": "数据反馈",
            "color": "#7C3AED",
            "bg": "rgba(237,233,254,0.94)",
            "border": "rgba(196,181,253,0.98)",
            "accent": "linear-gradient(90deg, #A78BFA 0%, #7C3AED 100%)",
        },
        "S": {
            "code": "S",
            "name": "社会比较",
            "color": "#F97316",
            "bg": "rgba(255,237,213,0.94)",
            "border": "rgba(253,186,116,0.98)",
            "accent": "linear-gradient(90deg, #FB923C 0%, #F97316 100%)",
        },
        "Unknown": {
            "code": "?",
            "name": "未分类",
            "color": "#64748B",
            "bg": "rgba(241,245,249,0.95)",
            "border": "rgba(203,213,225,0.95)",
            "accent": "linear-gradient(90deg, #CBD5E1 0%, #94A3B8 100%)",
        },
    }

    return themes.get(stype, themes["Unknown"])
# =========================
# 日志写入辅助函数
# =========================
def append_interaction_log(event: str, message_index: int, msg: dict, extra: dict | None = None):
    logs = st.session_state.get("interaction_logs", [])

    row = {
        "ts": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "date": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "event": event,
        "page": "messages",
        "message_index": int(message_index),
        "title": msg.get("title", ""),
        "dorm_id": msg.get("dorm_id", ""),
        "arm_id": msg.get("arm_id", ""),
        "template_id": msg.get("template_id", msg.get("arm_id", "")),
        "cluster_type": msg.get("cluster_type", ""),
        "message_key": msg.get("message_key", ""),
        "source_page": msg.get("source_page", ""),
        "score_basis": msg.get("score_basis", ""),
        "strategy_type": infer_strategy_type(msg.get("arm_id", "")),
        "channel": infer_channel(msg),
    }

    if extra:
        row.update(extra)

    logs.append(row)
    st.session_state["interaction_logs"] = logs



def mark_message_read(real_idx: int):
    messages = st.session_state.get("messages", [])
    if 0 <= real_idx < len(messages):
        if not bool(messages[real_idx].get("read", False)):
            messages[real_idx]["read"] = True
            messages[real_idx]["read_ts"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state["messages"] = messages



def build_message_df(messages: list[dict]) -> pd.DataFrame:
    rows = []
    for i, msg in enumerate(messages):
        arm_id = msg.get("arm_id", "")
        strategy_type = msg.get("strategy_type", infer_strategy_type(arm_id))
        channel = infer_channel(msg)

        rows.append(
            {
                "message_index": i,
                "title": msg.get("title", "Untitled Message"),
                "body": msg.get("body", ""),
                "ts": msg.get("ts", ""),
                "dorm_id": msg.get("dorm_id", ""),
                "arm_id": arm_id,
                "template_id": msg.get("template_id", arm_id),
                "cluster_type": msg.get("cluster_type", ""),
                "message_key": msg.get("message_key", ""),
                "source_page": msg.get("source_page", ""),
                "score_basis": msg.get("score_basis", ""),
                "sync_status": msg.get("sync_status", ""),
                "strategy_type": strategy_type,
                "strategy_name": strategy_label_text(strategy_type),
                "channel": channel,
                "channel_name": channel_label_text(channel),
                "goal": msg.get("goal", ""),
                "tone": msg.get("tone", ""),
                "cta_text": msg.get("cta_text", ""),
                "read": bool(msg.get("read", False)),
                "read_ts": msg.get("read_ts", ""),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty and "ts" in df.columns:
        df = df.sort_values("ts", ascending=False, kind="stable").reset_index(drop=True)
    return df



def pick_featured_message(msg_df: pd.DataFrame) -> pd.Series | None:
    if msg_df is None or msg_df.empty:
        return None

    push_unread = msg_df[(msg_df["channel"] == "push") & (msg_df["read"] == False)]
    if not push_unread.empty:
        return push_unread.iloc[0]

    tip_unread = msg_df[(msg_df["channel"] == "tip") & (msg_df["read"] == False)]
    if not tip_unread.empty:
        return tip_unread.iloc[0]

    return msg_df.iloc[0]



def make_featured_key_suffix(row: pd.Series) -> str:
    real_idx = int(row["message_index"])
    arm_id = str(row.get("arm_id", "na")).replace(" ", "_")
    ts = str(row.get("ts", "na")).replace(" ", "_").replace(":", "-")
    channel = str(row.get("channel", "na"))
    return f"{real_idx}_{channel}_{arm_id}_{ts}"



def _html_escape(text) -> str:
    if text is None:
        return ""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br>")
    )



def _message_meta_line(row: pd.Series, compact: bool = False) -> str:
    meta_parts = []
    ts = row.get("ts", "")
    dorm_id = row.get("dorm_id", "")
    cluster_type = row.get("cluster_type", "")
    strategy_name = row.get("strategy_name", "")
    arm_id = row.get("arm_id", "")
    source_page = row.get("source_page", "")

    if ts:
        meta_parts.append(f"时间：{ts}")
    if dorm_id:
        meta_parts.append(f"宿舍：{dorm_id}")
    if cluster_type:
        meta_parts.append(f"群体：{cluster_type}")
    if strategy_name:
        meta_parts.append(f"类型：{strategy_name}")
    if arm_id and not compact:
        meta_parts.append(f"模板：{arm_id}")
    if source_page and not compact:
        meta_parts.append(f"来源：{source_page}")

    return "｜".join(meta_parts)



def render_tip_card(row: pd.Series):
    """P 类：知识卡片式贴士。所有日志写入逻辑不变。"""
    real_idx = int(row["message_index"])
    msg = st.session_state["messages"][real_idx]
    arm_id = row["arm_id"]
    read_flag = bool(row["read"])

    # 日志（不变）
    if not msg.get("_view_logged_tip", False):
        append_interaction_log(
            event="message_view", message_index=real_idx, msg=msg,
            extra={"section": "tip"},
        )
        st.session_state["messages"][real_idx]["_view_logged_tip"] = True

    top_cls     = "read" if read_flag else ""
    label_cls   = "read" if read_flag else ""
    status_text = "已读" if read_flag else "未读"
    ts_short    = str(row.get("ts", ""))[:16]
    cluster     = str(row.get("cluster_type", ""))

    carousel_idx   = st.session_state.get("tip_card_index", 0)
    msg_unique_key = f"{real_idx}_{carousel_idx}_{arm_id}".replace(" ", "_").replace(":", "-")

    chip_html = ""
    if cluster: chip_html += f'<span class="tip-kcard-tag">{cluster} 群体</span>'
    if ts_short: chip_html += f'<span class="tip-kcard-time">{ts_short}</span>'

    _render_html(f"""
    <div class="tip-kcard">
        <div class="tip-kcard-top {top_cls}"></div>
        <div class="tip-kcard-inner">
            <div class="tip-kcard-header">
                <div class="tip-kcard-quote">"</div>
                <span class="tip-kcard-label {label_cls}">{status_text}</span>
            </div>
            <div class="tip-kcard-title">{_html_escape(row.get('title',''))}</div>
            <div class="tip-kcard-body">{_html_escape(row.get('body',''))}</div>
            <div class="tip-kcard-footer">{chip_html}</div>
        </div>
    </div>
    """)

    c1, c2 = st.columns([1, 4])
    with c1:
        if st.button("知道了", key=f"ack_tip_{msg_unique_key}", use_container_width=True):
            mark_message_read(real_idx)
            append_interaction_log(event="message_ack", message_index=real_idx, msg=msg,
                                   extra={"section": "tip"})
            st.rerun()
    with c2:
        with st.expander("查看详情", expanded=False):
            if arm_id:
                st.write(f"模板：`{arm_id}`")
            st.json(msg)


def render_push_card(row: pd.Series):
    real_idx = int(row["message_index"])
    msg = st.session_state["messages"][real_idx]
    read_flag = bool(row["read"])
    arm_id = row.get("arm_id", "")
    msg_unique_key = str(
        row.get("message_key", f"{real_idx}_{arm_id}")
    ).replace(" ", "_").replace(":", "-")

    view_key = f"_push_view_logged_{real_idx}_{arm_id}"
    if not msg.get(view_key, False):
        append_interaction_log(
            event="message_view",
            message_index=real_idx,
            msg=msg,
            extra={"section": "push"},
        )
        st.session_state["messages"][real_idx][view_key] = True

    strategy_type = str(row.get("strategy_type", infer_strategy_type(arm_id))).upper().strip()
    theme = get_strategy_theme(strategy_type)

    ts_short = str(row.get("ts", ""))[:16]
    dorm_id = _html_escape(str(row.get("dorm_id", "")))
    cluster = _html_escape(str(row.get("cluster_type", "")))
    title = _html_escape(row.get("title", ""))
    body = _html_escape(row.get("body", ""))

    meta_parts = []
    if dorm_id:
        meta_parts.append(dorm_id)
    if cluster:
        meta_parts.append(cluster)
    if ts_short:
        meta_parts.append(ts_short)
    meta_text = " · ".join(meta_parts)

    status_text = "完成" if read_flag else "待办"
    status_bg = "#F3F4F6" if read_flag else "#DCFCE7"
    status_fg = "#9CA3AF" if read_flag else "#15803D"
    card_bg = "#FFFFFF" if read_flag else "#F7FFF9"
    border_color = "#E6EDE8" if read_flag else "#BBF7D0"
    accent_color = "#CBD5E1" if read_flag else theme["color"]
    title_color = "#6B7280" if read_flag else "#0A1A0F"
    body_color = "#9CA3AF" if read_flag else "#6B7280"

    card_html = f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8">
      <style>
        * {{
          box-sizing: border-box;
          margin: 0;
          padding: 0;
        }}
        body {{
          background: transparent;
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }}
        .notif-card {{
          position: relative;
          width: 100%;
          background: {card_bg};
          border: 1px solid {border_color};
          border-radius: 16px;
          overflow: hidden;
        }}
        .notif-card-accent {{
          position: absolute;
          left: 0;
          top: 14px;
          bottom: 14px;
          width: 3px;
          border-radius: 0 3px 3px 0;
          background: {accent_color};
        }}
        .notif-card-inner {{
          padding: 16px 18px 12px 22px;
        }}
        .notif-topline {{
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 10px;
          margin-bottom: 8px;
        }}
        .notif-num-row {{
          display: flex;
          align-items: center;
          gap: 7px;
        }}
        .notif-pip {{
          width: 5px;
          height: 5px;
          border-radius: 50%;
          background: #22C55E;
        }}
        .notif-num {{
          font-size: 11px;
          font-weight: 800;
          color: #22C55E;
          letter-spacing: .5px;
        }}
        .notif-status {{
          font-size: 10px;
          font-weight: 600;
          border-radius: 5px;
          padding: 2px 8px;
          white-space: nowrap;
          color: {status_fg};
          background: {status_bg};
        }}
        .notif-brand {{
          display: flex;
          align-items: center;
          gap: 8px;
          flex-wrap: wrap;
          margin-bottom: 8px;
        }}
        .notif-brand-name {{
          font-size: 13px;
          font-weight: 700;
          color: {theme["color"]};
        }}
        .notif-badge {{
          font-size: 10px;
          font-weight: 700;
          color: {theme["color"]};
          background: {theme["bg"]};
          border: 1px solid {theme["border"]};
          border-radius: 999px;
          padding: 3px 10px;
        }}
        .notif-title {{
          font-size: 15px;
          font-weight: 700;
          color: {title_color};
          line-height: 1.3;
          margin-bottom: 6px;
        }}
        .notif-body {{
          font-size: 13px;
          color: {body_color};
          line-height: 1.75;
          margin-bottom: 8px;
        }}
        .notif-meta {{
          font-size: 10px;
          color: #B0BAB5;
          line-height: 1.6;
        }}
      </style>
    </head>
    <body>
      <div class="notif-card">
        <div class="notif-card-accent"></div>
        <div class="notif-card-inner">
          <div class="notif-topline">
            <div class="notif-num-row">
              <span class="notif-pip"></span>
              <span class="notif-num">提醒</span>
            </div>
            <span class="notif-status">{status_text}</span>
          </div>

          <div class="notif-brand">
            <span style="font-size:15px;">🔔</span>
            <span class="notif-brand-name">节能小助手</span>
            <span class="notif-badge">{theme["code"]} · {theme["name"]}</span>
          </div>

          <div class="notif-title">{title}</div>
          <div class="notif-body">{body}</div>
          <div class="notif-meta">{meta_text}</div>
        </div>
      </div>
    </body>
    </html>
    """

    shell_l, shell_c, shell_r = st.columns([0.03, 0.94, 0.03])
    with shell_c:
        components.html(card_html, height=160, scrolling=False)

        btn_l, btn_r = st.columns([6, 1.4])
        with btn_r:
            if read_flag:
                st.button(
                    "完成",
                    key=f"status_{msg_unique_key}",
                    use_container_width=True,
                    disabled=True,
                )
            else:
                if st.button(
                    "待办",
                    key=f"status_{msg_unique_key}",
                    use_container_width=True,
                ):
                    mark_message_read(real_idx)
                    append_interaction_log(
                        event="message_complete",
                        message_index=real_idx,
                        msg=msg,
                        extra={"section": "push"},
                    )
                    st.rerun()


def render_other_card(row: pd.Series):
    real_idx = int(row["message_index"])
    msg = st.session_state["messages"][real_idx]
    arm_id = row.get("arm_id", "")

    view_key = f"_other_view_logged_{real_idx}_{arm_id}"
    if not msg.get(view_key, False):
        append_interaction_log(
            event="message_view",
            message_index=real_idx,
            msg=msg,
            extra={"section": "other"},
        )
        st.session_state["messages"][real_idx][view_key] = True

    with st.container(border=False):
        st.markdown(f"**{row['title']}**")
        st.caption(_message_meta_line(row))
        st.write(row["body"])

        c1, c2 = st.columns(2)
        with c1:
            if st.button("知道了", key=f"ack_other_{real_idx}_{arm_id}"):
                mark_message_read(real_idx)
                append_interaction_log(
                    event="message_ack",
                    message_index=real_idx,
                    msg=msg,
                    extra={"section": "other"},
                )
                st.success("已记录：知道了")
                st.rerun()
        with c2:
            st.caption("已读" if bool(row["read"]) else "未读")

def _latest_score_for_dorm_in_messages(dorm_id: str):
    scores = st.session_state.get("scores", [])
    if not scores:
        return None

    latest = None
    for rec in scores:
        if not isinstance(rec, dict):
            continue
        if str(rec.get("dorm_id", "")) != str(dorm_id):
            continue
        if latest is None or str(rec.get("ts", "")) > str(latest.get("ts", "")):
            latest = rec
    return latest

def _message_exists_by_key_in_messages(message_key: str) -> bool:
    if not message_key:
        return False

    messages = st.session_state.get("messages", [])
    for row in messages:
        if isinstance(row, dict) and str(row.get("message_key", "")).strip() == str(message_key).strip():
            return True
    return False

def _pick_daily_arm(arm_list, date_str: str, offset: int = 0):
    if not arm_list:
        return None
    try:
        day_num = int(str(date_str).replace("-", "")) + int(offset)
    except Exception:
        day_num = int(offset)
    return arm_list[day_num % len(arm_list)]


def ensure_daily_messages_for_current_dorm():
    """
    根据 current_dorm_for_messages + business_date_str
    自动为当前日期补齐 1 条 tip 和 1 条 push。
    已存在则跳过，不重复写入。
    """
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("interaction_logs", [])
    st.session_state.setdefault("intervention_logs", [])

    date_str = str(st.session_state.get("business_date_str", pd.Timestamp.now().strftime("%Y-%m-%d")))
    ts_now = f"{date_str} 09:00:00"

    dorm_id = str(st.session_state.get("current_dorm_for_messages", "")).strip()

    # 如果当前没有显式宿舍，就从 scores 里取一个最近宿舍作为兜底
    if not dorm_id:
        scores = st.session_state.get("scores", [])
        if scores:
            dorm_id = str(scores[-1].get("dorm_id", "")).strip()

    if not dorm_id:
        return 0, "当前没有可用于生成消息的宿舍。请先在 Home 页面选择一个宿舍。"

    latest_score = _latest_score_for_dorm_in_messages(dorm_id)
    if not latest_score:
        return 0, f"宿舍 {dorm_id} 暂无 score，无法判断群体。"

    cluster_type = str(latest_score.get("cluster_type", "")).strip().upper()
    if not cluster_type:
        return 0, f"宿舍 {dorm_id} 暂无 cluster_type，无法生成消息。"

    plan = get_cluster_message_plan(cluster_type)
    tip_arms = plan.get("tip", []) if isinstance(plan, dict) else []
    push_arms = plan.get("push", []) if isinstance(plan, dict) else []

    score_basis = "simulated" if str(dorm_id).startswith("SIM_") else "real"

    generated_n = 0

    # ---------- tip ----------
    tip_arm = _pick_daily_arm(tip_arms, date_str, offset=0)
    if tip_arm:
        tip_msg = build_message_from_arm(
            arm_id=tip_arm,
            dorm_id=dorm_id,
            cluster_type=cluster_type,
            ts=ts_now,
            source_page="messages_auto",
            score_basis=score_basis,
        )
        if isinstance(tip_msg, dict):
            if not _message_exists_by_key_in_messages(tip_msg.get("message_key", "")):
                st.session_state["messages"].append(tip_msg)
                st.session_state["intervention_logs"].append({
                    "timestamp": ts_now,
                    "date": date_str,
                    "dorm_id": dorm_id,
                    "cluster_type": cluster_type,
                    "arm_id": tip_arm,
                    "message_key": tip_msg.get("message_key", ""),
                    "message_channel": tip_msg.get("channel", "tip"),
                    "message_strategy_type": tip_msg.get("strategy_type", "P"),
                    "algo_type": "message_auto",
                    "source_page": "messages_auto",
                    "score_basis": score_basis,
                })
                generated_n += 1

    # ---------- push ----------
    push_arm = _pick_daily_arm(push_arms, date_str, offset=1)
    if push_arm:
        push_msg = build_message_from_arm(
            arm_id=push_arm,
            dorm_id=dorm_id,
            cluster_type=cluster_type,
            ts=ts_now,
            source_page="messages_auto",
            score_basis=score_basis,
        )
        if isinstance(push_msg, dict):
            if not _message_exists_by_key_in_messages(push_msg.get("message_key", "")):
                st.session_state["messages"].append(push_msg)
                st.session_state["intervention_logs"].append({
                    "timestamp": ts_now,
                    "date": date_str,
                    "dorm_id": dorm_id,
                    "cluster_type": cluster_type,
                    "arm_id": push_arm,
                    "message_key": push_msg.get("message_key", ""),
                    "message_channel": push_msg.get("channel", "push"),
                    "message_strategy_type": push_msg.get("strategy_type", "R"),
                    "algo_type": "message_auto",
                    "source_page": "messages_auto",
                    "score_basis": score_basis,
                })
                generated_n += 1

    if generated_n > 0:
        st.session_state["interaction_logs"].append({
            "ts": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "date": date_str,
            "event": "messages_auto_generate",
            "page": "messages",
            "dorm_id": dorm_id,
            "cluster_type": cluster_type,
            "generated_n": generated_n,
        })

    return generated_n, f"{dorm_id} 在 {date_str} 自动补齐了 {generated_n} 条消息。"


# ══════════════════════════════════════════════════════════════════
# 新架构辅助渲染函数
# ══════════════════════════════════════════════════════════════════
def _render_group_header(label: str, count_text: str, color: str):
    c1, c2, c3 = st.columns([1.2, 8, 1], vertical_alignment="center")
    with c1:
        st.markdown(
            f"<span style='font-size:12px;font-weight:800;color:{color};'>{label}</span>",
            unsafe_allow_html=True,
        )
    with c2:
        st.divider()
    with c3:
        st.caption(count_text)


def _render_summary_bar(msg_df: pd.DataFrame, today_str: str):
    """顶部四格摘要，纯原生 st.metric，不注入任何 HTML。"""
    total_n  = len(msg_df)
    unread_n = int((~msg_df["read"]).sum()) if "read" in msg_df.columns else 0
    done_n   = total_n - unread_n
    today_n  = int((msg_df["ts"].astype(str).str.startswith(today_str)).sum()) if "ts" in msg_df.columns else 0

    latest_stype = ""
    if "strategy_type" in msg_df.columns:
        st_counts = msg_df["strategy_type"].value_counts()
        if not st_counts.empty:
            latest_stype = str(st_counts.index[0])

    _STYPE_NAME  = {"P": "信息提示", "R": "定期提醒", "F": "数据反馈", "S": "社会比较"}
    _STYPE_ICON  = {"P": "🔵", "R": "🟢", "F": "🟣", "S": "🟠"}
    _STYPE_COLOR = {"P": ("#2563EB","#EFF6FF"), "R": ("#16A34A","#F0FDF4"),
                    "F": ("#7C3AED","#F5F3FF"), "S": ("#F97316","#FFF7ED")}
    sname  = _STYPE_NAME.get(latest_stype, "未知")
    sicon  = _STYPE_ICON.get(latest_stype, "⚪")
    scolor, sbg = _STYPE_COLOR.get(latest_stype, ("#6B7280", "#F8FAFC"))
    stype_display = f"{sicon} {latest_stype} · {sname}" if latest_stype else "—"

    banner_html = f"""<!doctype html><html><head><meta charset="utf-8"><style>
    *{{box-sizing:border-box;margin:0;padding:0;}}
    body{{background:transparent;font-family:-apple-system,'Inter Tight',sans-serif;}}
    .bar{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;}}
    .stat{{background:#fff;border-radius:14px;padding:16px 18px 14px;
           box-shadow:0 1px 4px rgba(0,0,0,.06),0 0 0 1px rgba(0,0,0,.05);}}
    .stat.highlight{{background:linear-gradient(135deg,#052E12,#166534);
                     box-shadow:0 4px 16px rgba(22,163,74,.25);}}
    .label{{font-size:11px;font-weight:700;color:#9CA3AF;letter-spacing:.7px;
             text-transform:uppercase;margin-bottom:7px;}}
    .label.hl{{color:rgba(255,255,255,.55);}}
    .value{{font-family:'Inter Tight',sans-serif;font-size:34px;font-weight:900;
             color:#061408;line-height:1;letter-spacing:-1.2px;}}
    .value.green{{color:#16A34A;}}
    .value.hl{{color:#4ADE80;}}
    .value.gray{{color:#9CA3AF;}}
    .badge{{display:inline-block;margin-top:6px;font-size:12px;font-weight:700;
             padding:4px 11px;border-radius:7px;}}
    </style></head><body>
    <div class="bar">
      <div class="stat highlight">
        <div class="label hl">待处理</div>
        <div class="value hl">{unread_n}</div>
      </div>
      <div class="stat">
        <div class="label">完成</div>
        <div class="value gray">{done_n}</div>
      </div>
      <div class="stat">
        <div class="label">今日新增</div>
        <div class="value green">{today_n}</div>
      </div>
      <div class="stat">
        <div class="label">主策略类型</div>
        <div class="value" style="font-size:16px;margin-top:4px;">
          <span class="badge" style="background:{sbg};color:{scolor};">{stype_display}</span>
        </div>
      </div>
    </div>
    </body></html>"""
    components.html(banner_html, height=118, scrolling=False)



def _render_new_push_card(row: pd.Series):
    """
    Push 卡片：
    - 按钮真正放在 st.container(border=True) 内部
    - 白底跟随同一个容器
    - 不再出现按钮漂浮到卡片外
    """
    real_idx = int(row["message_index"])
    msg = st.session_state["messages"][real_idx]
    read_flag = bool(row["read"])
    arm_id = str(row.get("arm_id", ""))

    view_key = f"_push_view_logged_{real_idx}_{arm_id}"
    if not msg.get(view_key, False):
        append_interaction_log(
            event="message_view",
            message_index=real_idx,
            msg=msg,
            extra={"section": "push"},
        )
        st.session_state["messages"][real_idx][view_key] = True

    strategy_type = str(row.get("strategy_type", infer_strategy_type(arm_id))).upper().strip()
    theme = get_strategy_theme(strategy_type)
    ts_short = str(row.get("ts", ""))[:10]
    cluster = str(row.get("cluster_type", ""))
    title = str(row.get("title", ""))
    body_txt = str(row.get("body", ""))

    msg_unique_key = str(
        row.get("message_key", f"{real_idx}_{arm_id}")
    ).replace(" ", "_").replace(":", "-")

    chip_html = ""
    if cluster:
        chip_html += f'<span class="push-shell-chip">{_html_escape(cluster)} 群体</span>'
    if ts_short:
        chip_html += f'<span class="push-shell-chip">{_html_escape(ts_short)}</span>'


    status_text = "完成" if read_flag else "待办"
    status_color = "#9CA3AF" if read_flag else theme["color"]

    with st.container():
        top_left, top_right = st.columns([6.6, 0.9], vertical_alignment="center")

        with top_left:
            _render_html(f"""
            <div class="push-card">
                <div class="push-card-bar" style="background:{'#CBD5E1' if read_flag else theme['color']};"></div>
                <div style="padding-left:6px;">
                    <div class="push-shell-top">
                        <div class="push-shell-left">
                            <span class="push-shell-badge" style="color:{theme['color']};background:{theme['bg']};border:1px solid {theme['border']};">
                                {theme['code']} · {theme['name']}
                            </span>
                            {chip_html}
                        </div>
                    </div>
                    <div class="push-shell-title {'read' if read_flag else ''}">{_html_escape(title)}</div>
                    <div class="push-shell-body {'read' if read_flag else ''}">{_html_escape(body_txt)}</div>
                    <div class="push-shell-foot">{_html_escape(arm_id)}</div>
                </div>
            </div>
            """)

        with top_right:
            if read_flag:
                st.button(
                    "完成",
                    key=f"status_done_{msg_unique_key}",
                    use_container_width=True,
                    disabled=True,
                )
            else:
                if st.button(
                    "待办",
                    key=f"status_todo_{msg_unique_key}",
                    use_container_width=True,
                ):
                    mark_message_read(real_idx)
                    append_interaction_log(
                        event="message_complete",
                        message_index=real_idx,
                        msg=msg,
                        extra={"section": "push"},
                    )
                    st.rerun()



def _render_tip_carousel_new(tip_df: pd.DataFrame):
    """
    贴士轮播 v3：
    - iframe 内含完整卡片 + 左右翻页按钮 + 分页点（纯展示+翻页）
    - "知道了"按钮通过 st.button 放在 iframe 正下方，外层 wrapper 统一圆角
    - 元信息精简为群体 + 时间两项
    - 日志写入不变
    """
    if tip_df.empty:
        st.caption("暂无节能贴士")
        return

    _STYPE_NAME = {"P": "信息提示", "R": "定期提醒", "F": "数据反馈", "S": "社会比较"}
    _STYPE_BG   = {
        "P": ("#EFF6FF", "#2563EB", "#BFDBFE"),
        "R": ("#F0FDF4", "#16A34A", "#BBF7D0"),
        "F": ("#F5F3FF", "#7C3AED", "#DDD6FE"),
        "S": ("#FFF7ED", "#F97316", "#FED7AA"),
    }

    cards_data = []
    for _, row in tip_df.iterrows():
        real_idx = int(row["message_index"])
        msg      = st.session_state["messages"][real_idx]
        arm_id   = str(row.get("arm_id", ""))
        stype    = str(row.get("strategy_type", infer_strategy_type(arm_id))).upper()
        cluster  = str(row.get("cluster_type", ""))
        cta      = str(msg.get("cta_text", "知道了"))
        read_f   = bool(row.get("read", False))
        cbg, ccl, cbd = _STYPE_BG.get(stype, ("#F8FAFC", "#64748B", "#E2E8F0"))

        view_key = f"_tip_view_logged_{real_idx}_{arm_id}"
        if not msg.get(view_key, False):
            append_interaction_log(event="message_view", message_index=real_idx, msg=msg,
                                   extra={"section": "tip"})
            st.session_state["messages"][real_idx][view_key] = True

        cards_data.append({
            "real_idx": real_idx,
            "arm_id":   arm_id,
            "stype":    stype,
            "sname":    _STYPE_NAME.get(stype, stype),
            "title":    _html_escape(str(row.get("title", ""))),
            "body":     _html_escape(str(row.get("body", ""))),
            "cluster":  _html_escape(cluster),
            "ts":       str(row.get("ts", ""))[:10],
            "read":     read_f,
            "cta":      _html_escape(cta),
            "cbg": cbg, "ccl": ccl, "cbd": cbd,
        })

    total      = len(cards_data)
    cards_json = _json.dumps(cards_data, ensure_ascii=False)

    # ── 轮播 carousel_key 维护 Python 侧当前页 ───────────────────
    carousel_key = "tip_carousel_cur"
    if carousel_key not in st.session_state:
        st.session_state[carousel_key] = 0
    py_cur = max(0, min(int(st.session_state.get(carousel_key, 0)), total - 1))

    # ── iframe：卡片展示区 + 左右翻页（← →）+ 分页指示 ─────────
    # "知道了"不在 iframe 里，在 iframe 正下方无间距紧贴
    carousel_html = f"""<!doctype html><html><head><meta charset="utf-8"><style>
    *{{box-sizing:border-box;margin:0;padding:0;}}
    html,body{{width:100%;background:transparent;
               font-family:-apple-system,'Inter Tight',sans-serif;overflow:hidden;}}
    .root{{display:flex;flex-direction:column;height:100%;}}
    .track-wrap{{flex:1;overflow:hidden;border-radius:18px 18px 0 0;}}
    .track{{display:flex;transition:transform .36s cubic-bezier(.4,0,.2,1);height:100%;}}
    .slide{{min-width:100%;height:100%;}}
    .card{{
        background:#fff;height:100%;
        box-shadow:0 6px 24px rgba(0,0,0,.07),0 0 0 1px rgba(0,0,0,.05);
        display:flex;flex-direction:column;border-radius:18px 18px 0 0;overflow:hidden;
    }}
    .card-top{{flex:1;padding:20px 20px 14px;}}
    .top-row{{display:flex;align-items:center;gap:7px;margin-bottom:12px;}}
    .badge{{font-size:11px;font-weight:700;padding:4px 12px;border-radius:999px;
             letter-spacing:.3px;}}
    .read-chip{{font-size:10px;color:#9CA3AF;background:#F3F4F6;
                border-radius:5px;padding:2px 8px;}}
    .title{{font-size:21px;font-weight:800;color:#061408;line-height:1.34;
             margin-bottom:10px;letter-spacing:-.35px;}}
    .body{{font-size:13px;color:#374151;line-height:1.72;}}
    .card-meta{{
        padding:10px 20px 12px;
        border-top:1px solid #F0F4F1;
        display:flex;gap:14px;flex-wrap:wrap;
        background:#FAFBFA;
    }}
    .meta-item{{font-size:10px;color:#9CA3AF;display:flex;gap:4px;align-items:center;}}
    .meta-val{{color:#6B7280;font-weight:500;}}
    .nav-bar{{
        display:flex;align-items:center;justify-content:space-between;
        padding:8px 16px;
        background:#F7F9F8;
        border-top:1px solid #EEF2EF;
        border-radius:0;
    }}
    .dots{{display:flex;gap:4px;align-items:center;}}
    .dot{{width:6px;height:6px;border-radius:50%;background:#D1D9D5;
           transition:all .28s ease;cursor:pointer;}}
    .dot.active{{width:22px;border-radius:999px;}}
    .counter{{font-size:12px;font-weight:700;color:#9CA3AF;}}
    .nav-btns{{display:flex;gap:6px;}}
    .nbtn{{
        width:30px;height:30px;border-radius:50%;border:1px solid #E5E7EB;
        background:#fff;color:#374151;font-size:13px;
        cursor:pointer;transition:all .14s;
        display:flex;align-items:center;justify-content:center;
        font-weight:600;
    }}
    .nbtn:hover{{background:#F0FDF4;border-color:#86EFAC;color:#16A34A;}}
    .nbtn:disabled{{opacity:.3;cursor:default;}}
    </style></head><body>
    <div class="root">
      <div class="track-wrap"><div class="track" id="track"></div></div>
      <div class="nav-bar">
        <div style="display:flex;align-items:center;gap:8px;">
          <div class="dots" id="dots"></div>
          <span class="counter" id="counter"></span>
        </div>
        <div class="nav-btns">
          <button class="nbtn" id="prev" onclick="move(-1)">&#8592;</button>
          <button class="nbtn" id="next" onclick="move(1)">&#8594;</button>
        </div>
      </div>
    </div>
    <script>
    const cards = {cards_json};
    const total = cards.length;
    let cur = {py_cur};

    function buildSlide(c) {{
      const readChip = c.read ? '<span class="read-chip">✓ 已读</span>' : '';
      let metaHtml = '';
      if (c.cluster) metaHtml += `<div class="meta-item"><span>群体</span><span class="meta-val">${{c.cluster}}</span></div>`;
      if (c.ts)      metaHtml += `<div class="meta-item"><span>时间</span><span class="meta-val">${{c.ts}}</span></div>`;
      return `<div class="slide"><div class="card">
        <div class="card-top">
          <div class="top-row">
            <span class="badge" style="color:${{c.ccl}};background:${{c.cbg}};border:1px solid ${{c.cbd}};">
              ${{c.stype}} · ${{c.sname}}
            </span>
            ${{readChip}}
          </div>
          <div class="title">${{c.title}}</div>
          <div class="body">${{c.body}}</div>
        </div>
        ${{metaHtml ? `<div class="card-meta">${{metaHtml}}</div>` : ''}}
      </div></div>`;
    }}

    function render() {{
      document.getElementById('track').innerHTML = cards.map(buildSlide).join('');
      document.getElementById('track').style.transform = `translateX(-${{cur * 100}}%)`;
      const maxDots = Math.min(total, 8);
      document.getElementById('dots').innerHTML = Array.from({{length:maxDots}},(_,i)=>{{
        const active = (i === (total<=8 ? cur : Math.floor(cur/total*maxDots)));
        const c = cards[Math.min(i,total-1)];
        return `<div class="dot ${{active?'active':''}}" onclick="jumpTo(${{i}})"
                  style="${{active?`background:${{c.ccl}};`:''}}"></div>`;
      }}).join('');
      document.getElementById('counter').textContent = `${{cur+1}} / ${{total}}`;
      document.getElementById('prev').disabled = cur===0;
      document.getElementById('next').disabled = cur===total-1;
    }}

    function move(dir) {{
      const next = cur + dir;
      if (next<0||next>=total) return;
      cur = next;
      document.getElementById('track').style.transform = `translateX(-${{cur*100}}%)`;
      render();
    }}

    function jumpTo(i) {{
      if (i<0||i>=total) return;
      cur = i;
      document.getElementById('track').style.transform = `translateX(-${{cur*100}}%)`;
      render();
    }}

    render();
    </script>
    </body></html>"""

    iframe_h = 245
    components.html(carousel_html, height=iframe_h, scrolling=False)
    st.markdown('</div>', unsafe_allow_html=True)  # 关闭操作行


def _render_right_panel(msg_df: pd.DataFrame, today_str: str):
    """右侧面板：今日聚焦 + 周历 + 策略分布，纯原生组件 + 局部 iframe。"""

    _STYPE_NAME  = {"P": "信息提示", "R": "定期提醒", "F": "数据反馈", "S": "社会比较"}
    _STYPE_ICON  = {"P": "🔵", "R": "🟢", "F": "🟣", "S": "🟠"}
    _STYPE_COLOR = {
        "P": ("#2563EB", "#EFF6FF", "#BFDBFE"),
        "R": ("#16A34A", "#F0FDF4", "#BBF7D0"),
        "F": ("#7C3AED", "#F5F3FF", "#DDD6FE"),
        "S": ("#F97316", "#FFF7ED", "#FED7AA"),
    }

    # ── 今日聚焦卡 ────────────────────────────────────────────────
    st.markdown(
        '<div style="font-size:10px;font-weight:700;color:#9CA3AF;'
        'letter-spacing:.8px;text-transform:uppercase;margin-bottom:8px;">'
        '📌 今日聚焦</div>',
        unsafe_allow_html=True,
    )

    today_msgs = (msg_df[msg_df["ts"].astype(str).str.startswith(today_str)]
                  if "ts" in msg_df.columns else pd.DataFrame())
    unread_today = (today_msgs[~today_msgs["read"]]
                    if not today_msgs.empty and "read" in today_msgs.columns else pd.DataFrame())
    featured = (unread_today.iloc[0] if not unread_today.empty
                else (today_msgs.iloc[0] if not today_msgs.empty else None))

    if featured is not None:
        stype   = str(featured.get("strategy_type", "R")).upper()
        cluster = str(featured.get("cluster_type", ""))
        title   = _html_escape(str(featured.get("title", "")))
        body    = _html_escape(str(featured.get("body", ""))[:120])
        sc, sbg, sbd = _STYPE_COLOR.get(stype, ("#64748B", "#F8FAFC", "#E2E8F0"))
        chip = f"{stype} · {_STYPE_NAME.get(stype, '')}" + (f" · {cluster} 群体" if cluster else "")

        focus_html = f"""<!doctype html><html><head><meta charset="utf-8"><style>
        *{{box-sizing:border-box;margin:0;padding:0;}}
        body{{background:transparent;font-family:-apple-system,'Inter Tight',sans-serif;}}
        .card{{background:linear-gradient(135deg,#052E12 0%,#166534 100%);
               border-radius:16px;padding:18px 16px 16px;}}
        .pre{{font-size:9px;font-weight:700;color:rgba(255,255,255,.45);
               letter-spacing:1.2px;text-transform:uppercase;margin-bottom:8px;}}
        .title{{font-size:15px;font-weight:800;color:#fff;line-height:1.3;margin-bottom:8px;}}
        .body{{font-size:12px;color:rgba(255,255,255,.72);line-height:1.65;margin-bottom:12px;}}
        .chip{{display:inline-block;font-size:10px;font-weight:600;
                background:rgba(255,255,255,.14);color:rgba(255,255,255,.82);
                border-radius:6px;padding:3px 10px;}}
        </style></head><body>
        <div class="card">
          <div class="pre">最新待处理</div>
          <div class="title">{title}</div>
          <div class="body">{body}{"…" if len(str(featured.get("body","")))>120 else ""}</div>
          <div class="chip">{chip}</div>
        </div>
        </body></html>"""
        components.html(focus_html, height=192, scrolling=False)
    else:
        st.success("✅ 今日消息已全部处理")

    st.divider()

    # ── 消息活跃周历 ──────────────────────────────────────────────
    st.markdown(
        '<div style="font-size:10px;font-weight:700;color:#9CA3AF;'
        'letter-spacing:.8px;text-transform:uppercase;margin-bottom:10px;">'
        '📅 本周活跃</div>',
        unsafe_allow_html=True,
    )
    try:
        today_dt   = pd.Timestamp(today_str)
        weekday    = today_dt.dayofweek
        week_days  = [today_dt - pd.Timedelta(days=weekday - i) for i in range(7)]
        day_labels = ["一", "二", "三", "四", "五", "六", "日"]

        day_counts: dict[str, int] = {}
        if "ts" in msg_df.columns:
            for d in week_days:
                dstr = d.strftime("%Y-%m-%d")
                day_counts[dstr] = int(msg_df["ts"].astype(str).str.startswith(dstr).sum())

        max_c = max(day_counts.values()) if day_counts else 1
        cols7 = st.columns(7)
        for di, d in enumerate(week_days):
            dstr     = d.strftime("%Y-%m-%d")
            cnt      = day_counts.get(dstr, 0)
            is_today = dstr == today_str
            alpha    = max(0.12, cnt / max(max_c, 1)) if cnt > 0 else 0
            with cols7[di]:
                # 日期字母
                label_style = "font-weight:800;color:#16A34A;" if is_today else "color:#9CA3AF;"
                st.markdown(
                    f'<div style="text-align:center;font-size:11px;{label_style}">{day_labels[di]}</div>',
                    unsafe_allow_html=True,
                )
                # 数量格子
                if is_today:
                    border = "2px solid #16A34A"
                    bg     = f"rgba(22,163,74,{max(alpha,0.18):.2f})"
                    color  = "#16A34A"
                elif cnt > 0:
                    border = "1px solid #BBF7D0"
                    bg     = f"rgba(22,163,74,{alpha:.2f})"
                    color  = "#16A34A"
                else:
                    border = "1px solid #E5E7EB"
                    bg     = "#F9FAFB"
                    color  = "#D1D5DB"
                st.markdown(
                    f'<div style="text-align:center;margin-top:4px;'
                    f'width:100%;height:32px;border-radius:8px;'
                    f'background:{bg};border:{border};'
                    f'display:flex;align-items:center;justify-content:center;'
                    f'font-size:11px;font-weight:700;color:{color};">'
                    f'{cnt if cnt > 0 else "·"}</div>',
                    unsafe_allow_html=True,
                )
    except Exception:
        st.caption("周历数据暂不可用")

    st.divider()

    # ── 策略分布 ──────────────────────────────────────────────────
    st.markdown(
        '<div style="font-size:10px;font-weight:700;color:#9CA3AF;'
        'letter-spacing:.8px;text-transform:uppercase;margin-bottom:10px;">'
        '📊 策略分布</div>',
        unsafe_allow_html=True,
    )
    if "strategy_type" in msg_df.columns and not msg_df.empty:
        counts = msg_df["strategy_type"].value_counts()
        total  = len(msg_df)
        _ORDER = ["P", "R", "F", "S"]
        for sk in _ORDER:
            cnt_k = int(counts.get(sk, 0))
            if cnt_k == 0:
                continue
            pct   = cnt_k / total
            sc, sbg, sbd = _STYPE_COLOR.get(sk, ("#64748B", "#F8FAFC", "#E2E8F0"))
            icon  = _STYPE_ICON.get(sk, "⚪")
            sname = _STYPE_NAME.get(sk, sk)
            lc, rc = st.columns([3, 1])
            with lc:
                st.markdown(
                    f'<div style="font-size:11px;font-weight:600;color:{sc};">'
                    f'{icon} {sk} · {sname}</div>',
                    unsafe_allow_html=True,
                )
            with rc:
                st.markdown(
                    f'<div style="font-size:11px;color:#9CA3AF;text-align:right;">{cnt_k} 条</div>',
                    unsafe_allow_html=True,
                )
            st.progress(pct)
    else:
        st.caption("暂无策略数据")

def render_messages_page():
    inject_messages_style()

    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("interaction_logs", [])
    st.session_state.setdefault("intervention_logs", [])

    generated_n, gen_msg = ensure_daily_messages_for_current_dorm()
    messages         = st.session_state.get("messages", [])
    interaction_logs = st.session_state.get("interaction_logs", [])

    if generated_n > 0:
        st.toast(gen_msg, icon="✅")

    if not messages:
        st.info("当前还没有消息。请先到 Home 页面进入该宿舍一次，以生成当日 score。")
        return

    msg_df = build_message_df(messages)
    if msg_df.empty:
        st.info("当前还没有可展示的消息。")
        return

    # ── 今日日期字符串 ─────────────────────────────────────────────
    today_str_val = str(st.session_state.get("business_date_str",
                        pd.Timestamp.now().strftime("%Y-%m-%d")))

    # ── 宿舍/群体信息 ─────────────────────────────────────────────
    latest_dorm = ""
    if "dorm_id" in msg_df.columns and not msg_df["dorm_id"].dropna().empty:
        latest_dorm = str(msg_df["dorm_id"].dropna().iloc[0])
    latest_cluster = ""
    if "cluster_type" in msg_df.columns and not msg_df["cluster_type"].dropna().empty:
        latest_cluster = str(msg_df["cluster_type"].dropna().iloc[0])

    # ══ 页头：标题行 ══════════════════════════════════════════════
    sub_parts = []
    if latest_dorm:    sub_parts.append(latest_dorm)
    if latest_cluster: sub_parts.append(f"{latest_cluster} 群体")
    sub_text = _html_escape(" · ".join(sub_parts) if sub_parts else "节能干预平台")

    unread_n = int((~msg_df["read"]).sum()) if "read" in msg_df.columns else 0
    push_n = int((msg_df["channel"] == "push").sum()) if "channel" in msg_df.columns else 0
    tip_n  = int((msg_df["channel"] == "tip").sum()) if "channel" in msg_df.columns else 0

    st.markdown(
        f"""
        <div class="msg-page-hero">
            <div class="msg-eyebrow">
                <span class="msg-eyebrow-dot"></span>
                MESSAGE CENTER
            </div>
            <div class="msg-page-title">节能消息中心</div>
            <div class="msg-page-sub">{sub_text}</div>
            <div class="msg-page-guide">
                先处理上方待处理提醒，再浏览下方节能贴士；右侧卡片用于快速查看今日重点内容。
            </div>
            <div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:12px;">
                <span class="msg-page-kpi">待处理 {unread_n} 条</span>
                <span class="msg-page-kpi">提醒 {push_n} 条</span>
                <span class="msg-page-kpi">贴士 {tip_n} 条</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ══ 四格摘要 ══════════════════════════════════════════════════
    _render_summary_bar(msg_df, today_str_val)

    # ══ 紧凑工具栏 ════════════════════════════════════════════════
    all_dorms = sorted(
        [str(x) for x in msg_df["dorm_id"].dropna().astype(str).unique().tolist()]
    ) if "dorm_id" in msg_df.columns else []

    current_dorm_for_messages = str(st.session_state.get("current_dorm_for_messages", "")).strip()
    default_dorm_option = (
        current_dorm_for_messages if (current_dorm_for_messages and current_dorm_for_messages in all_dorms)
        else (latest_dorm if latest_dorm else "全部宿舍")
    )

    dorm_options = ["全部宿舍"]
    if default_dorm_option and default_dorm_option not in dorm_options and default_dorm_option in all_dorms:
        dorm_options.append(default_dorm_option)
    for d in all_dorms:
        if d not in dorm_options:
            dorm_options.append(d)

    source_options = ["全部来源"] + sorted(
        [str(x) for x in msg_df["source_page"].dropna().astype(str).unique() if str(x).strip()]
    ) if "source_page" in msg_df.columns else ["全部来源"]

    # filter bar start
    tf1, tf2, tf3, tf4 = st.columns([2, 1.5, 1, 0.9])
    with tf1:
        selected_dorm_filter = st.selectbox(
            "宿舍", options=dorm_options,
            index=dorm_options.index(default_dorm_option) if default_dorm_option in dorm_options else 0,
            key="messages_dorm_filter", label_visibility="collapsed",
        )
    with tf2:
        selected_source_filter = st.selectbox(
            "来源", options=source_options, index=0,
            key="messages_source_filter", label_visibility="collapsed",
        )
    with tf3:
        show_read = st.selectbox(
            "状态", options=["全部状态", "仅未读", "仅已读"],
            key="messages_read_filter", label_visibility="collapsed",
        )
    with tf4:
        if st.button("补发今日", key="messages_manual_generate", use_container_width=True):
            n, msg_r = ensure_daily_messages_for_current_dorm()
            st.toast(msg_r, icon="✅" if n > 0 else "ℹ️")
            st.rerun()
    # filter bar end

    # ── 过滤 ──────────────────────────────────────────────────────
    filtered_df = msg_df.copy()
    if selected_dorm_filter != "全部宿舍":
        filtered_df = filtered_df[filtered_df["dorm_id"].astype(str) == str(selected_dorm_filter)]
    if selected_source_filter != "全部来源":
        filtered_df = filtered_df[filtered_df["source_page"].astype(str) == str(selected_source_filter)]
    if show_read == "仅未读":
        filtered_df = filtered_df[~filtered_df["read"]]
    elif show_read == "仅已读":
        filtered_df = filtered_df[filtered_df["read"]]

    if filtered_df.empty:
        st.info("当前筛选条件下没有消息。")
        return

    msg_df = filtered_df.reset_index(drop=True)

    # ══ 主体：8:4 双栏 ════════════════════════════════════════════
    left_col, right_col = st.columns([8, 4], gap="large")

    # ────────────────────────────────────────────────────────────
    # 左栏：按"今天 / 昨天 / 更早"分组的消息时间流
    # ────────────────────────────────────────────────────────────
    with left_col:
        push_df = msg_df[msg_df["channel"] == "push"].copy().reset_index(drop=True)
        tip_df  = msg_df[msg_df["channel"] == "tip"].copy().reset_index(drop=True)

        # 给 push_df 打时间组标签
        def _time_group(ts_val: str) -> str:
            try:
                d = str(ts_val)[:10]
                if d == today_str_val:
                    return "今天"
                yesterday = (pd.Timestamp(today_str_val) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                if d == yesterday:
                    return "昨天"
                return "更早"
            except Exception:
                return "更早"

        if not push_df.empty:
            push_df["_group"] = push_df["ts"].astype(str).apply(_time_group)

            _GROUP_STYLE = {
                "今天":  ("#16A34A", "#F0FDF4", "#BBF7D0", "#16A34A"),
                "昨天":  ("#6B7280", "#F9FAFB", "#E5E7EB", "#6B7280"),
                "更早":  ("#9CA3AF", "#F9FAFB", "#E5E7EB", "#9CA3AF"),
            }

            # 整段提醒消息统一滚动：从第一条开始进入固定高度窗口；超过约 3 条后不再向下延伸
            # 仍可通过滚轮查看所有历史消息。
            with st.container(height=520, border=False):
                for group_name in ["今天", "昨天", "更早"]:
                    group_df = push_df[push_df["_group"] == group_name]
                    if group_df.empty:
                        continue

                    unread_in_group = int((~group_df["read"]).sum())
                    dot_c, label_c, pill_bg, pill_c = _GROUP_STYLE.get(group_name, _GROUP_STYLE["更早"])
                    pill_html = (
                        f'<span style="font-size:10px;font-weight:700;padding:2px 9px;'
                        f'border-radius:20px;background:{pill_bg};color:{pill_c};margin-left:6px;">'
                        f'{unread_in_group} 待处理</span>'
                    ) if unread_in_group else ""

                    st.markdown(
                        f'<div class="msg-group-hd">'
                        f'<div class="msg-group-dot" style="background:{dot_c};"></div>'
                        f'<span class="msg-group-label" style="color:{label_c};">{group_name}</span>'
                        f'{pill_html}'
                        f'<div class="msg-group-line"></div>'
                        f'<span style="font-size:10px;color:#B0BAB5;">{len(group_df)} 条</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    for _, row in group_df.iterrows():
                        _render_new_push_card(row)
        else:
            st.info("当前筛选下暂无提醒消息。")
        
        # ── 节能贴士区（左栏底部）──────────────────────────────
        if not tip_df.empty:
            tip_unread = int((~tip_df["read"]).sum())
            tip_badge  = f"{len(tip_df)} 条" + (f" · {tip_unread} 未读" if tip_unread else "")
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:8px;'
                f'margin:16px 0 8px;padding-top:12px;border-top:1px solid #E2E8E4;">'
                f'<span style="font-size:10px;font-weight:700;color:#2563EB;'
                f'letter-spacing:.8px;text-transform:uppercase;">🔵 节能贴士</span>'
                f'<span style="font-size:10px;color:#B0BAB5;">{tip_badge}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            _render_tip_carousel_new(tip_df)

        # ── 其他类型消息 ───────────────────────────────────────
        other_df = msg_df[~msg_df["channel"].isin(["tip", "push"])].copy().reset_index(drop=True)
        if not other_df.empty:
            st.markdown(
                '<div style="font-size:10px;font-weight:700;color:#9CA3AF;'
                'letter-spacing:.8px;text-transform:uppercase;'
                'margin:14px 0 8px;padding-top:10px;border-top:1px solid #E2E8E4;">⚪ 其他消息</div>',
                unsafe_allow_html=True,
            )
            for _, row in other_df.iterrows():
                render_other_card(row)

    # ────────────────────────────────────────────────────────────
    # 右栏：今日聚焦 + 周历 + 策略分布 + 贴士
    # ────────────────────────────────────────────────────────────
    with right_col:
        _render_right_panel(msg_df, today_str_val)

    # ══ 底部：导出区 ══════════════════════════════════════════════
    st.write("")
    with st.expander("导出数据", expanded=False):
        d1, d2 = st.columns(2)
        with d1:
            st.markdown("**messages 导出**")
            export_messages_df = pd.DataFrame(st.session_state.get("messages", []))
            if not export_messages_df.empty:
                st.dataframe(export_messages_df, use_container_width=True, height=220)
                st.download_button(
                    "下载 messages.csv",
                    data=export_messages_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name="messages.csv", mime="text/csv",
                    key="download_messages_csv",
                )
            else:
                st.info("当前没有 messages 可导出。")
        with d2:
            st.markdown("**interaction_logs 导出**")
            export_logs_df = pd.DataFrame(interaction_logs)
            if not export_logs_df.empty:
                st.dataframe(export_logs_df, use_container_width=True, height=220)
                st.download_button(
                    "下载 interaction_logs.csv",
                    data=export_logs_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name="interaction_logs.csv", mime="text/csv",
                    key="download_interaction_logs_csv",
                )
            else:
                st.info("当前还没有交互记录。")

# ===================== 手机端消息页 v2（通知卡片流） =====================
def render_mobile_messages_page():
    """
    手机端消息页：只保留通知列表和展开详情，不复用桌面双栏。
    """
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("interaction_logs", [])
    st.session_state.setdefault("intervention_logs", [])

    generated_n, gen_msg = ensure_daily_messages_for_current_dorm()
    if generated_n > 0:
        st.toast(gen_msg, icon="✅")

    messages = st.session_state.get("messages", [])

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;700;900&display=swap');

        .msg-mobile-head {
            padding: 4px 0 7px;
            font-family: 'Noto Sans SC', sans-serif;
        }
        .msg-mobile-title {
            display:flex;
            align-items:center;
            justify-content:space-between;
            gap:10px;
            font-size:23px;
            font-weight:900;
            color:#0D2B18;
            line-height:1.15;
        }
        .msg-mobile-count {
            flex-shrink:0;
            border-radius:999px;
            padding:3px 10px;
            background:#EAF7EF;
            color:#1C6E3D;
            font-size:12px;
            font-weight:900;
        }
        .msg-mobile-sub {
            margin-top:3px;
            font-size:13px;
            color:#7E9287;
            line-height:1.45;
        }
        .msg-day-label {
            margin:8px 0 4px;
            color:#91A39A;
            font-size:12px;
            font-weight:800;
            letter-spacing:.3px;
        }
        .msg-detail-card {
            background:#fff;
            border:1px solid #E6EDE8;
            border-radius:16px;
            padding:11px 13px;
            margin:0 0 4px;
            font-family:'Noto Sans SC', sans-serif;
        }
        .msg-detail-meta {
            display:flex;
            flex-wrap:wrap;
            gap:5px;
            margin-bottom:6px;
        }
        .msg-chip {
            border-radius:999px;
            padding:4px 9px;
            font-size:11px;
            font-weight:800;
            background:#F3F7F4;
            color:#60736A;
        }
        .msg-chip.unread {
            background:#D1FAE5;
            color:#065F46;
        }
        .msg-detail-title {
            font-size:17px;
            font-weight:900;
            color:#10251A;
            line-height:1.35;
            margin-bottom:5px;
            overflow-wrap:anywhere;
        }
        .msg-detail-body {
            font-size:14px;
            color:#465B50;
            line-height:1.75;
            overflow-wrap:anywhere;
        }
        .msg-detail-foot {
            margin-top:8px;
            padding-top:7px;
            border-top:1px solid #EFF3F0;
            font-size:11px;
            color:#A6B4AD;
            line-height:1.55;
            overflow-wrap:anywhere;
        }
        div[data-testid="stExpander"] {
            border:1px solid #E1EAE4 !important;
            border-radius:18px !important;
            background:#FFFFFF !important;
            box-shadow:0 8px 22px rgba(16,24,18,.06) !important;
            margin-bottom:5px !important;
            overflow:hidden !important;
        }
        div[data-testid="stExpander"] details summary {
            min-height:54px !important;
            padding:8px 10px !important;
            font-family:'Noto Sans SC', sans-serif !important;
        }
        div[data-testid="stExpander"] details summary p {
            font-size:14px !important;
            font-weight:800 !important;
            color:#143321 !important;
            line-height:1.4 !important;
            overflow-wrap:anywhere !important;
        }
        div[data-testid="stExpander"] [data-testid="stButton"] > button {
            min-height:38px !important;
            border-radius:999px !important;
            font-size:13px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if not messages:
        st.markdown(
            '<div style="padding:40px 0;text-align:center;color:#9CA3AF;font-size:13px;">'
            '暂无消息<br>请先在首页选择宿舍生成评分</div>',
            unsafe_allow_html=True,
        )
        return

    msg_df = build_message_df(messages)
    if msg_df.empty:
        st.markdown(
            '<div style="padding:40px 0;text-align:center;color:#9CA3AF;font-size:13px;">暂无可展示消息</div>',
            unsafe_allow_html=True,
        )
        return

    # 用户端消息只展示提醒通知；若当天没有 push，兜底展示全部消息，避免录屏空白。
    flow_df = msg_df[msg_df["channel"] == "push"].copy()
    if flow_df.empty:
        flow_df = msg_df.copy()
    flow_df = flow_df.sort_values("ts", ascending=False).reset_index(drop=True)

    unread_n = int((~flow_df["read"]).sum()) if "read" in flow_df.columns else 0
    st.markdown(
        f"""
        <div class="msg-mobile-head">
          <div class="msg-mobile-title">
            <span>消息</span>
            <span class="msg-mobile-count">{unread_n} 未读</span>
          </div>
          <div class="msg-mobile-sub">节能提醒和系统推送集中在这里，展开卡片查看详情。</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    today_str = str(st.session_state.get("business_date_str", pd.Timestamp.now().strftime("%Y-%m-%d")))
    yesterday_str = (pd.to_datetime(today_str) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    def _group(ts_val: str) -> str:
        day = str(ts_val)[:10]
        if day == today_str:
            return "今天"
        if day == yesterday_str:
            return "昨天"
        return "更早"

    def _time_text(ts_val: str) -> str:
        txt = str(ts_val)
        day = txt[:10]
        if day == today_str:
            return txt[11:16] or "今天"
        if day == yesterday_str:
            return "昨天"
        return day[5:] if len(day) >= 10 else txt[:16]

    flow_df["_group"] = flow_df["ts"].astype(str).apply(_group)
    type_names = {"P": "信息提示", "R": "定期提醒", "F": "数据反馈", "S": "社会比较"}

    for group_name in ["今天", "昨天", "更早"]:
        group_df = flow_df[flow_df["_group"] == group_name]
        if group_df.empty:
            continue

        st.markdown(f'<div class="msg-day-label">{group_name}</div>', unsafe_allow_html=True)
        for pos, row in group_df.iterrows():
            real_idx = int(row.get("message_index", pos))
            is_read = bool(row.get("read", False))
            title = str(row.get("title", "")).strip() or "节能提醒"
            body = str(row.get("body", "")).strip()
            arm_id = str(row.get("arm_id", "")).strip()
            stype = str(row.get("strategy_type", infer_strategy_type(arm_id))).upper()
            ts_val = str(row.get("ts", ""))
            status = "已读" if is_read else "未读"
            dot = "○" if is_read else "●"
            summary = f"{dot} {_time_text(ts_val)} · {type_names.get(stype, '消息')} · {title}"

            with st.expander(summary, expanded=False):
                chip_cls = "" if is_read else " unread"
                st.markdown(
                    f"""
                    <div class="msg-detail-card">
                      <div class="msg-detail-meta">
                        <span class="msg-chip{chip_cls}">{status}</span>
                        <span class="msg-chip">{_html_escape(type_names.get(stype, stype))}</span>
                      </div>
                      <div class="msg-detail-title">{_html_escape(title)}</div>
                      <div class="msg-detail-body">{_html_escape(body)}</div>
                      <div class="msg-detail-foot">
                        时间：{_html_escape(ts_val[:16]) if ts_val else "—"}
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if not is_read:
                    if st.button("标为已读", key=f"mob_notice_read_{real_idx}", use_container_width=True):
                        mark_message_read(real_idx)
                        append_interaction_log(
                            event="message_view",
                            message_index=real_idx,
                            msg=messages[real_idx] if real_idx < len(messages) else {},
                            extra={"section": "mobile_notice_expander"},
                        )
                        st.rerun()

    return
