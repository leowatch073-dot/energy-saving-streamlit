"""
profile_page.py
===============
「我的」Tab 页面。

结构：
  render_profile_page()          ← app.py 调用入口，不改参数
    ├─ _profile_main()           ← 主页面（用户卡 / 统计 / 连续天数 / 徽章）
    └─ _profile_detail()         ← 详情子页（节能记录 / CS-BS 象限 / 热力图）

跳转通过 session_state["profile_subpage"] 控制：
  "main"   → 主页面
  "detail" → 详情子页

所有数据只读，不写入任何现有 session_state 键：
  scores / user_progress_by_dorm / interaction_logs
  weekly_metrics_by_dorm / dorm_outcome_map / messages
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from datetime import date as _date, timedelta as _td

from state.user_progress import get_badges, get_user_progress


# ══════════════════════════════════════════════════════════════
# 公共工具
# ══════════════════════════════════════════════════════════════

def _safe_float(x, default=0.0) -> float:
    try:
        v = float(x)
        return default if (v != v) else v  # NaN guard
    except Exception:
        return float(default)


def _html(s: str):
    st.markdown(s.strip(), unsafe_allow_html=True)


def _iframe_html(body: str, css: str = "", height: int = 200):
    """
    用 components.html 渲染复杂嵌套 HTML，避免 st.markdown sanitizer 截断。
    body: 完整的 HTML body 内容（不含 <html>/<head> 等外层标签）
    css:  额外的 CSS 字符串（可选）
    height: iframe 高度（px）
    """
    full = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<style>"
        "*{box-sizing:border-box;margin:0;padding:0;}"
        "html,body{background:transparent;font-family:'Noto Sans SC',-apple-system,sans-serif;"
        "overflow:hidden;scrollbar-width:none;}"
        "body::-webkit-scrollbar{display:none;}"
        + css +
        "</style></head><body>"
        + body +
        "</body></html>"
    )
    components.html(full, height=height, scrolling=False)


def _esc(s) -> str:
    return (str(s)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;"))


# ── 从 session_state 读取当前宿舍 ──────────────────────────────
def _current_dorm() -> str:
    return str(st.session_state.get("current_dorm_for_messages", "")).strip() or "—"


# ── 取最新 score 记录 ─────────────────────────────────────────
def _latest_score(dorm_id: str) -> dict | None:
    scores = st.session_state.get("scores", [])
    best = None
    for rec in scores:
        if not isinstance(rec, dict):
            continue
        if str(rec.get("dorm_id", "")) != dorm_id:
            continue
        if best is None or str(rec.get("ts", "")) > str(best.get("ts", "")):
            best = rec
    return best


# ── 连续节能天数（从统一用户进度读取） ───────────────────────
def _calc_streak(dorm_id: str) -> int:
    progress = get_user_progress(dorm_id)
    return int(progress.get("streak_days", 0))


# ── 消息已读数量 ──────────────────────────────────────────────
def _read_msg_count() -> int:
    return sum(
        1 for m in st.session_state.get("messages", [])
        if isinstance(m, dict) and bool(m.get("read", False))
    )


# ── 群体标签 ─────────────────────────────────────────────────
_CLUSTER_NAMES = {
    "A": "高意识低行为型",
    "B": "习惯待建立型",
    "C": "稳定执行型",
    "D": "中等协调型",
    "E": "节能积极型",
}

# ══════════════════════════════════════════════════════════════
# 样式注入（仅用户端使用，不影响 Admin）
# ══════════════════════════════════════════════════════════════

def _inject_profile_style():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;700;900&display=swap');

        /* ── 全局 ── */
        .prof-wrap {
            font-family: 'Noto Sans SC', sans-serif;
            max-width: 100%;
            overflow-x: hidden;
        }
        .prof-wrap * { box-sizing: border-box; }

        /* ── 英雄卡 ── */
        .prof-hero {
            background: linear-gradient(145deg, #052E16 0%, #1C6E3D 100%);
            border-radius: 20px;
            padding: 15px 16px;
            margin: 4px 0 10px;
            box-shadow: 0 10px 28px rgba(16,24,18,.10);
        }
        .prof-hero-row { display: flex; align-items: center; gap: 14px; min-width: 0; }
        .prof-av {
            width: 50px; height: 50px; border-radius: 50%;
            background: rgba(255,255,255,0.14);
            border: 2px solid rgba(255,255,255,0.30);
            display: flex; align-items: center; justify-content: center;
            font-size: 20px; font-weight: 900; color: #fff; flex-shrink: 0;
        }
        .prof-name { font-size: 18px; font-weight: 900; color: #fff; overflow-wrap:anywhere; }
        .prof-dorm { font-size: 12px; color: rgba(255,255,255,0.66); margin-top: 3px; overflow-wrap:anywhere; }
        .prof-cluster-pill {
            display: inline-block; margin-top: 7px;
            font-size: 12px; font-weight: 800;
            background: rgba(74,222,128,0.18); color: #6EF0A2;
            border-radius: 99px; padding: 3px 12px;
            border: 1px solid rgba(74,222,128,0.22);
        }

        /* CS/BS 进度条 */
        .csbs-wrap { margin-top: 12px; display: flex; flex-direction: column; gap: 6px; }
        .csbs-row { display: flex; align-items: center; gap: 8px; }
        .csbs-key {
            font-size: 10px; font-weight: 800; color: rgba(255,255,255,0.58);
            width: 24px; flex-shrink: 0; letter-spacing: 0.5px;
        }
        .csbs-track {
            flex: 1; height: 5px;
            background: rgba(255,255,255,0.14);
            border-radius: 99px; overflow: hidden;
        }
        .csbs-fill { height: 100%; border-radius: 99px; }
        .csbs-val {
            font-size: 11px; font-weight: 900; min-width: 30px; text-align: right;
        }

        /* ── 三格统计 ── */
        .summary-grid {
            display:grid;
            grid-template-columns:1fr 1fr;
            gap:8px;
            margin-bottom:9px;
        }
        .summary-card {
            min-width:0;
            background:#fff;
            border:1px solid #E6EDE8;
            border-radius:16px;
            padding:12px 12px;
            box-shadow:0 8px 22px rgba(16,24,18,.05);
        }
        .summary-k {
            font-size:10px;
            font-weight:900;
            color:#8A9E94;
            letter-spacing:.4px;
            margin-bottom:5px;
        }
        .summary-v {
            font-size:15px;
            font-weight:900;
            color:#0D2B18;
            line-height:1.25;
            overflow-wrap:anywhere;
        }
        .summary-s {
            font-size:11px;
            font-weight:700;
            color:#6B8A78;
            margin-top:3px;
            line-height:1.35;
            overflow-wrap:anywhere;
        }
        .profile-overview {
            background:#fff;
            border:1px solid #E6EDE8;
            border-radius:16px;
            padding:13px 14px;
            margin-bottom:10px;
            box-shadow:0 8px 22px rgba(16,24,18,.05);
        }
        .overview-row {
            display:flex;
            justify-content:space-between;
            gap:12px;
            padding:9px 0;
            border-bottom:1px solid #F0F4F1;
        }
        .overview-row:last-child { border-bottom:none; }
        .overview-k {
            font-size:11px;
            font-weight:900;
            color:#8A9E94;
            flex-shrink:0;
        }
        .overview-v {
            font-size:13px;
            font-weight:900;
            color:#0D2B18;
            text-align:right;
            line-height:1.35;
            overflow-wrap:anywhere;
        }
        .overview-note {
            display:block;
            margin-top:2px;
            font-size:10px;
            font-weight:700;
            color:#6B8A78;
        }
        .stat-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 7px; margin-bottom: 9px; }
        .stat-box {
            min-width:0;
            background: #fff; border-radius: 15px; padding: 11px 7px; text-align: center;
            box-shadow:0 6px 18px rgba(16,24,18,.045);
        }
        .stat-val { font-size: clamp(15px, 4vw, 19px); font-weight: 900; color: #0D3D20; line-height: 1.05; overflow-wrap:anywhere; }
        .stat-lbl { font-size: 10px; color: #6B8A78; font-weight: 700; margin-top: 4px; overflow-wrap:anywhere; }

        /* ── 连续天数卡 ── */
        .streak-card {
            background: linear-gradient(180deg,#F7FCF8 0%,#FFFFFF 100%);
            border: 1px solid #DDEBE2;
            border-radius: 16px;
            padding: 12px 13px;
            margin-bottom: 10px;
            min-width:0;
            box-shadow:0 8px 22px rgba(16,24,18,.045);
        }
        .streak-top {
            display:flex;
            align-items:center;
            justify-content:space-between;
            gap:10px;
            margin-bottom:10px;
        }
        .streak-title {
            font-size:13px;
            font-weight:900;
            color:#0D2B18;
        }
        .streak-sub { font-size:11px; color:#6B8A78; margin-top:2px; font-weight:700; }
        .streak-big {
            font-size:24px;
            font-weight:900;
            color:#1C6E3D;
            line-height:1;
            white-space:nowrap;
        }
        .streak-big span {
            font-size:11px;
            color:#6B8A78;
            margin-left:2px;
        }
        .badge-grid {
            display:grid;
            grid-template-columns:repeat(4,minmax(0,1fr));
            gap:6px;
            margin-bottom:8px;
        }
        .badge-item {
            min-width:0;
            display:flex;
            flex-direction:column;
            align-items:center;
            gap:3px;
            padding:6px 3px;
            border-radius:12px;
            background:#F7FBF8;
            border:1px solid #EDF3EF;
        }
        .badge-ic {
            width:30px;
            height:30px;
            border-radius:10px;
            display:flex;
            align-items:center;
            justify-content:center;
            font-size:16px;
        }
        .badge-nm {
            font-size:9px;
            font-weight:800;
            color:#3A5A46;
            text-align:center;
            line-height:1.2;
            overflow-wrap:anywhere;
        }
        .badge-lock { opacity:0.38; filter:saturate(.7); }
        .next-goal {
            background:#ECFDF3;
            border:1px solid #BBF7D0;
            border-radius:10px;
            padding:7px 9px;
            font-size:10px;
            color:#1C6E3D;
            font-weight:800;
            line-height:1.35;
        }

        /* ── 详情子页 ── */
        .detail-back-btn {
            display: flex; align-items: center; gap: 8px;
            font-size: 14px; font-weight: 800; color: #1C6E3D;
            padding: 10px 0 12px; cursor: pointer;
        }

        /* 节能记录条形 */
        .rec-card {
            background: #fff; border-radius: 16px;
            padding: 13px 14px; margin-bottom: 9px;
            box-shadow:0 6px 18px rgba(16,24,18,.045);
        }
        .rec-hd { font-size: 14px; font-weight: 900; color: #0D2B18; margin-bottom: 10px; }
        .rec-row { display: flex; align-items: center; gap: 8px; padding: 4px 0; border-bottom: 0.5px solid #F0F4F1; }
        .rec-row:last-child { border-bottom: none; }
        .rec-day { font-size: 11px; color: #8A9E94; width: 24px; flex-shrink: 0; }
        .rec-track { flex: 1; height: 6px; background: #EEF2EE; border-radius: 99px; overflow: hidden; }
        .rec-fill { height: 100%; border-radius: 99px; }
        .rec-val { font-size: 11px; font-weight: 800; min-width: 36px; text-align: right; }

        /* CS/BS 象限 */
        .quad-wrap {
            background: #fff; border-radius: 16px;
            padding: 13px 14px; margin-bottom: 9px;
            box-shadow:0 6px 18px rgba(16,24,18,.045);
        }
        .quad-hd { font-size: 14px; font-weight: 900; color: #0D2B18; margin-bottom: 10px; }
        .quad-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 5px; }
        .quad-q { border-radius: 10px; padding: 8px 10px; position: relative; }
        .quad-ql { font-size: 11px; font-weight: 800; }
        .quad-qd { font-size: 10px; color: #6B8A78; margin-top: 2px; line-height: 1.4; }
        .quad-you {
            display: inline-flex; align-items: center; gap: 4px;
            margin-top: 6px; font-size: 10px; font-weight: 800; color: #1C6E3D;
        }
        .quad-dot { width: 8px; height: 8px; border-radius: 50%; background: #1C6E3D; flex-shrink: 0; }

        /* 热力图 */
        .heat-wrap {
            background: #fff; border-radius: 16px;
            padding: 13px 14px; margin-bottom: 9px;
            box-shadow:0 6px 18px rgba(16,24,18,.045);
        }
        .heat-hd { font-size: 14px; font-weight: 900; color: #0D2B18; margin-bottom: 10px; }
        .heat-legend { display: flex; gap: 10px; margin-bottom: 8px; flex-wrap: wrap; }
        .heat-leg-item { display: flex; align-items: center; gap: 4px; font-size: 10px; color: #6B8A78; }
        .heat-leg-dot { width: 8px; height: 8px; border-radius: 2px; flex-shrink: 0; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════
# 主页面
# ══════════════════════════════════════════════════════════════

def _profile_main(dorm_id: str, score: dict | None, streak: int, read_n: int, progress: dict):
    cs  = _safe_float(score.get("CS", 0.5) if score else 0.5)
    bs  = _safe_float(score.get("BS_energy_week", 0.5) if score else 0.5)
    cluster = str(score.get("cluster_type", "—") if score else "—")
    cluster_label = _CLUSTER_NAMES.get(cluster, f"{cluster} 类")
    messages = st.session_state.get("messages", [])
    latest_msg = next((m for m in reversed(messages) if isinstance(m, dict)), {})
    latest_arm = str(latest_msg.get("arm_id", "—"))
    latest_strategy = str(latest_msg.get("strategy_type", "")).upper()
    if not latest_strategy:
        latest_strategy = latest_arm[:1].upper() if latest_arm and latest_arm != "—" else "—"
    strategy_name = {
        "P": "信息提示",
        "R": "定期提醒",
        "F": "数据反馈",
        "S": "社会比较",
    }.get(latest_strategy, "待生成")

    # ── ① 英雄卡 ──────────────────────────────────────────────
    cs_pct = max(4, min(100, int(cs * 100)))
    bs_pct = max(4, min(100, int(bs * 100)))
    cs_color = "#4ADE80" if cs >= 0.6 else "#FCD34D"
    bs_color = "#86EFAC" if bs >= 0.6 else "#FDE68A"

    _html(f"""
    <div class="prof-wrap">
    <div class="prof-hero">
        <div class="prof-hero-row">
            <div class="prof-av">张</div>
            <div>
                <div class="prof-name">张同学</div>
                <div class="prof-dorm">宿舍 {_esc(dorm_id)}</div>
                <div class="prof-cluster-pill">{_esc(cluster_label)}</div>
            </div>
        </div>
        <div class="csbs-wrap">
            <div class="csbs-row">
                <span class="csbs-key">CS</span>
                <div class="csbs-track">
                    <div class="csbs-fill" style="width:{cs_pct}%;background:{cs_color};"></div>
                </div>
                <span class="csbs-val" style="color:{cs_color};">{cs:.2f}</span>
            </div>
            <div class="csbs-row">
                <span class="csbs-key">BS</span>
                <div class="csbs-track">
                    <div class="csbs-fill" style="width:{bs_pct}%;background:{bs_color};"></div>
                </div>
                <span class="csbs-val" style="color:{bs_color};">{bs:.2f}</span>
            </div>
        </div>
    </div>
    </div>
    """)

    wm  = st.session_state.get("weekly_metrics_by_dorm", {}).get(dorm_id, {})
    actual   = _safe_float(wm.get("actual_sum"))
    baseline = _safe_float(wm.get("baseline_sum"), 1.0)
    rate     = (baseline - actual) / baseline if baseline > 1e-9 else 0.0
    rate_txt = f"−{rate*100:.1f}%" if rate > 0 else (f"+{abs(rate)*100:.1f}%" if rate < 0 else "0.0%")
    rate_col = "#1C6E3D" if rate > 0 else ("#D97706" if rate < 0 else "#6B7280")
    kwh_txt  = f"{actual:.1f}" if actual else "—"
    total_xp = int(progress.get("total_xp", 0))
    weekly_count = len(progress.get("weekly_checkins", []))

    _html(f"""
    <div class="profile-overview">
        <div class="overview-row">
            <div class="overview-k">用户画像</div>
            <div class="overview-v">张同学<span class="overview-note">宿舍 {_esc(dorm_id)}</span></div>
        </div>
        <div class="overview-row">
            <div class="overview-k">当前分组</div>
            <div class="overview-v">{_esc(cluster_label)}<span class="overview-note">CS {cs:.2f} · BS {bs:.2f}</span></div>
        </div>
        <div class="overview-row">
            <div class="overview-k">干预策略</div>
            <div class="overview-v">{_esc(strategy_name)}<span class="overview-note">面向当前分组推送</span></div>
        </div>
        <div class="overview-row">
            <div class="overview-k">累计表现</div>
            <div class="overview-v">{streak} 天<span class="overview-note">本周打卡 {weekly_count}/7 · {total_xp} XP · 本周 {kwh_txt} kWh · <span style="color:{rate_col};">{rate_txt}</span></span></div>
        </div>
    </div>
    """)

    items_html = ""
    for badge in get_badges(progress):
        unlocked = bool(badge.get("unlocked"))
        lock_cls = "" if unlocked else "badge-lock"
        title_attr = f'已解锁：{badge["name"]}' if unlocked else badge["desc"]
        items_html += f"""
        <div class="badge-item {lock_cls}" title="{_esc(title_attr)}">
            <div class="badge-ic" style="background:{badge["bg"]};">{badge["icon"]}</div>
            <div class="badge-nm">{_esc(badge["name"])}</div>
        </div>"""

    next_badge = next((badge for badge in get_badges(progress) if not badge.get("unlocked")), None)
    if next_badge:
        next_goal_html = f'下一个：{_esc(next_badge["name"])} — {_esc(next_badge["desc"])}'
    else:
        next_goal_html = "已解锁全部成就"

    unlock_days = max(0, 10 - streak)
    unlock_text = "已解锁「节能达人」" if streak >= 10 else f"再坚持 {unlock_days} 天解锁「节能达人」"
    _html(f"""
    <div class="streak-card">
        <div class="streak-top">
            <div>
                <div class="streak-title">连续节能</div>
                <div class="streak-sub">{_esc(unlock_text)}</div>
            </div>
            <div class="streak-big">{streak}<span>天</span></div>
        </div>
        <div class="badge-grid">{items_html}</div>
        <div class="next-goal">{next_goal_html}</div>
    </div>
    """)

    # ── 「查看详细记录」跳转按钮 ─────────────────────────────
    st.markdown("<div style='height:2px;'></div>", unsafe_allow_html=True)
    if st.button(
        "📈  查看详细记录  →",
        key="profile_goto_detail",
        use_container_width=True,
    ):
        st.session_state["profile_subpage"] = "detail"
        st.rerun()


# ══════════════════════════════════════════════════════════════
# 详情子页
# ══════════════════════════════════════════════════════════════

def _profile_detail(dorm_id: str, score: dict | None, streak: int):
    # ── 返回按钮 ──────────────────────────────────────────────
    if st.button("← 返回我的", key="profile_back", use_container_width=False):
        st.session_state["profile_subpage"] = "main"
        st.rerun()

    _html('<div style="height:4px;"></div>')

    # ══ ⑤ 本周节能记录（逐日条形） ═══════════════════════════
    dorm_out = st.session_state.get("dorm_outcome_map", {}).get(dorm_id)
    today_str = str(st.session_state.get(
        "business_date_str", pd.Timestamp.now().strftime("%Y-%m-%d")
    ))

    day_records: list[tuple[str, float, float]] = []  # (label, rate, color)
    if dorm_out is not None and not dorm_out.empty:
        tmp = dorm_out.copy()
        tcol = "timestamp_hour" if "timestamp_hour" in tmp.columns else "timestamp"
        tmp["_t"] = pd.to_datetime(tmp.get(tcol), errors="coerce")

        actual_col   = "kwh" if "kwh" in tmp.columns else (
                        "energy_kwh" if "energy_kwh" in tmp.columns else None)
        baseline_col = "baseline_pred" if "baseline_pred" in tmp.columns else None

        if actual_col and baseline_col:
            end_dt   = pd.to_datetime(today_str) + pd.Timedelta(days=1)
            start_dt = end_dt - pd.Timedelta(days=7)
            wk = tmp[(tmp["_t"] >= start_dt) & (tmp["_t"] < end_dt)].copy()
            wk["_date"] = wk["_t"].dt.date

            day_labels = {0:"今天",1:"昨天",2:"前天"}
            for delta in range(7):
                d = (pd.to_datetime(today_str) - pd.Timedelta(days=delta)).date()
                sub = wk[wk["_date"] == d]
                if sub.empty:
                    continue
                a = float(pd.to_numeric(sub[actual_col], errors="coerce").fillna(0).sum())
                b = float(pd.to_numeric(sub[baseline_col], errors="coerce").fillna(0).sum())
                if b < 1e-9:
                    continue
                r = (b - a) / b
                lbl = day_labels.get(delta, d.strftime("%m/%d").lstrip("0").replace("/0", "/"))
                color = "#1C6E3D" if r > 0.05 else ("#34D399" if r > 0 else "#F59E0B")
                day_records.append((lbl, r, color))
        day_records = list(reversed(day_records))  # 时间从早到晚

    if day_records:
        rows_html = ""
        for lbl, r, color in day_records:
            bar_w  = max(3, min(100, int(abs(r) * 100 * 6)))
            val_txt = f"−{r*100:.0f}%" if r > 0 else f"+{abs(r)*100:.0f}%"
            rows_html += (
                f'<div class="rec-row">'
                f'<div class="rec-day">{_esc(lbl)}</div>'
                f'<div class="rec-track"><div class="rec-fill" style="width:{bar_w}%;background:{color};"></div></div>'
                f'<div class="rec-val" style="color:{color};">{val_txt}</div>'
                f'</div>'
            )
        rec_css = (
            ".rec-card{background:#fff;border-radius:14px;padding:14px 15px;}"
            ".rec-hd{font-size:15px;font-weight:900;color:#0D2B18;margin-bottom:11px;}"
            ".rec-row{display:flex;align-items:center;gap:8px;padding:7px 0;"
            "border-bottom:0.5px solid #F0F4F1;}"
            ".rec-row:last-child{border-bottom:none;}"
            ".rec-day{font-size:12px;color:#8A9E94;width:32px;flex-shrink:0;}"
            ".rec-track{flex:1;height:6px;background:#EEF2EE;border-radius:99px;overflow:hidden;}"
            ".rec-fill{height:100%;border-radius:99px;}"
            ".rec-val{font-size:12px;font-weight:800;min-width:42px;text-align:right;}"
        )
        rec_h = 46 + len(day_records) * 34
        _iframe_html(
            body=f'<div class="rec-card"><div class="rec-hd">本周节能记录（vs 基线）</div>{rows_html}</div>',
            css=rec_css,
            height=rec_h,
        )
    else:
        _html("""
        <div style="background:#fff;border-radius:14px;padding:14px;
                    margin-bottom:10px;color:#9CA3AF;font-size:12px;text-align:center;">
            暂无逐日用电数据，请先在首页选择宿舍并导入数据。
        </div>
        """)

    # ══ ⑥ CS/BS 象限定位 ═════════════════════════════════════
    cs = _safe_float(score.get("CS", 0.5) if score else 0.5)
    bs = _safe_float(score.get("BS_energy_week", 0.5) if score else 0.5)
    hi_cs = cs >= 0.5
    hi_bs = bs >= 0.5

    quads = [
        # (label, desc, hi_cs, hi_bs, bg, border, text_color)
        ("高意识 · 高行为", "习惯稳固，适合正向强化",   True,  True,  "#F0FDF4", "#BBF7D0", "#1C6E3D"),
        ("高意识 · 低行为", "认知强，执行待提升",       True,  False, "#EFF6FF", "#BFDBFE", "#2563EB"),
        ("低意识 · 高行为", "行为好，继续稳定习惯",     False, True,  "#F9FAFB", "#E5E7EB", "#6B7280"),
        ("低意识 · 低行为", "降低行动门槛，逐步养成",   False, False, "#F9FAFB", "#E5E7EB", "#6B7280"),
    ]

    quads_html = ""
    for ql, qd, qc, qb, qbg, qbd, qtxt in quads:
        is_you = (hi_cs == qc and hi_bs == qb)
        border_w = "2px" if is_you else "1px"
        you_badge = (
            '<div class="quad-you"><div class="quad-dot"></div> 当前位置</div>'
            if is_you else ""
        )
        quads_html += (
            f'<div class="quad-q" style="background:{qbg};border:{border_w} solid {qbd};">'
            f'<div class="quad-ql" style="color:{qtxt};">{_esc(ql)}</div>'
            f'<div class="quad-qd">{_esc(qd)}</div>'
            f'{you_badge}'
            f'</div>'
        )

    quad_css = (
        ".quad-wrap{background:#fff;border-radius:14px;padding:14px 15px;}"
        ".quad-hd{font-size:15px;font-weight:900;color:#0D2B18;margin-bottom:11px;}"
        ".quad-hd span{font-size:12px;color:#6B8A78;font-weight:600;margin-left:6px;}"
        ".quad-grid{display:grid;grid-template-columns:1fr 1fr;gap:5px;}"
        ".quad-q{border-radius:10px;padding:9px 10px;}"
        ".quad-ql{font-size:12px;font-weight:800;}"
        ".quad-qd{font-size:11px;color:#6B8A78;margin-top:3px;line-height:1.45;}"
        ".quad-you{display:inline-flex;align-items:center;gap:4px;margin-top:6px;"
        "font-size:11px;font-weight:800;color:#1C6E3D;}"
        ".quad-dot{width:8px;height:8px;border-radius:50%;background:#1C6E3D;flex-shrink:0;}"
    )
    _iframe_html(
        body=(
            f'<div class="quad-wrap">'
            f'<div class="quad-hd">意识\u2013行为象限'
            f'<span>CS={cs:.2f} · BS={bs:.2f}</span></div>'
            f'<div class="quad-grid">{quads_html}</div>'
            f'</div>'
        ),
        css=quad_css,
        height=200,
    )

    # ══ ⑦ 近 4 周互动热力图（interaction_logs） ══════════════
    ilog = st.session_state.get("interaction_logs", [])
    stypes = ["P", "R", "F", "S"]
    stype_colors = {
        "P": ("#EFF6FF", "#93C5FD", "#2563EB"),   # (empty, mid, full)
        "R": ("#F0FDF4", "#4ADE80", "#16A34A"),
        "F": ("#F5F3FF", "#C4B5FD", "#7C3AED"),
        "S": ("#FFF7ED", "#FED7AA", "#F97316"),
    }
    stype_names = {"P": "信息提示", "R": "定期提醒", "F": "数据反馈", "S": "社会比较"}

    # 构建 date × stype 计数矩阵（最近 28 天）
    heat_data: dict[str, dict[str, int]] = {}  # {date_str: {stype: count}}
    if ilog:
        df_ilog = pd.DataFrame(ilog)
        if not df_ilog.empty and "date" in df_ilog.columns:
            today_d = _date.today()
            for delta in range(27, -1, -1):
                d = (today_d - _td(days=delta)).strftime("%Y-%m-%d")
                heat_data[d] = {s: 0 for s in stypes}
            for _, row in df_ilog.iterrows():
                ds  = str(row.get("date", ""))[:10]
                st_ = str(row.get("strategy_type", row.get("channel", ""))).strip().upper()
                if ds in heat_data and st_ in stypes:
                    heat_data[ds][st_] += 1

    if heat_data:
        legend_html = "".join(
            f'<div class="heat-leg-item">'
            f'<div class="heat-leg-dot" style="background:{stype_colors[s][2]};"></div>'
            f'<span>{stype_names[s]}</span></div>'
            for s in stypes
        )

        rows_html = ""
        for s in stypes:
            cells = ""
            for ds, counts in heat_data.items():
                cnt  = counts.get(s, 0)
                cell_bg = stype_colors[s][0] if cnt == 0 else (stype_colors[s][1] if cnt == 1 else stype_colors[s][2])
                cells += (
                    f'<div class="heat-cell" style="background:{cell_bg};"></div>'
                )
            rows_html += (
                f'<div class="heat-row">'
                f'<span class="heat-row-label">{s}</span>'
                f'<div class="heat-cells">{cells}</div>'
                f'</div>'
            )

        date_keys = list(heat_data.keys())
        label_start = date_keys[0][5:] if date_keys else ""
        label_end   = date_keys[-1][5:] if date_keys else ""

        heat_css = (
            ".heat-wrap{background:#fff;border-radius:14px;padding:15px 16px;overflow:hidden;}"
            ".heat-hd{font-size:16px;font-weight:900;color:#0D2B18;margin-bottom:10px;}"
            ".heat-legend{display:flex;gap:8px 10px;margin-bottom:11px;flex-wrap:wrap;}"
            ".heat-leg-item{display:flex;align-items:center;gap:4px;font-size:11px;color:#6B8A78;line-height:1.2;}"
            ".heat-leg-dot{width:9px;height:9px;border-radius:2px;flex-shrink:0;}"
            ".heat-row{display:flex;align-items:center;gap:6px;margin-bottom:5px;}"
            ".heat-row-label{font-size:10px;color:#7B8D83;width:15px;flex-shrink:0;font-weight:900;}"
            ".heat-cells{display:flex;gap:1px;min-width:0;}"
            ".heat-cell{width:11px;height:11px;border-radius:3px;flex:0 0 11px;}"
        )
        _iframe_html(
            body=(
                f'<div class="heat-wrap">'
                f'<div class="heat-hd">近 4 周干预互动热力图</div>'
                f'<div class="heat-legend">{legend_html}</div>'
                f'{rows_html}'
                f'<div style="display:flex;justify-content:space-between;'
                f'font-size:10px;color:#AEBBB4;margin-top:8px;">'
                f'<span>{_esc(label_start)}</span>'
                f'<span>今天 {_esc(label_end)}</span>'
                f'</div></div>'
            ),
            css=heat_css,
            height=210,
        )
    else:
        _html("""
        <div style="background:#fff;border-radius:14px;padding:14px;
                    margin-bottom:10px;color:#9CA3AF;font-size:12px;text-align:center;">
            暂无互动记录，请先使用消息页面与贴士互动后返回查看。
        </div>
        """)

# ══════════════════════════════════════════════════════════════
# 入口函数（app.py 调用）
# ══════════════════════════════════════════════════════════════

def render_profile_page():
    """
    「我的」Tab 渲染入口。
    - 不接收任何参数，所有数据从 session_state 读取
    - 不写入任何现有 session_state 键
    - 通过 session_state["profile_subpage"] 控制主/详情子页跳转
    """
    _inject_profile_style()

    st.session_state.setdefault("profile_subpage", "main")

    dorm_id = _current_dorm()
    score   = _latest_score(dorm_id)
    progress = get_user_progress(dorm_id)
    streak  = int(progress.get("streak_days", 0))
    read_n  = _read_msg_count()

    subpage = st.session_state.get("profile_subpage", "main")

    if subpage == "detail":
        _profile_detail(dorm_id=dorm_id, score=score, streak=streak)
    else:
        _profile_main(dorm_id=dorm_id, score=score, streak=streak, read_n=read_n, progress=progress)
