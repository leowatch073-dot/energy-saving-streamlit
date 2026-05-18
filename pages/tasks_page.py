"""
tasks_page.py  v3
=================
「任务」Tab — 打卡日历 + 今日任务 + 成果页 + 知识库

布局（从上到下）：
  1. 打卡日历（近7天，深绿英雄区）
  2. 今日任务卡 + 打卡按钮；小字入口→历史任务列表
  3. 知识库轮播（翻到自动标记已读，无任何已读/未读文字）

子页面（session_state["tasks_view"]）：
  "main"    → 主页面
  "result"  → 打卡成果页
  "history" → 历史任务列表

安全写法：
  * 每个 st.markdown() 内 HTML 完整闭合
  * 无 HTML 注释（<!-- -->）
  * 知识库用 components.html iframe，JS 内部翻页
  * iframe postMessage → st.session_state["_tk_read_idx"] → mark_message_read

复用接口（不改 messages_page.py）：
  ensure_daily_messages_for_current_dorm()
  build_message_df(messages)
  mark_message_read(real_idx)
  append_interaction_log(event, message_index, msg, extra)

新增 session_state 键（不与现有键冲突）：
  tasks_view         str       "main" / "result" / "history"
  tasks_checkin_log  list      [{date, dorm_id, score, ts}, ...]
  _tk_read_idx       int|None  知识库曝光传递用，处理后立即清空
"""

from __future__ import annotations
import json
from datetime import date as _date, timedelta as _td, datetime as _dt

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from state.user_progress import get_badges, get_user_progress, record_checkin


# ══════════════════════════════════════════════════════════════
# 工具
# ══════════════════════════════════════════════════════════════

def _esc(s) -> str:
    return (str(s)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;"))


def _today_str() -> str:
    return str(st.session_state.get(
        "business_date_str", _dt.now().strftime("%Y-%m-%d")
    ))


def _dorm_id() -> str:
    return str(st.session_state.get("current_dorm_for_messages", "")).strip()


# ── 打卡日历：从 tasks_checkin_log 读取 ───────────────────────
def _checkin_dates() -> set:
    """返回当前宿舍已打卡的日期字符串集合（YYYY-MM-DD）。"""
    progress = get_user_progress(_dorm_id(), _today_str())
    return set(progress.get("checkin_dates", []))


def _already_checked_today() -> bool:
    progress = get_user_progress(_dorm_id(), _today_str())
    return bool(progress.get("today_checked_in", False))


# ── 连续天数（从 tasks_checkin_log 计算） ─────────────────────
def _streak() -> int:
    progress = get_user_progress(_dorm_id(), _today_str())
    return int(progress.get("streak_days", 0))


# ── 今日 tip（今天的 tip channel 消息，兜底取最新3条） ─────────
def _today_tips(tip_df: pd.DataFrame) -> pd.DataFrame:
    ts = _today_str()
    if tip_df.empty or "ts" not in tip_df.columns:
        return tip_df.head(3).copy().reset_index(drop=True)
    mask = tip_df["ts"].astype(str).str.startswith(ts)
    result = tip_df[mask].copy()
    if result.empty:
        result = tip_df.head(3).copy()
    return result.reset_index(drop=True)


# ── 写入今日打卡记录 ──────────────────────────────────────────
def _write_checkin():
    """向 tasks_checkin_log 写入今日打卡，同时写 interaction_logs。"""
    dorm = _dorm_id()
    today = _today_str()
    record_checkin(dorm, today)
    # 同时写入 interaction_logs，保持与现有分析链路兼容
    st.session_state.setdefault("interaction_logs", [])
    st.session_state["interaction_logs"].append({
        "ts":      _dt.now().strftime("%Y-%m-%d %H:%M:%S"),
        "date":    today,
        "event":   "tasks_checkin",
        "page":    "tasks",
        "dorm_id": dorm,
    })


_KB_ICONS = ["🌡️", "🌿", "🪟", "💡", "❄️", "🌬️", "☀️", "⚡"]
_KB_BG    = ["#FFF8DC", "#EAF7EF", "#EFF6FF", "#FFFBEB",
             "#EFF6FF", "#F0F9FF", "#FFF8DC", "#FEF3C7"]


# ══════════════════════════════════════════════════════════════
# 样式
# ══════════════════════════════════════════════════════════════

def _inject_style():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;700;900&display=swap');

.tk-hero {
    background: linear-gradient(145deg, #052E16 0%, #1C6E3D 100%);
    margin: 0 0 10px;
    padding: 14px 16px 15px;
    font-family: 'Noto Sans SC', sans-serif;
    border-radius: 20px;
}
.tk-hero-row1 {
    display: flex; justify-content: space-between;
    align-items: flex-start; margin-bottom: 11px;
}
.tk-title  { font-size: 22px; font-weight: 900; color: #fff; letter-spacing: 0; }
.tk-date   { font-size: 12px; color: rgba(255,255,255,.58); margin-top: 3px; }
.tk-streak-box {
    background: rgba(255,255,255,.12); border-radius: 12px;
    padding: 6px 11px; border: 1px solid rgba(255,255,255,.18);
    text-align: center; flex-shrink: 0;
}
.tk-streak-n { font-size: 20px; font-weight: 900; color: #FBBF24; line-height: 1; }
.tk-streak-l { font-size: 10px; color: rgba(255,255,255,.62); margin-top: 2px; }

.tk-cal {
    display: flex; gap: 0;
    justify-content: space-between;
}
.tk-cal-day {
    display: flex; flex-direction: column;
    align-items: center; gap: 4px; flex: 1;
}
.tk-cal-lbl { font-size: 10px; color: rgba(255,255,255,.58); font-weight: 700; }
.tk-cal-dot {
    width: 26px; height: 26px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 10px; font-weight: 900;
}
.tcd-done   { background: #4ADE80; color: #052E16; }
.tcd-miss   { background: rgba(255,255,255,.1); color: rgba(255,255,255,.25); }
.tcd-today  { background: rgba(255,255,255,.22); color: #fff; border: 2px solid rgba(255,255,255,.7); }
.tcd-future { border: 1px dashed rgba(255,255,255,.15); color: rgba(255,255,255,.15); }
.tcd-done-today { background: #4ADE80; color: #052E16; box-shadow: 0 0 0 2.5px rgba(255,255,255,.8); }

.tk-sep {
    display: flex; align-items: center; gap: 8px;
    padding: 9px 0 6px; font-family: 'Noto Sans SC', sans-serif;
}
.tk-sep-badge {
    font-size: 11px; font-weight: 800; padding: 4px 11px;
    border-radius: 99px; white-space: nowrap; flex-shrink: 0; letter-spacing: .4px;
}
.tk-sep-line  { flex: 1; height: 1px; background: #DCE8DF; }
.tk-sep-sub   { font-size: 11px; color: #87968F; white-space: nowrap; }

.tk-task-card {
    background: #fff; border-radius: 18px;
    padding: 14px 15px; margin-bottom: 8px;
    font-family: 'Noto Sans SC', sans-serif;
    border: 1px solid #E6EDE8;
    box-shadow: 0 8px 22px rgba(16,24,18,.06);
}
.tk-task-card.checked {
    opacity: .72; border-color: #D1FAE5; box-shadow: none;
}
.tk-task-hd {
    display: flex; justify-content: space-between;
    align-items: center; margin-bottom: 8px;
}
.tk-task-badge {
    font-size: 11px; font-weight: 800; border-radius: 99px; padding: 4px 10px;
}
.tk-task-status {
    width: 26px; height: 26px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 12px; flex-shrink: 0;
}
.tts-done { background: #D1FAE5; color: #065F46; }
.tts-todo { background: #F0F8FF; border: 2px solid #3B82F6; color: #3B82F6; font-size: 9px; font-weight: 900; }
.tk-task-title { font-size: 16px; font-weight: 900; color: #1A2820; margin-bottom: 6px; line-height: 1.35; }
.tk-task-title.done { text-decoration: line-through; color: #9CA3AF; }
.tk-task-body  { font-size: 13px; color: #536B5D; line-height: 1.65; }
.tk-task-body.done { color: #C8D3CC; }
.tk-task-ft {
    display: flex; justify-content: space-between; align-items: center;
    margin-top: 10px; padding-top: 9px; border-top: .5px solid #F0F4F1; gap: 8px;
}
.tk-task-arm  { font-size: 10px; color: #B6C2BC; min-width: 0; overflow-wrap: anywhere; }
.tk-task-xp   { font-size: 11px; font-weight: 800; color: #D97706; background: #FFF8DC; border-radius: 99px; padding: 3px 9px; flex-shrink: 0; }
.tk-task-done-chip { font-size: 11px; color: #8DA098; flex-shrink: 0; }
.tk-action-row {
    margin: 6px 0 10px;
}
.tk-action-row [data-testid="stHorizontalBlock"] {
    flex-wrap: nowrap !important;
    gap: 8px !important;
}
.tk-action-row [data-testid="stButton"] > button {
    min-height: 38px !important;
    border-radius: 999px !important;
    font-size: 13px !important;
    padding: 7px 12px !important;
    box-shadow: 0 6px 14px rgba(16,24,18,.06) !important;
}
.tk-action-row [data-testid="stButton"] > button[kind="secondary"] {
    background: #FFFFFF !important;
    border-color: #DCE8DF !important;
    color: #315A42 !important;
}

.tk-result-hero {
    background: linear-gradient(145deg, #052E16 0%, #1C6E3D 100%);
    margin: 0; padding: 18px 16px 16px;
    text-align: center; font-family: 'Noto Sans SC', sans-serif;
}
.tk-result-emoji { font-size: 32px; margin-bottom: 8px; }
.tk-result-title { font-size: 18px; font-weight: 900; color: #fff; margin-bottom: 5px; }
.tk-result-sub   { font-size: 11px; color: rgba(255,255,255,.6); margin-bottom: 14px; }
.tk-result-stats {
    display: flex; justify-content: center; gap: 10px;
}
.tk-rs-box {
    background: rgba(255,255,255,.12); border-radius: 12px;
    padding: 8px 14px; text-align: center; min-width: 56px;
}
.tk-rs-val { font-size: 18px; font-weight: 900; line-height: 1; }
.tk-rs-lbl { font-size: 8px; color: rgba(255,255,255,.55); margin-top: 2px; }

.tk-ach-grid {
    display: grid; grid-template-columns: repeat(4, 1fr);
    gap: 8px; font-family: 'Noto Sans SC', sans-serif;
}
.tk-ach-item { display: flex; flex-direction: column; align-items: center; gap: 4px; }
.tk-ach-ic   { width: 44px; height: 44px; border-radius: 14px; display: flex; align-items: center; justify-content: center; font-size: 20px; }
.tk-ach-nm   { font-size: 8.5px; font-weight: 700; color: #3A5A46; text-align: center; }
.tk-ach-lock { opacity: .28; }

.tk-hist-card {
    background: #fff; border-radius: 13px;
    padding: 11px 13px; margin-bottom: 6px;
    display: flex; align-items: center; gap: 10px;
    font-family: 'Noto Sans SC', sans-serif;
}
.tk-hist-dot { width: 28px; height: 28px; border-radius: 50%; background: #D1FAE5; color: #065F46; font-size: 11px; display: flex; align-items: center; justify-content: center; flex-shrink: 0; font-weight: 900; }
.tk-hist-date { font-size: 12px; font-weight: 800; color: #1A2820; }
.tk-hist-sub  { font-size: 10px; color: #9CA3AF; margin-top: 2px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# 打卡日历（英雄区）
# ══════════════════════════════════════════════════════════════

def _render_calendar():
    done_dates = _checkin_dates()
    streak     = _streak()
    today      = _today_str()
    today_d    = pd.to_datetime(today).date()
    checked_today = _already_checked_today()

    day_labels = ["一", "二", "三", "四", "五", "六", "日"]
    days_html = ""
    for delta in range(6, -1, -1):
        d      = today_d - _td(days=delta)
        dstr   = d.strftime("%Y-%m-%d")
        dlbl   = day_labels[d.weekday()]
        is_today = (dstr == today)

        if is_today and checked_today:
            dot_cls = "tcd-done-today"
            inner   = "✓"
        elif is_today:
            dot_cls = "tcd-today"
            inner   = "今"
        elif dstr in done_dates:
            dot_cls = "tcd-done"
            inner   = "✓"
        elif dstr < today:
            dot_cls = "tcd-miss"
            inner   = "·"
        else:
            dot_cls = "tcd-future"
            inner   = str(d.day)

        lbl_style = "color:rgba(255,255,255,.9);font-weight:900;" if is_today else ""
        days_html += f"""
<div class="tk-cal-day">
  <div class="tk-cal-lbl" style="{lbl_style}">{dlbl}</div>
  <div class="tk-cal-dot {dot_cls}">{inner}</div>
</div>"""

    streak_disp = max(streak, 1) if done_dates else 0

    st.markdown(f"""
<div class="tk-hero">
  <div class="tk-hero-row1">
    <div>
      <div class="tk-title">任务</div>
      <div class="tk-date">{_esc(today)} · 打卡日历</div>
    </div>
    <div class="tk-streak-box">
      <div class="tk-streak-n">🔥{streak_disp}</div>
      <div class="tk-streak-l">连续天数</div>
    </div>
  </div>
  <div class="tk-cal">{days_html}</div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# 今日任务区
# ══════════════════════════════════════════════════════════════

def _render_today_task(today_df: pd.DataFrame):
    checked = _already_checked_today()

    st.markdown(f"""
<div class="tk-sep">
  <div class="tk-sep-badge" style="background:#D1FAE5;color:#065F46;">📋 今日任务</div>
  <div class="tk-sep-line"></div>
  <span class="tk-sep-sub">{"已完成" if checked else "待打卡"}</span>
</div>
""", unsafe_allow_html=True)

    if today_df.empty:
        st.markdown("""
<div style="background:#fff;border-radius:16px;padding:20px 16px;
            text-align:center;color:#B0BAB5;font-size:12px;margin-bottom:10px;
            font-family:'Noto Sans SC',sans-serif;">
  今日暂无任务，请先在首页生成当日评分
</div>
""", unsafe_allow_html=True)
        return

    # 只展示第一条今日任务
    row     = today_df.iloc[0]
    title   = _esc(str(row.get("title", "")))
    body    = _esc(str(row.get("body", "")))
    arm_id  = _esc(str(row.get("arm_id", "")))
    stype   = str(row.get("strategy_type", "P")).upper()
    sname   = {"P": "信息提示", "R": "定期提醒",
               "F": "数据反馈", "S": "社会比较"}.get(stype, stype)

    if checked:
        card_extra = "checked"
        status_cls = "tts-done"
        status_icon = "✓"
        title_cls   = "done"
        body_cls    = "done"
        badge_sty   = "background:#D1FAE5;color:#065F46;"
        foot_right  = '<span class="tk-task-done-chip">今日已打卡</span>'
    else:
        card_extra = ""
        status_cls = "tts-todo"
        status_icon = "!"
        title_cls   = ""
        body_cls    = ""
        badge_sty   = "background:#EFF6FF;color:#1D4ED8;"
        foot_right  = '<span class="tk-task-xp">完成得 +10 XP</span>'

    st.markdown(f"""
<div class="tk-task-card {card_extra}">
  <div class="tk-task-hd">
    <span class="tk-task-badge" style="{badge_sty}">{_esc(stype)} · {_esc(sname)}</span>
    <div class="tk-task-status {status_cls}">{status_icon}</div>
  </div>
  <div class="tk-task-title {title_cls}">{title}</div>
  <div class="tk-task-body {body_cls}">{body[:100]}{"…" if len(str(row.get("body",""))) > 100 else ""}</div>
  <div class="tk-task-ft">
    <span class="tk-task-arm">{arm_id[:24]}</span>
    {foot_right}
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown('<div class="tk-action-row">', unsafe_allow_html=True)
    primary_col, history_col = st.columns([1.15, 0.85])
    with primary_col:
        if checked:
            if st.button("今日成果", key="tk_to_result", use_container_width=True, type="primary"):
                st.session_state["tasks_view"] = "result"
                st.rerun()
        else:
            if st.button("完成打卡", key="tk_checkin_btn", use_container_width=True, type="primary"):
                _write_checkin()
                st.session_state["tasks_view"] = "result"
                st.rerun()
    with history_col:
        if st.button("历史记录", key="tk_to_hist", use_container_width=True):
            st.session_state["tasks_view"] = "history"
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# 知识库轮播（iframe，翻到自动已读，无已读/未读文字）
# ══════════════════════════════════════════════════════════════

def _render_kb(all_tip_df: pd.DataFrame, today_df: pd.DataFrame):
    today = _today_str()[:10]
    if all_tip_df.empty or "ts" not in all_tip_df.columns:
        kb_df = all_tip_df.iloc[0:0].copy()
    else:
        kb_df = all_tip_df[all_tip_df["ts"].astype(str).str.startswith(today)].copy()

    total_n = len(kb_df)

    st.markdown("""
<div class="tk-sep" style="padding-top:10px;">
  <div class="tk-sep-badge" style="background:#EFF6FF;color:#1D4ED8;">📚 知识库</div>
  <div class="tk-sep-line"></div>
</div>
""", unsafe_allow_html=True)

    if kb_df.empty:
        st.markdown(f"""
<div style="padding:20px;text-align:center;color:#B0BAB5;font-size:12px;
            font-family:'Noto Sans SC',sans-serif;">{_esc(today)} 暂无知识内容</div>
""", unsafe_allow_html=True)
        return

    # 处理知识库曝光事件（postMessage 传来的 real_idx）
    _process_kb_exposure(kb_df)

    # 构建卡片数据，包含 real_idx 供 JS postMessage 回传
    cards_data = []
    for i, (_, row) in enumerate(kb_df.iterrows()):
        cards_data.append({
            "real_idx": int(row["message_index"]),
            "icon":     _KB_ICONS[i % len(_KB_ICONS)],
            "ibg":      _KB_BG[i % len(_KB_BG)],
            "title":    str(row.get("title", "")),
            "body":     str(row.get("body", ""))[:220],
            "arm":      str(row.get("arm_id", ""))[:20],
            "ts":       str(row.get("ts", ""))[:10],
        })

    cards_json = json.dumps(cards_data, ensure_ascii=False)

    # iframe 高度给足卡片正文空间，避免移动端知识卡片被裁切。
    card_h  = 208
    nav_h   = 36
    total_h = card_h + nav_h

    iframe_html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
*{{box-sizing:border-box;margin:0;padding:0;}}
html,body{{background:transparent;font-family:'Noto Sans SC',-apple-system,sans-serif;overflow:hidden;}}
.root{{display:flex;flex-direction:column;height:100%;padding:0;}}
.track-wrap{{flex:1;overflow:hidden;border-radius:14px;}}
.track{{display:flex;transition:transform .32s cubic-bezier(.4,0,.2,1);height:100%;}}
.slide{{min-width:100%;height:100%;}}
.card{{background:#fff;border-radius:14px;padding:14px 15px;height:100%;display:flex;flex-direction:column;gap:7px;border:1px solid #EEF2EE;}}
.ic{{width:34px;height:34px;border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:18px;}}
.title{{font-size:16px;font-weight:800;color:#1A2820;line-height:1.35;}}
.body{{font-size:13px;color:#536B5D;line-height:1.62;flex:1;min-height:0;overflow:auto;}}
.foot{{display:flex;justify-content:space-between;gap:8px;padding-top:7px;border-top:1px solid #F0F4F1;margin-top:auto;}}
.arm{{font-size:10px;color:#AEBBB4;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}}
.ts{{font-size:10px;color:#AEBBB4;white-space:nowrap;}}
.nav{{display:flex;align-items:center;justify-content:space-between;padding:5px 1px 0;}}
.dots{{display:flex;gap:4px;align-items:center;}}
.dot{{width:6px;height:6px;border-radius:50%;background:#D1D9D5;transition:all .2s;}}
.dot.on{{width:18px;border-radius:99px;background:#1C6E3D;}}
.counter{{font-size:12px;color:#87968F;font-weight:700;}}
.btns{{display:flex;gap:6px;}}
.btn{{width:28px;height:28px;border-radius:50%;border:1px solid #E5E7EB;background:#fff;
      font-size:14px;cursor:pointer;display:flex;align-items:center;justify-content:center;color:#374151;}}
.btn:hover{{background:#F0FDF4;border-color:#86EFAC;color:#1C6E3D;}}
.btn:disabled{{opacity:.3;cursor:default;}}
</style>
</head>
<body>
<div class="root">
  <div class="track-wrap"><div class="track" id="track"></div></div>
  <div class="nav">
    <div style="display:flex;align-items:center;gap:8px;">
      <div class="dots" id="dots"></div>
      <span class="counter" id="counter"></span>
    </div>
    <div class="btns">
      <button class="btn" id="prev" onclick="move(-1)">&#8592;</button>
      <button class="btn" id="next" onclick="move(1)">&#8594;</button>
    </div>
  </div>
</div>
<script>
const cards={cards_json};
const n=cards.length;
let cur=0;
function buildSlide(c){{
  return '<div class="slide"><div class="card">'
    +'<div class="ic" style="background:'+c.ibg+';">'+c.icon+'</div>'
    +'<div class="title">'+c.title+'</div>'
    +'<div class="body">'+c.body+'</div>'
    +'<div class="foot"><span class="arm">'+c.arm+'</span><span class="ts">'+c.ts+'</span></div>'
    +'</div></div>';
}}
function notifyRead(idx){{
  try{{window.parent.postMessage({{type:'tk_read',real_idx:idx}},'*');}}catch(e){{}}
}}
function render(){{
  document.getElementById('track').innerHTML=cards.map(buildSlide).join('');
  document.getElementById('track').style.transform='translateX(-'+cur*100+'%)';
  const maxD=Math.min(n,8);
  document.getElementById('dots').innerHTML=Array.from({{length:maxD}},(_,i)=>{{
    const on=(n<=maxD?i===cur:i===Math.floor(cur/n*maxD));
    return '<div class="dot'+(on?' on':'')+'"></div>';
  }}).join('');
  document.getElementById('counter').textContent=(cur+1)+' / '+n;
  document.getElementById('prev').disabled=(cur===0);
  document.getElementById('next').disabled=(cur===n-1);
  notifyRead(cards[cur].real_idx);
}}
function move(d){{
  const nx=cur+d;
  if(nx<0||nx>=n)return;
  cur=nx;
  document.getElementById('track').style.transform='translateX(-'+cur*100+'%)';
  render();
}}
render();
</script>
</body>
</html>"""

    components.html(iframe_html, height=total_h, scrolling=False)


def _process_kb_exposure(kb_df: pd.DataFrame | None = None):
    """
    处理知识库卡片曝光：
    JS postMessage 无法直接触发 Python，改用 st.session_state["_tk_read_idx"]
    通过 URL query_params 传递 real_idx，在下次 rerun 时处理。
    实际上 Streamlit Community Cloud 里 postMessage 跨 iframe 通常被沙箱阻止，
    所以这里采用更可靠的方案：知识库卡片翻到即在 iframe 内标记，
    不依赖 postMessage，而是在用户下次交互时批量处理（标记最近看过的卡片）。
    简化版：直接把 kb 中所有卡片标记为已读（用户打开知识库即视为阅读）。
    """
    from pages.messages_page import mark_message_read, append_interaction_log
    messages = st.session_state.get("messages", [])
    today = _today_str()[:10]
    mark_key = f"_tk_kb_auto_read_done_{today}"
    if st.session_state.get(mark_key):
        return
    if kb_df is None or kb_df.empty or "message_index" not in kb_df.columns:
        return
    visible_idx = set()
    for value in kb_df["message_index"].tolist():
        try:
            visible_idx.add(int(value))
        except Exception:
            continue
    if not visible_idx:
        return
    # 第一次打开当天知识库时，仅将当前展示的 tip 标记为已读（静默、无视觉提示）
    for i, msg in enumerate(messages):
        if i not in visible_idx:
            continue
        if isinstance(msg, dict):
            ch = str(msg.get("channel", "")).strip().lower()
            if ch == "tip" and not bool(msg.get("read", False)):
                mark_message_read(i)
                append_interaction_log(
                    event="message_view", message_index=i, msg=msg,
                    extra={"section": "kb_auto"},
                )
    st.session_state[mark_key] = True


# ══════════════════════════════════════════════════════════════
# 打卡成果页
# ══════════════════════════════════════════════════════════════

def _render_result(tip_df: pd.DataFrame):
    today    = _today_str()
    progress = get_user_progress(_dorm_id(), today)
    streak   = int(progress.get("streak_days", 0))
    earned_xp = int(progress.get("earned_xp", 0))
    xp_total = int(progress.get("total_xp", 0))
    done_dates = _checkin_dates()
    today_d  = pd.to_datetime(today).date()

    # 返回按钮
    if st.button("← 返回任务", key="tk_result_back"):
        st.session_state["tasks_view"] = "main"
        st.rerun()

    # 庆祝英雄卡
    st.markdown(f"""
<div class="tk-result-hero">
  <div class="tk-result-emoji">🎉</div>
  <div class="tk-result-title">今日打卡完成！</div>
  <div class="tk-result-sub">{_esc(today)} · 节能知识已掌握</div>
  <div class="tk-result-stats">
    <div class="tk-rs-box">
      <div class="tk-rs-val" style="color:#FBBF24;">{streak}</div>
      <div class="tk-rs-lbl">连续天数</div>
    </div>
    <div class="tk-rs-box">
      <div class="tk-rs-val" style="color:#4ADE80;">+{earned_xp}</div>
      <div class="tk-rs-lbl">获得 XP</div>
    </div>
    <div class="tk-rs-box">
      <div class="tk-rs-val" style="color:#60A5FA;">{xp_total}</div>
      <div class="tk-rs-lbl">总 XP</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)

    # 本周日历（白底卡片）
    day_labels = ["一", "二", "三", "四", "五", "六", "日"]
    days_html = ""
    for delta in range(6, -1, -1):
        d      = today_d - _td(days=delta)
        dstr   = d.strftime("%Y-%m-%d")
        dlbl   = day_labels[d.weekday()]
        is_today = (dstr == today)

        if dstr in done_dates:
            bg = "#4ADE80"; color = "#052E16"; icon = "✓"
            ring = "box-shadow:0 0 0 2.5px #fff,0 0 0 4px #1C6E3D;" if is_today else ""
        elif dstr < today:
            bg = "#FECACA"; color = "#991B1B"; icon = "✕"; ring = ""
        else:
            bg = "#F3F4F6"; color = "#D1D5DB"; icon = str(d.day); ring = ""

        lbl_style = "color:#1C6E3D;font-weight:900;" if is_today else "color:#9CA3AF;"
        days_html += f"""
<div style="display:flex;flex-direction:column;align-items:center;gap:4px;">
  <div style="font-size:8px;font-weight:700;{lbl_style}">{dlbl}</div>
  <div style="width:26px;height:26px;border-radius:50%;background:{bg};color:{color};
              font-size:10px;font-weight:900;display:flex;align-items:center;
              justify-content:center;{ring}">{icon}</div>
</div>"""

    st.markdown(f"""
<div style="background:#fff;border-radius:16px;padding:13px 15px;margin-bottom:10px;
            font-family:'Noto Sans SC',sans-serif;">
  <div style="font-size:12px;font-weight:900;color:#0D2B18;margin-bottom:10px;">本周打卡</div>
  <div style="display:flex;justify-content:space-between;">{days_html}</div>
</div>
""", unsafe_allow_html=True)

    # 成就徽章
    ach_items = ""
    for badge in get_badges(progress):
        lock_cls = "" if badge.get("unlocked") else "tk-ach-lock"
        ach_items += f"""
<div class="tk-ach-item {lock_cls}">
  <div class="tk-ach-ic" style="background:{badge["bg"]};">{badge["icon"]}</div>
  <div class="tk-ach-nm">{_esc(badge["name"])}</div>
</div>"""

    st.markdown(f"""
<div style="background:#fff;border-radius:16px;padding:13px 15px;margin-bottom:10px;
            font-family:'Noto Sans SC',sans-serif;">
  <div style="font-size:12px;font-weight:900;color:#0D2B18;margin-bottom:10px;">成就徽章</div>
  <div class="tk-ach-grid">{ach_items}</div>
</div>
""", unsafe_allow_html=True)

    # XP 进度条
    xp_mod  = xp_total % 100
    xp_next = 100 - xp_mod
    xp_pct  = xp_mod

    st.markdown(f"""
<div style="background:#fff;border-radius:16px;padding:13px 15px;margin-bottom:12px;
            font-family:'Noto Sans SC',sans-serif;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
    <div style="font-size:12px;font-weight:900;color:#0D2B18;">经验值</div>
    <div style="font-size:11px;color:#D97706;font-weight:800;">{xp_total} XP</div>
  </div>
  <div style="height:6px;background:#EEF2EE;border-radius:99px;overflow:hidden;margin-bottom:6px;">
    <div style="width:{xp_pct}%;height:100%;background:linear-gradient(90deg,#4ADE80,#22D3EE);border-radius:99px;"></div>
  </div>
  <div style="font-size:10px;color:#9CA3AF;">距下一个徽章还差 {xp_next} XP</div>
</div>
""", unsafe_allow_html=True)

    st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# 历史打卡记录页
# ══════════════════════════════════════════════════════════════

def _render_history():
    if st.button("← 返回任务", key="tk_hist_back"):
        st.session_state["tasks_view"] = "main"
        st.rerun()

    st.markdown("""
<div style="padding:10px 0 6px;font-size:18px;font-weight:900;color:#0D2B18;
            font-family:'Noto Sans SC',sans-serif;">历史打卡记录</div>
""", unsafe_allow_html=True)

    done_dates = sorted(_checkin_dates(), reverse=True)
    logs       = st.session_state.get("tasks_checkin_log", [])
    dorm       = _dorm_id()

    if not done_dates:
        st.markdown("""
<div style="padding:40px 0;text-align:center;color:#B0BAB5;font-size:13px;
            font-family:'Noto Sans SC',sans-serif;">暂无打卡记录</div>
""", unsafe_allow_html=True)
        return

    for ds in done_dates:
        d_obj = pd.to_datetime(ds).date()
        weekday_map = ["周一","周二","周三","周四","周五","周六","周日"]
        wday = weekday_map[d_obj.weekday()]
        month_day = f"{d_obj.month}月{d_obj.day}日"

        st.markdown(f"""
<div class="tk-hist-card">
  <div class="tk-hist-dot">✓</div>
  <div>
    <div class="tk-hist-date">{_esc(month_day)} {_esc(wday)}</div>
    <div class="tk-hist-sub">{_esc(ds)} · 打卡成功</div>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# 主页面
# ══════════════════════════════════════════════════════════════

def _tasks_main(tip_df: pd.DataFrame):
    today_df = _today_tips(tip_df)

    _render_calendar()
    _render_today_task(today_df)
    _render_kb(tip_df, today_df)


# ══════════════════════════════════════════════════════════════
# 入口（app.py 调用）
# ══════════════════════════════════════════════════════════════

def render_tasks_page():
    from pages.messages_page import (
        ensure_daily_messages_for_current_dorm,
        build_message_df,
    )

    _inject_style()

    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("interaction_logs", [])
    st.session_state.setdefault("intervention_logs", [])
    st.session_state.setdefault("tasks_checkin_log", [])
    st.session_state.setdefault("tasks_view", "main")
    st.session_state.setdefault("_tk_kb_auto_read_done", False)

    generated_n, gen_msg = ensure_daily_messages_for_current_dorm()
    if generated_n > 0:
        st.toast(gen_msg, icon="✅")

    messages = st.session_state.get("messages", [])
    if not messages:
        st.markdown("""
<div style="padding:60px 20px;text-align:center;color:#B0BAB5;
            font-size:13px;font-family:'Noto Sans SC',sans-serif;line-height:1.8;">
  暂无任务数据<br>请先在「首页」选择宿舍并生成当日评分
</div>
""", unsafe_allow_html=True)
        return

    msg_df = build_message_df(messages)
    if msg_df.empty:
        st.markdown('<div style="padding:40px;text-align:center;color:#B0BAB5;font-size:13px;">暂无消息数据。</div>',
                    unsafe_allow_html=True)
        return

    tip_df = msg_df[msg_df["channel"] == "tip"].copy().reset_index(drop=True)
    if tip_df.empty:
        st.markdown("""
<div style="padding:60px 20px;text-align:center;color:#B0BAB5;
            font-size:13px;font-family:'Noto Sans SC',sans-serif;line-height:1.8;">
  暂无 P 类贴士<br>请先在「首页」完成宿舍选择与评分生成
</div>
""", unsafe_allow_html=True)
        return

    view = st.session_state.get("tasks_view", "main")

    if view == "result":
        _render_result(tip_df)
    elif view == "history":
        _render_history()
    else:
        _tasks_main(tip_df)
