import pandas as pd
import streamlit as st
import numpy as np
import json
import altair as alt

from config import DORM_CS_PROFILE, DEFAULT_CLUSTER_ARMS
from models.scoring import (
    compute_BS_energy_week,
    classify_cluster,
)
from models.bandit import (
    parse_state_vector,
    linucb_choose_arm,
    ensure_arm,
)
from utils.data_utils import get_dorm_hourly, week_start
from strategy_library import (
    build_message_from_arm,
    make_message_key,
    get_cluster_message_plan,
)

# ===================== 样式 =====================

def _inject_home_style(mobile: bool = False):
    layout_css = (
        """
        .block-container {
            padding-top: 1.0rem !important;
            padding-bottom: 2.2rem !important;
            max-width: 100% !important;
        }
        """
        if not mobile else
        """
        .st-key-home_mobile_cards {
            width: 100% !important;
            max-width: 100% !important;
            box-sizing: border-box !important;
            overflow: hidden !important;
        }
        .st-key-home_demo_control_bar {
            margin: 0 0 8px !important;
            padding: 0 !important;
            width: 100% !important;
            max-width: 100% !important;
            box-sizing: border-box !important;
            overflow: visible !important;
        }
        .st-key-home_demo_control_bar [data-testid="stHorizontalBlock"] {
            display: grid !important;
            grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
            gap: 8px !important;
            flex-wrap: nowrap !important;
            width: 100% !important;
            min-width: 0 !important;
            max-width: 100% !important;
            margin: 0 !important;
            padding: 0 !important;
            overflow: visible !important;
        }
        .st-key-home_demo_control_bar [data-testid="column"],
        .st-key-home_demo_control_bar [data-testid="stColumn"] {
            width: 100% !important;
            min-width: 0 !important;
            max-width: none !important;
            flex: initial !important;
            padding: 0 !important;
        }
        .st-key-home_demo_control_bar [data-testid="stElementContainer"] {
            width: 100% !important;
            min-width: 0 !important;
            max-width: 100% !important;
            margin: 0 !important;
            padding: 0 !important;
            overflow: visible !important;
        }
        .st-key-home_demo_control_bar [data-testid="stSelectbox"] {
            width: 100% !important;
            min-width: 0 !important;
            max-width: 100% !important;
            margin: 0 !important;
        }
        .st-key-home_demo_control_bar [data-testid="stSelectbox"] > div {
            min-height: 36px !important;
            width: 100% !important;
            min-width: 0 !important;
        }
        .st-key-home_demo_control_bar [data-baseweb="select"] > div {
            min-height: 36px !important;
            border-radius: 13px !important;
            border-color: #DCE8DF !important;
            background: rgba(255,255,255,0.78) !important;
            box-shadow: none !important;
            font-size: 12px !important;
            font-weight: 800 !important;
        }
        .st-key-home_demo_control_bar [data-baseweb="select"] {
            width: 100% !important;
            min-width: 0 !important;
        }
        .st-key-home_mobile_cards [data-testid="stVerticalBlock"] {
            width: 100% !important;
            max-width: 100% !important;
            box-sizing: border-box !important;
        }
        .st-key-home_mobile_cards [data-testid="stElementContainer"] {
            width: 100% !important;
            max-width: 100% !important;
            margin: 0 auto 10px !important;
            overflow: hidden !important;
            box-sizing: border-box !important;
        }
        .st-key-home_mobile_cards [data-testid="stIFrame"] {
            width: 100% !important;
            max-width: 100% !important;
            box-sizing: border-box !important;
            overflow: hidden !important;
        }
        .st-key-home_mobile_cards iframe {
            display: block !important;
            width: 100% !important;
            max-width: 100% !important;
            margin: 0 !important;
            border-radius: 18px !important;
            box-sizing: border-box !important;
        }
        """
    )
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter+Tight:wght@300;400;500;600;700;800&display=swap');

        /* ══ 全局 ══ */
        __HOME_LAYOUT_CSS__
        .stApp { background: #F2F4F3; }

        /* ══ 通用右栏容器：改为 grid，更稳 ══ */
        .side-stack {
            display: grid;
            grid-template-rows: auto minmax(0, 1fr);
            gap: 16px;
            height: 100%;
            min-height: 0;
        }
        .side-info-card {
            min-height: 0;
            background: #FFFFFF;
            border: 1px solid #E6EDE8;
            border-radius: 24px;
            padding: 26px 28px;
            display: grid;
            align-content: start;
            row-gap: 10px;
        }

        /* ── 群体识别卡 ── */
        .c-tag {
            font-size: 10px;
            font-weight: 600;
            color: #15803D;
            background: #F0FDF4;
            border-radius: 5px;
            padding: 3px 8px;
            letter-spacing: .3px;
            display: inline-block;
            margin-bottom: 12px;
        }
        .c-type {
            font-size: 11px;
            font-weight: 500;
            color: #9CA3AF;
            margin-bottom: 5px;
            letter-spacing: .2px;
        }
        .cluster-title {
            font-family: 'Inter Tight', sans-serif;
            font-size: 22px;
            font-weight: 800;
            color: #0A1A0F;
            line-height: 1.15;
        }

        /* ── CS/BS 评分卡 ── */
        .score-row-big {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-bottom: 16px;
        }
        .score-mini-card {
            border-radius: 14px;
            padding: 15px 16px;
        }
        .score-mini-card.cs {
            background: #F5F8FF;
            border: 1px solid #DBEAFE;
        }
        .score-mini-card.bs {
            background: #F5FFF8;
            border: 1px solid #D1FAE5;
        }
        .score-mini-label {
            font-size: 10px;
            font-weight: 600;
            letter-spacing: .4px;
            margin-bottom: 8px;
        }
        .score-mini-label.cs { color: #3B82F6; }
        .score-mini-label.bs { color: #22C55E; }

        .score-mini-value {
            font-family: 'Inter Tight', sans-serif;
            font-size: 38px;
            font-weight: 800;
            line-height: 1;
            letter-spacing: -2px;
        }
        .score-mini-value.cs { color: #2563EB; }
        .score-mini-value.bs { color: #16A34A; }

        .score-bar-wrap { margin-bottom: 12px; }
        .score-bar-row {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 7px;
        }
        .score-bar-key {
            width: 20px;
            flex-shrink: 0;
            font-size: 10px;
            font-weight: 700;
        }
        .score-bar-key.cs { color: #3B82F6; }
        .score-bar-key.bs { color: #22C55E; }

        .score-bar-track {
            flex: 1;
            height: 4px;
            background: #EEF2EE;
            border-radius: 999px;
            overflow: hidden;
        }
        .score-bar-fill-cs {
            height: 100%;
            border-radius: 999px;
            background: #3B82F6;
        }
        .score-bar-fill-bs {
            height: 100%;
            border-radius: 999px;
            background: #22C55E;
        }
        .gap-chip {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            width: 100%;
            border-radius: 8px;
            padding: 8px 12px;
            font-size: 11px;
            font-weight: 600;
            box-sizing: border-box;
        }

        /* ══ 三指标横排 ══ */
        .trend-strip {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;
            margin-top: 20px;
            margin-bottom: 0;
        }
        .trend-item {
            background: #FFFFFF;
            border: 1px solid #E6EDE8;
            border-radius: 20px;
            padding: 20px 22px;
        }
        .trend-lbl {
            font-size: 10px;
            font-weight: 600;
            color: #9CA3AF;
            letter-spacing: .6px;
            text-transform: uppercase;
            margin-bottom: 10px;
        }
        .trend-val {
            font-family: 'Inter Tight', sans-serif;
            font-size: 26px;
            font-weight: 800;
            color: #0A1A0F;
            line-height: 1;
            letter-spacing: -1px;
            margin-bottom: 4px;
        }
        .trend-val.pos { color: #16A34A; }
        .trend-val.neg { color: #D97706; }

        .trend-unit {
            font-size: 12px;
            font-weight: 400;
            color: #9CA3AF;
            margin-left: 3px;
            letter-spacing: 0;
        }
        .trend-change {
            font-size: 11px;
            font-weight: 500;
            color: #9CA3AF;
        }
        .trend-change .up { color: #16A34A; font-weight: 700; }
        .trend-change .dn { color: #D97706; font-weight: 700; }
        .trend-change .hl { color: #0A1A0F; font-weight: 700; }

        .trend-bar-bg {
            height: 3px;
            background: #EEF2EE;
            border-radius: 999px;
            overflow: hidden;
            margin-top: 12px;
        }
        .trend-bar-fg {
            height: 100%;
            border-radius: 999px;
            background: #22C55E;
        }

        /* ══ 排行榜 ══ */
        .lb-card {
            background: linear-gradient(180deg, #FFFFFF 0%, #FBFCFB 100%);
            border: 1px solid #D7E3DB;
            border-radius: 28px;
            padding: 34px 38px 28px;
            margin-top: 22px;
            margin-bottom: 0;
            box-shadow: 0 10px 28px rgba(16,24,18,.05);
        }
        .lb-header {
            display: flex;
            align-items: baseline;
            justify-content: space-between;
            margin-bottom: 26px;
        }
        .lb-title {
            font-family: 'Inter Tight', sans-serif;
            font-size: 22px;
            font-weight: 800;
            color: #061408;
            letter-spacing: -.4px;
        }
        .lb-total {
            font-size: 13px;
            color: #6B7280;
            font-weight: 600;
        }

        /* 奖台 */
        .podium-row {
            display: flex;
            align-items: flex-end;
            justify-content: center;
            gap: 6px;
            margin-bottom: 26px;
            height: 158px;
        }
        .podium-col {
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .podium-avatar {
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: 'Inter Tight', sans-serif;
            font-size: 13px;
            font-weight: 800;
            margin-bottom: 8px;
            position: relative;
            color: #475569;
            background: #E5E7EB;
            letter-spacing: -.2px;
        }
        .podium-avatar.r1 {
            width: 64px;
            height: 64px;
            font-size: 16px;
            background: #052E16;
            color: #4ADE80;
            box-shadow: 0 10px 24px rgba(5,46,22,.18);
        }
        .podium-avatar.r2, .podium-avatar.r3 {
            width: 50px;
            height: 50px;
            font-size: 14px;
        }
        .podium-crown {
            position: absolute;
            top: -14px;
            left: 50%;
            transform: translateX(-50%);
            font-size: 15px;
            line-height: 1;
            color: #22C55E;
        }
        .podium-name {
            font-size: 12px;
            font-weight: 700;
            color: #334155;
            margin-bottom: 8px;
            max-width: 120px;
            text-align: center;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .podium-block {
            width: 100%;
            border-radius: 10px 10px 0 0;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .podium-block.b1 {
            background: linear-gradient(180deg, #052E16 0%, #14532D 100%);
            height: 66px;
        }
        .podium-block.b2 {
            background: linear-gradient(180deg, #DDE5E1 0%, #CBD5CF 100%);
            height: 48px;
        }
        .podium-block.b3 {
            background: linear-gradient(180deg, #ECEFED 0%, #DDE3DF 100%);
            height: 36px;
        }
        .podium-num {
            font-family: 'Inter Tight', sans-serif;
            font-size: 16px;
            font-weight: 900;
        }
        .podium-num.n1 { color: #4ADE80; }
        .podium-num.n2 { color: #475569; }
        .podium-num.n3 { color: #64748B; }

        /* 同辈比较细条（嵌入排行榜，不再割裂奖台和榜单） */
        .lb-peer-strip {
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 10px;
            background: #F7FBF8;
            border: 1px solid #D7EBDC;
            border-radius: 14px;
            padding: 10px 14px;
            margin: -4px 0 16px;
        }

        .lb-peer-chip {
            display: inline-flex;
            align-items: center;
            padding: 3px 9px;
            border-radius: 999px;
            background: #DCFCE7;
            color: #15803D;
            font-size: 10px;
            font-weight: 800;
            white-space: nowrap;
        }

        .lb-peer-cluster-inline {
            font-size: 12px;
            font-weight: 700;
            color: #475569;
            white-space: nowrap;
        }

        .lb-peer-mainline {
            flex: 1;
            min-width: 180px;
            font-size: 13px;
            font-weight: 600;
            color: #334155;
            line-height: 1.6;
        }

        .lb-peer-mainline strong {
            font-weight: 800;
            color: #061408;
        }

        .lb-peer-metric {
            font-size: 12px;
            font-weight: 600;
            color: #64748B;
            white-space: nowrap;
        }

        .lb-peer-metric strong {
            font-family: 'Inter Tight', sans-serif;
            font-size: 15px;
            font-weight: 900;
            color: #061408;
            margin-left: 4px;
        }

        .lb-peer-delta.up { color: #16A34A; }
        .lb-peer-delta.down { color: #D97706; }
        .lb-peer-delta.neu { color: #64748B; }

        .lb-peer-note {
            width: 100%;
            font-size: 12px;
            color: #6B7280;
            line-height: 1.6;
            margin-top: -2px;
        }

        /* 我的成绩条 */
        .lb-my-strip {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: linear-gradient(135deg, #F7FBF8 0%, #EEF8F1 100%);
            border: 1px solid #D7EBDC;
            border-radius: 16px;
            padding: 18px 20px;
            margin-top: 2px;
            margin-bottom: 18px;
        }
        .lb-my-label {
            font-size: 11px;
            font-weight: 700;
            color: #64748B;
            letter-spacing: .45px;
            text-transform: uppercase;
            margin-bottom: 4px;
        }
        .lb-my-rank {
            font-family: 'Inter Tight', sans-serif;
            font-size: 42px;
            font-weight: 900;
            color: #061408;
            line-height: 1;
            letter-spacing: -1.8px;
        }
        .lb-my-rank-unit {
            font-size: 13px;
            font-weight: 700;
            color: #94A3B8;
            margin-left: 4px;
        }
        .lb-my-pct {
            font-family: 'Inter Tight', sans-serif;
            font-size: 32px;
            font-weight: 900;
            letter-spacing: -1px;
            line-height: 1;
        }
        .lb-my-pct.pos { color: #16A34A; }
        .lb-my-pct.neg { color: #D97706; }
        .lb-my-pct.neu { color: #64748B; }

        /* 榜单行 */
        .lb-divider {
            height: 1px;
            background: #E8EFEB;
            margin-bottom: 14px;
        }
        .lb-list {
            display: flex;
            flex-direction: column;
        }
        .lb-row {
            display: flex;
            align-items: center;
            gap: 16px;
            padding: 15px 0;
            border-bottom: 1px solid #EEF3F0;
        }
        .lb-row:last-child { border-bottom: none; }

        .lb-rank-idx {
            font-family: 'Inter Tight', sans-serif;
            font-size: 15px;
            font-weight: 800;
            width: 24px;
            flex-shrink: 0;
            text-align: center;
            color: #94A3B8;
        }
        .lb-rank-idx.top { color: #061408; }

        .lb-me-dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: #22C55E;
            flex-shrink: 0;
        }
        .lb-name-wrap {
            flex: 1;
            display: flex;
            align-items: center;
            gap: 8px;
            min-width: 0;
        }
        .lb-dorm-name {
            font-size: 15px;
            font-weight: 600;
            color: #1F2937;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .lb-dorm-name.me {
            font-weight: 800;
            color: #061408;
        }
        .lb-me-chip {
            display: inline-flex;
            align-items: center;
            padding: 2px 8px;
            border-radius: 999px;
            background: #DCFCE7;
            color: #15803D;
            font-size: 10px;
            font-weight: 800;
            flex-shrink: 0;
        }
        .lb-pct {
            font-family: 'Inter Tight', sans-serif;
            font-size: 16px;
            font-weight: 900;
            min-width: 72px;
            text-align: right;
        }
        .lb-pct.pos { color: #16A34A; }
        .lb-pct.neg { color: #D97706; }
        .lb-pct.neu { color: #64748B; }

        .lb-section-hd {
            font-size: 11px;
            font-weight: 800;
            color: #94A3B8;
            letter-spacing: .9px;
            text-transform: uppercase;
            margin: 16px 0 8px;
        }

        /* ══ 本周干预日历 ══ */
        .week-cal {
            background: #FFFFFF;
            border: 1px solid #E6EDE8;
            border-radius: 20px;
            padding: 22px 24px;
            margin-top: 20px;
        }
        .week-cal-hd {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 16px;
        }
        .week-cal-title {
            font-size: 10px;
            font-weight: 600;
            color: #9CA3AF;
            letter-spacing: .6px;
            text-transform: uppercase;
        }
        .week-days {
            display: grid;
            grid-template-columns: repeat(7, 1fr);
            gap: 6px;
        }
        .wday {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 5px;
        }
        .wday-lbl {
            font-size: 9px;
            font-weight: 600;
            color: #B0BAB5;
            letter-spacing: .3px;
        }
        .wday-dot {
            width: 28px;
            height: 28px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 11px;
            font-weight: 700;
            color: #9CA3AF;
            background: #F2F4F3;
        }
        .wday-dot.eco { background: #F0FDF4; color: #16A34A; }
        .wday-dot.sent { background: #EFF6FF; color: #2563EB; }
        .wday-dot.today { background: #0A1A0F; color: #22C55E; }

        .wday-bar {
            width: 28px;
            height: 3px;
            border-radius: 999px;
            background: #EEF2EE;
            overflow: hidden;
        }
        .wday-bar-fg {
            height: 100%;
            border-radius: 999px;
            background: #22C55E;
        }

        /* ══ 消息/策略（接口保留） ══ */
        .sync-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
            margin-bottom: 4px;
        }
        .msg-preview {
            background: #FFFFFF;
            border: 1px solid rgba(196,233,223,.80);
            border-radius: 16px;
            padding: 14px 16px 12px;
        }
        .msg-preview.push { border-color: rgba(147,197,253,.65); }

        .msg-channel-tag {
            display: inline-block;
            font-size: 10px;
            font-weight: 700;
            letter-spacing: .4px;
            color: #0F766E;
            background: rgba(204,251,241,.90);
            border-radius: 6px;
            padding: 2px 8px;
            margin-bottom: 8px;
        }
        .msg-channel-tag.push {
            color: #1D4ED8;
            background: rgba(219,234,254,.90);
        }

        .msg-preview-title {
            font-size: 14px;
            font-weight: 700;
            color: #111827;
            line-height: 1.4;
            margin-bottom: 5px;
        }
        .msg-preview-body {
            font-size: 12px;
            color: #6B7280;
            line-height: 1.7;
            margin-bottom: 6px;
            display: -webkit-box;
            -webkit-line-clamp: 3;
            -webkit-box-orient: vertical;
            overflow: hidden;
        }
        .msg-preview-meta {
            font-size: 10px;
            color: #C4CCCA;
        }
        .sync-hint {
            font-size: 11px;
            color: #9CA3AF;
            margin-top: 4px;
            line-height: 1.6;
        }
        .strategy-noarm {
            background: #F9FAFB;
            border: 1px solid #E5E7EB;
            border-radius: 16px;
            padding: 16px 18px;
            font-size: 13px;
            color: #6B7280;
            line-height: 1.75;
            margin-bottom: 12px;
        }

        @media (max-width: 900px) {
            .side-stack {
                grid-template-rows: auto auto;
                height: auto;
            }
            .trend-strip { grid-template-columns: 1fr; }
            .sync-row { grid-template-columns: 1fr; }
        }
        </style>
        """.replace("__HOME_LAYOUT_CSS__", layout_css),
        unsafe_allow_html=True,
    )


# ===================== 辅助函数（接口完全不变） =====================

def _latest_score_for_dorm(dorm_id: str):
    scores = st.session_state.get("scores", [])
    if not scores:
        return None
    latest = None
    for rec in scores:
        if not isinstance(rec, dict):
            continue
        if str(rec.get("dorm_id")) != str(dorm_id):
            continue
        if latest is None or str(rec.get("ts", "")) > str(latest.get("ts", "")):
            latest = rec
    return latest

def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return float(default)

def _cluster_display_meta(cluster_type: str) -> dict:
    mapping = {
        "A": {"title": "A 类｜高意识 · 低行为", "tag": "重点干预对象",
              "desc": "节能意识较强但实际行为执行偏弱，更需要降低行动门槛、减少拖延并提供明确提醒。"},
        "B": {"title": "B 类｜中等意识 · 低行为", "tag": "习惯待建立",
              "desc": "具备一定节能意识但行为习惯不稳定，适合简单、低负担、可持续的提示与提醒。"},
        "C": {"title": "C 类｜高意识 · 中等行为", "tag": "需要稳定执行",
              "desc": "意识较高也有部分行为，但执行还不够稳定，适合通过反馈和提醒来巩固习惯。"},
        "D": {"title": "D 类｜中等意识 · 中等行为", "tag": "中间协调型",
              "desc": "整体处于平稳中间状态，行为有一定基础但容易受便利性和场景变化影响。"},
        "E": {"title": "E 类｜意识 · 行为一致", "tag": "表现较好",
              "desc": "意识与行为较一致，适合轻量反馈、正向强化和示范带动。"},
    }
    return mapping.get(str(cluster_type),
                       {"title": f"{cluster_type} 类", "tag": "待识别", "desc": "当前群体说明暂未配置。"})

def _arm_display_meta(arm_id: str) -> dict:
    arm = str(arm_id or "").lower()
    if not arm:
        return {"title": "暂未生成策略", "short": "等待可用模板", "desc": ""}
    label_map = {"p": "信息提示", "r": "定期提醒", "f": "数据反馈", "s": "社会比较"}
    labels = [label_map[k] for k in ["p", "r", "f", "s"] if k in arm]
    if not labels:
        return {"title": arm_id, "short": "未知类型", "desc": ""}
    if len(labels) == 1:
        return {"title": f"{labels[0]}型策略", "short": "单一机制", "desc": ""}
    return {"title": f"组合策略｜{' + '.join(labels)}", "short": "多机制联合", "desc": ""}

def _build_strategy_reason_text(cluster_type: str, best_arm: str, cs: float, bs: float) -> str:
    gap = float(cs) - float(bs)
    if cluster_type == "A":
        return f"CS={cs:.2f}、BS={bs:.2f}，意识高于行为（差距 {gap:.2f}），优先推荐促进行动落地的策略。"
    if cluster_type == "B":
        return f"CS={cs:.2f}、BS={bs:.2f}，有意识基础但行为偏弱，适合低负担、可重复的干预方式。"
    if cluster_type == "C":
        return f"CS={cs:.2f}、BS={bs:.2f}，有执行基础但稳定性不足，适合巩固和强化型策略。"
    if cluster_type == "D":
        return f"CS={cs:.2f}、BS={bs:.2f}，整体平衡，当前策略偏向轻量推动和持续维持。"
    if cluster_type == "E":
        return f"CS={cs:.2f}、BS={bs:.2f}，意识与行为一致，策略偏向正反馈和轻干预。"
    return f"根据 CS={cs:.2f}、BS={bs:.2f} 推荐策略 {best_arm}。"

def _safe_pct_text(x) -> str:
    try:
        if pd.isna(x):
            return "N/A"
        return f"{float(x) * 100:.1f}%"
    except Exception:
        return "N/A"

def _build_leaderboard_df(weekly_metrics_by_dorm: dict) -> pd.DataFrame:
    rows = []
    if not weekly_metrics_by_dorm:
        return pd.DataFrame()
    for dorm_id, m in weekly_metrics_by_dorm.items():
        if not isinstance(m, dict):
            continue
        actual = pd.to_numeric(m.get("actual_sum", np.nan), errors="coerce")
        baseline = pd.to_numeric(m.get("baseline_sum", np.nan), errors="coerce")
        if pd.isna(actual) or pd.isna(baseline) or float(baseline) <= 1e-9:
            continue
        actual, baseline = float(actual), float(baseline)
        saving = baseline - actual
        rows.append({"dorm_id": str(dorm_id), "actual_sum": actual, "baseline_sum": baseline,
                     "saving_kwh": saving, "saving_rate": saving / baseline})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values(["saving_rate", "saving_kwh"], ascending=[False, False],
                                         kind="mergesort").reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df

def _get_sim_dorm_ids_from_scores() -> list[str]:
    scores = st.session_state.get("scores", [])
    sim_ids = [str(r.get("dorm_id", "")).strip() for r in scores
               if str(r.get("dorm_id", "")).startswith("SIM_")]
    return sorted(set(sim_ids))

def _collect_demo_date_options(hourly_df=None, outcome_logs_df=None) -> list[str]:
    date_values: list[str] = []
    for df in (outcome_logs_df, hourly_df):
        if df is None or getattr(df, "empty", True):
            continue
        for col in ("timestamp_hour", "timestamp"):
            if col not in df.columns:
                continue
            dates = (
                pd.to_datetime(df[col], errors="coerce")
                .dropna()
                .dt.strftime("%Y-%m-%d")
                .tolist()
            )
            date_values.extend(dates)
            if dates:
                break

    if not date_values:
        current = str(st.session_state.get("business_date_str", pd.Timestamp.now().strftime("%Y-%m-%d")))[:10]
        return [current]

    return sorted(set(date_values))

def _sync_mobile_demo_day_selection():
    date_str = str(st.session_state.get("mobile_demo_day_select", "")).strip()[:10]
    try:
        date_val = pd.to_datetime(date_str).date()
    except Exception:
        return
    st.session_state["business_date_picker"] = date_val
    st.session_state["business_date"] = date_val
    st.session_state["business_date_str"] = pd.to_datetime(date_val).strftime("%Y-%m-%d")

def _message_exists_by_key(messages, message_key: str) -> bool:
    if not messages or not message_key:
        return False
    for row in messages:
        if isinstance(row, dict) and str(row.get("message_key", "")).strip() == str(message_key).strip():
            return True
    return False

def _get_primary_tip_for_cluster_fallback(cluster_type, dorm_id, ts=None,
                                           source_page="home", score_basis="simulated"):
    plan = get_cluster_message_plan(cluster_type)
    tip_arms = plan.get("tip", []) if isinstance(plan, dict) else []
    if not tip_arms:
        return None
    # 按日期轮转，不同日期选不同模板
    date_str = str(ts or "").split(" ")[0] or pd.Timestamp.now().strftime("%Y-%m-%d")
    try:
        day_num = int(date_str.replace("-", ""))
    except Exception:
        day_num = 0
    arm_id = tip_arms[day_num % len(tip_arms)]
    msg = build_message_from_arm(arm_id=arm_id, dorm_id=dorm_id, cluster_type=cluster_type,
                                  ts=ts, source_page=source_page, score_basis=score_basis)
    return msg if isinstance(msg, dict) else None

def _get_primary_push_for_cluster_fallback(cluster_type, dorm_id, ts=None,
                                            source_page="home", score_basis="simulated"):
    plan = get_cluster_message_plan(cluster_type)
    push_arms = plan.get("push", []) if isinstance(plan, dict) else []
    if not push_arms:
        return None
    # 按日期轮转，offset +1 让 tip 和 push 不在同一天选同一位置
    date_str = str(ts or "").split(" ")[0] or pd.Timestamp.now().strftime("%Y-%m-%d")
    try:
        day_num = int(date_str.replace("-", "")) + 1
    except Exception:
        day_num = 1
    arm_id = push_arms[day_num % len(push_arms)]
    msg = build_message_from_arm(arm_id=arm_id, dorm_id=dorm_id, cluster_type=cluster_type,
                                  ts=ts, source_page=source_page, score_basis=score_basis)
    return msg if isinstance(msg, dict) else None

def _build_home_generated_message(best_arm, selected_dorm, cluster_type, is_sim_mode, ts_now=None):
    if not best_arm:
        return None
    score_basis = "simulated" if is_sim_mode else "real"
    msg = build_message_from_arm(arm_id=best_arm, dorm_id=selected_dorm, cluster_type=cluster_type,
                                  ts=ts_now, source_page="home", score_basis=score_basis)
    if not isinstance(msg, dict):
        return None
    if not msg.get("message_key"):
        msg["message_key"] = make_message_key(dorm_id=selected_dorm, arm_id=best_arm,
                                               cluster_type=cluster_type, ts=msg.get("ts", ts_now))
    msg.setdefault("read", False)
    msg.setdefault("read_ts", None)
    msg.setdefault("source_page", "home")
    msg.setdefault("score_basis", score_basis)
    msg.setdefault("sync_status", "new")
    return msg

def _build_home_tip_message(selected_dorm, cluster_type, is_sim_mode, ts_now=None):
    score_basis = "simulated" if is_sim_mode else "real"
    msg = _get_primary_tip_for_cluster_fallback(cluster_type=cluster_type, dorm_id=selected_dorm,
                                                 ts=ts_now, source_page="home", score_basis=score_basis)
    return msg if isinstance(msg, dict) else None

def _build_home_push_message(selected_dorm, cluster_type, is_sim_mode, ts_now=None):
    score_basis = "simulated" if is_sim_mode else "real"
    msg = _get_primary_push_for_cluster_fallback(cluster_type=cluster_type, dorm_id=selected_dorm,
                                                  ts=ts_now, source_page="home", score_basis=score_basis)
    return msg if isinstance(msg, dict) else None


# ===================== 展示组件 =====================

def _render_top_bar(selected_dorm: str, actual, baseline, rate: float,
                    surpass_pct: float | None = None,
                    dorm_out=None):
    """
    顶部英雄卡：节电率大字 + 实际用电主视觉 + 嵌入式趋势小图。
    """
    actual_text = "N/A" if actual is None else f"{actual:.1f}"

    if np.isnan(rate):
        rate_text, rate_cls = "N/A", "neutral"
    elif rate >= 0.05:
        rate_text, rate_cls = f"+{rate*100:.1f}%", "positive"
    elif rate >= 0:
        rate_text, rate_cls = f"+{rate*100:.1f}%", "neutral"
    else:
        rate_text, rate_cls = f"{rate*100:.1f}%", "negative"

    surpass_html = ""
    if surpass_pct is not None:
        surpass_html = f'<div class="hero-surpass">超过了 {surpass_pct*100:.0f}% 的宿舍</div>'

    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-left">
                <div class="hero-dorm-label">宿舍 {selected_dorm}</div>
                <div class="hero-kwh-row">
                    <span class="hero-kwh-value">{actual_text}</span>
                    <span class="hero-kwh-unit">kWh</span>
                </div>
            </div>
            <div class="hero-divider"></div>
            <div class="hero-right">
                <div class="hero-rate-badge {rate_cls}">{rate_text}</div>
                {surpass_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── 趋势小图（只保留实际用电一条线，去掉"参考基线"标签） ──
    if dorm_out is not None and not dorm_out.empty:
        dplot = dorm_out.copy()
        if "timestamp_hour" not in dplot.columns and "timestamp" in dplot.columns:
            dplot["timestamp_hour"] = pd.to_datetime(dplot["timestamp"], errors="coerce")
        else:
            dplot["timestamp_hour"] = pd.to_datetime(dplot.get("timestamp_hour"), errors="coerce")
        if "kwh" not in dplot.columns and "energy_kwh" in dplot.columns:
            dplot["kwh"] = dplot["energy_kwh"]
        if all(c in dplot.columns for c in ["timestamp_hour", "kwh"]):
            chart_df = (
                dplot[["timestamp_hour", "kwh"]]
                .dropna(subset=["timestamp_hour"])
                .rename(columns={"kwh": "用电量 kWh"})
            )
            if not chart_df.empty:
                st.line_chart(
                    chart_df.set_index("timestamp_hour")[["用电量 kWh"]],
                    use_container_width=True,
                    height=120,
                )

def _render_hero_trend_chart(dorm_out):
    """
    左侧英雄卡内部趋势图：
    绿色折线 + 绿色渐变面积。
    """
    if dorm_out is None or dorm_out.empty:
        st.info("当前暂无趋势数据。")
        return

    dplot = dorm_out.copy()

    if "timestamp_hour" not in dplot.columns and "timestamp" in dplot.columns:
        dplot["timestamp_hour"] = pd.to_datetime(dplot["timestamp"], errors="coerce")
    else:
        dplot["timestamp_hour"] = pd.to_datetime(dplot.get("timestamp_hour"), errors="coerce")

    if "kwh" not in dplot.columns and "energy_kwh" in dplot.columns:
        dplot["kwh"] = pd.to_numeric(dplot["energy_kwh"], errors="coerce")

    if not all(c in dplot.columns for c in ["timestamp_hour", "kwh"]):
        st.info("趋势图字段不足。")
        return

    chart_df = (
        dplot[["timestamp_hour", "kwh"]]
        .dropna(subset=["timestamp_hour"])
        .copy()
        .sort_values("timestamp_hour")
        .tail(48)
    )

    if chart_df.empty:
        st.info("当前暂无趋势数据。")
        return

    base = alt.Chart(chart_df).encode(
        x=alt.X("timestamp_hour:T", axis=alt.Axis(title=None, grid=True, labelColor="#6B7280")),
        y=alt.Y("kwh:Q", axis=alt.Axis(title=None, grid=False, labelColor="#6B7280"))
    )

    area = base.mark_area(
        line=False,
        opacity=0.32
    ).encode(
        color=alt.value("#8BE28B")
    )

    line = base.mark_line(
        strokeWidth=3,
        interpolate="monotone"
    ).encode(
        color=alt.value("#67D96B")
    )

    point = base.mark_circle(size=28, opacity=0).encode(
        tooltip=[
            alt.Tooltip("timestamp_hour:T", title="时间"),
            alt.Tooltip("kwh:Q", title="用电量(kWh)", format=".2f"),
        ]
    )

    chart = (
        (area + line + point)
        .properties(height=250)
        .configure_view(stroke=None)
        .configure_axis(domain=False, tickColor="#D1D5DB")
    )

    st.altair_chart(chart, use_container_width=True)

def _render_cluster_csbs_cols(cluster_type, cluster_meta, cs, bs, gap, cs_source, bs_source):
    """双栏：左=群体识别，右=CS/BS 评分。合并原两个独立大卡。"""
    left, right = st.columns([1.1, 1.0])

    with left:
        st.markdown(
            f"""
            <div class="main-col-card">
                <div class="cluster-badge">{cluster_meta['tag']}</div>
                <div class="cluster-title">{cluster_meta['title']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        cs_w = max(4, min(100, int(cs * 100)))
        bs_w = max(4, min(100, int(bs * 100)))
        if gap > 0.10:
            gap_cls, gap_icon, gap_text = "warn", "↑", "意识高于行为"
        elif gap < -0.10:
            gap_cls, gap_icon, gap_text = "good", "✓", "行为表现积极"
        else:
            gap_cls, gap_icon, gap_text = "even", "●", "意识与行为均衡"

        st.markdown(
            f"""
            <div class="main-col-card">
                <div class="score-row">
                    <div class="score-chip cs">
                        <div class="score-chip-label cs">意识</div>
                        <div class="score-chip-value cs">{cs:.2f}</div>
                    </div>
                    <div class="score-chip bs">
                        <div class="score-chip-label bs">行为</div>
                        <div class="score-chip-value bs">{bs:.2f}</div>
                    </div>
                </div>
                <div class="score-bar-wrap">
                    <div class="score-bar-row">
                        <span class="score-bar-key cs">CS</span>
                        <div class="score-bar-track">
                            <div class="score-bar-fill-cs" style="width:{cs_w}%"></div>
                        </div>
                    </div>
                    <div class="score-bar-row">
                        <span class="score-bar-key bs">BS</span>
                        <div class="score-bar-track">
                            <div class="score-bar-fill-bs" style="width:{bs_w}%"></div>
                        </div>
                    </div>
                </div>
                <div class="gap-chip {gap_cls}">
                    <span>{gap_icon}</span>
                    <span>{gap_text}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

def _render_strategy_block(cluster_type, best_arm, strategy_meta, reason_text, candidate_arms, choose_err):
    """策略推荐块：精简版，去掉 strategy_meta.desc，只保留标题+arm_id+推荐理由。"""
    if best_arm is None:
        if not candidate_arms:
            diag = "当前群体尚未配置候选 arms，请前往 Admin → 配置中心 补充。"
        elif choose_err:
            diag = f"LinUCB 评分失败：{choose_err}"
        else:
            diag = reason_text
        st.markdown(
            f'<div class="strategy-noarm"><strong>暂无推荐策略</strong>　{diag}</div>',
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        f"""<div class="strategy-block">
    <div class="strategy-header">
        <div class="strategy-title">{strategy_meta.get('title', '推荐策略')}</div>
        <div class="strategy-type-badge">{strategy_meta.get('short', '')}</div>
    </div>
    <div class="strategy-arm-tag">arm: {best_arm}</div>
    <div class="strategy-reason">{reason_text}</div>
    </div>""",
        unsafe_allow_html=True,
    )

def _render_home_hero_grid(
    selected_dorm, actual, baseline, rate,
    cluster_type, cluster_meta, cs, bs, gap,
    dorm_out=None,
):
    """
    顶部主视觉：外层只保留一个 iframe 总高度，
    内部全部改成 grid，自适应分配左卡图表区和右侧评分区高度。
    """
    import json
    import streamlit.components.v1 as components

    actual_text = "N/A" if actual is None or pd.isna(actual) else f"{float(actual):.1f}"

    if pd.isna(rate):
        rate_text = "0.0%"
        rate_bg, rate_fg = "#F3F4F6", "#6B7280"
    elif rate > 0:
        rate_text = f"↗ +{rate*100:.1f}%"
        rate_bg, rate_fg = "#F0FDF4", "#15803D"
    elif rate < 0:
        rate_text = f"↘ {rate*100:.1f}%"
        rate_bg, rate_fg = "#FFFBEB", "#B45309"
    else:
        rate_text = "0.0%"
        rate_bg, rate_fg = "#F3F4F6", "#6B7280"

    # ── 趋势图数据 ────────────────────────────────────────────────
    chart_points = []
    if dorm_out is not None and not dorm_out.empty:
        dplot = dorm_out.copy()
        tcol = "timestamp_hour" if "timestamp_hour" in dplot.columns else "timestamp"
        dplot["_t"] = pd.to_datetime(dplot.get(tcol), errors="coerce")

        if "kwh" not in dplot.columns and "energy_kwh" in dplot.columns:
            dplot["kwh"] = pd.to_numeric(dplot["energy_kwh"], errors="coerce")
        else:
            dplot["kwh"] = pd.to_numeric(dplot.get("kwh"), errors="coerce")

        # ← 新增：同时读取 baseline_pred
        if "baseline_pred" in dplot.columns:
            dplot["_bl"] = pd.to_numeric(dplot["baseline_pred"], errors="coerce")
        else:
            dplot["_bl"] = float("nan")

        if "_t" in dplot.columns and "kwh" in dplot.columns:
            dplot = dplot.dropna(subset=["_t", "kwh"]).sort_values("_t")
            for _, row in dplot.iterrows():
                chart_points.append({
                    "t": str(row["_t"])[:16],
                    "v": round(float(row["kwh"]), 3),
                    "b": None if pd.isna(row["_bl"]) else round(float(row["_bl"]), 3),
                })

    pts_json = json.dumps(chart_points, ensure_ascii=False)

    # ── 干预事件数据（供标注层使用） ──────────────────────────────
    # 策略类型 → 色 / 标签映射，与 JS 端保持一致
    _STYPE_COLOR = {"P": "#3B82F6", "R": "#16A34A", "F": "#7C3AED", "S": "#F97316"}
    _STYPE_LABEL = {"P": "P·信息提示", "R": "R·定期提醒", "F": "F·数据反馈", "S": "S·社会比较"}

    ivt_points = []   # intervention points
    try:
        ivt_logs = st.session_state.get("intervention_logs", [])
        selected_dorm_str = str(selected_dorm)
        for log in ivt_logs:
            if not isinstance(log, dict):
                continue
            if str(log.get("dorm_id", "")) != selected_dorm_str:
                continue
            # 时间字段：优先 decision_t0_hour，兜底 timestamp / decision_time
            ts_raw = (log.get("decision_t0_hour")
                      or log.get("timestamp")
                      or log.get("decision_time", ""))
            if not ts_raw:
                continue
            ts_str = str(ts_raw)[:16]   # "YYYY-MM-DD HH:MM"

            stype = str(log.get("message_strategy_type", "")).strip().upper()
            if stype not in _STYPE_COLOR:
                # 从 arm_id 前缀推断
                arm = str(log.get("arm_id", "")).strip().upper()
                stype = arm[0] if arm and arm[0] in _STYPE_COLOR else "R"

            arm_id  = str(log.get("arm_id", "未知"))
            title   = str(log.get("message_title", arm_id))[:28]
            reward  = log.get("reward_sum_12h")
            reward_str = f"{float(reward):+.3f} kWh" if reward is not None else "待归因"

            arm_tail = arm_id.split("_")[-1] if "_" in arm_id else ""
            badge_text = f"{stype}{arm_tail}" if arm_tail.isdigit() else stype
            
            tip_title = _STYPE_LABEL.get(stype, stype)
            tip_sub = f"{arm_id}｜{title}"
            
            ivt_points.append({
                "t":      ts_str,
                "stype":  stype,
                "color":  _STYPE_COLOR[stype],
                "label":  _STYPE_LABEL.get(stype, stype),
                "arm":    arm_id,
                "title":  title,
                "reward": reward_str,
                "badge": badge_text,
                "tip_title": tip_title,
                "tip_sub": tip_sub,
            })
    except Exception:
        ivt_points = []

    ivt_json = json.dumps(ivt_points, ensure_ascii=False)

    # ── CS/BS 数值 ────────────────────────────────────────────────
    cs_w = max(4, min(100, int(cs * 100)))
    bs_w = max(4, min(100, int(bs * 100)))

    if gap > 0.10:
        gap_bg, gap_bd, gap_fg = "#FEF9EC", "#F6CC5A", "#7A4F00"
        gap_icon, gap_label = "↑", "意识高于行为"
    elif gap < -0.10:
        gap_bg, gap_bd, gap_fg = "#F0FDF4", "#86EFAC", "#15803D"
        gap_icon, gap_label = "✓", "行为表现积极"
    else:
        gap_bg, gap_bd, gap_fg = "#F5F7F5", "#D1D9D4", "#4B5563"
        gap_icon, gap_label = "●", "意识与行为均衡"

    # 小组件：CS-BS 象限解释
    dot_x = max(8, min(92, cs * 100))
    dot_y = max(8, min(92, (1 - bs) * 100))

    if cs >= 0.67 and bs >= 0.67:
        quad_title = "高意识 · 高行为"
        quad_desc = "意识与行为都较好，适合保持与轻反馈。"
    elif cs >= 0.67 and bs < 0.67:
        quad_title = "高意识 · 低行为"
        quad_desc = "认知较强，但执行仍有提升空间。"
    elif cs < 0.67 and bs >= 0.67:
        quad_title = "低意识 · 高行为"
        quad_desc = "行为表现较好，可继续强化稳定习惯。"
    else:
        quad_title = "低意识 · 低行为"
        quad_desc = "建议先降低行动门槛，再逐步强化习惯。"

    cluster_tag = cluster_meta.get("tag", "")
    cluster_title = cluster_meta.get("title", "")

    chart_html = (
        '<div class="chart-wrap">'
        '<canvas id="ec"></canvas>'
        '<div class="tip" id="tip">'
        '  <div class="tip-inner">'
        '    <div class="tip-avatar" id="tip-avatar"></div>'
        '    <div class="tip-body">'
        '      <div class="tip-title" id="tip-title"></div>'
        '      <div class="tip-sub" id="tip-sub"></div>'
        '    </div>'
        '  </div>'
        '</div>'
        '<div class="dot" id="dot"></div>'
        '</div>'
        '<div class="date-row"><span id="d0"></span><span id="d1"></span></div>'
    ) if chart_points else '<div class="no-data">暂无用电数据</div>'

    hero_height = 500 if chart_points else 340

    full_html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Inter+Tight:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<style>
* {{
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}}
html, body {{
  width: 100%;
  height: 100%;
  background: #F2F4F3;
  font-family: 'Inter Tight', sans-serif;
  overflow: hidden;
}}
.frame {{
  width: 100%;
  height: 100%;
}}
.grid {{
  display: grid;
  grid-template-columns: minmax(0, 1.34fr) minmax(0, 1fr);
  gap: 16px;
  width: 100%;
  height: 100%;
  align-items: stretch;
}}

/* ── 左：英雄卡 ── */
.hero {{
  background: #fff;
  border: 1px solid #E6EDE8;
  border-radius: 24px;
  padding: 28px 30px 0;
  display: grid;
  grid-template-rows: auto auto auto auto minmax(0, 1fr) auto;
  row-gap: 10px;
  height: 100%;
  min-width: 0;
  min-height: 0;
}}
.dorm-lbl {{
  font-size: 11px;
  font-weight: 500;
  color: #9CA3AF;
  letter-spacing: .45px;
  text-transform: uppercase;
}}
.kwh-num {{
  font-size: 78px;
  font-weight: 800;
  color: #0A1A0F;
  line-height: .92;
  letter-spacing: -4px;
}}
.kwh-unit {{
  font-size: 14px;
  font-weight: 500;
  color: #9CA3AF;
  line-height: 1.2;
}}
.rate-badge {{
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: max-content;
  min-height: 28px;
  padding: 0 12px;
  border-radius: 8px;
  font-size: 13px;
  font-weight: 700;
  line-height: 1;
  background: {rate_bg};
  color: {rate_fg};
}}
.chart-wrap {{
  position: relative;
  min-height: 0;
  margin: 2px -30px 0;
  overflow: hidden;
}}
canvas {{
  display: block;
  width: 100%;
  height: 100%;
  cursor: crosshair;
}}
.tip {{
  position: absolute;
  background: #FFFFFF;
  border: none;
  border-radius: 14px;
  padding: 10px 14px 10px 10px;
  font-size: 12px;
  font-weight: 400;
  pointer-events: none;
  display: none;
  transform: translate(-50%, calc(-100% - 16px));
  box-shadow: 0 8px 28px rgba(0,0,0,0.13), 0 1px 4px rgba(0,0,0,0.08);
  white-space: nowrap;
  min-width: 160px;
  z-index: 10;
  font-family: 'Inter Tight', sans-serif;
}}
.tip::after {{
  content: '';
  position: absolute;
  bottom: -6px;
  left: 50%;
  transform: translateX(-50%);
  width: 12px; height: 12px;
  background: #FFFFFF;
  clip-path: polygon(0 0, 100% 0, 50% 100%);
  box-shadow: 0 2px 4px rgba(0,0,0,0.06);
}}
.tip-inner {{
  display: flex;
  align-items: center;
  gap: 10px;
}}
.tip-avatar {{
  width: 32px; height: 32px;
  border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 13px; font-weight: 800;
  color: #fff;
  flex-shrink: 0;
}}
.tip-body {{
  display: flex;
  flex-direction: column;
  gap: 1px;
}}
.tip-title {{
  font-size: 13px;
  font-weight: 700;
  color: #0F172A;
  line-height: 1.3;
}}
.tip-sub {{
  font-size: 11px;
  color: #6B7280;
  line-height: 1.5;
}}
.dot {{
  position: absolute;
  width: 10px; height: 10px;
  border-radius: 50%;
  background: #16A34A;
  border: 2.5px solid #FFFFFF;
  box-shadow: 0 2px 8px rgba(22,163,74,0.35);
  pointer-events: none;
  display: none;
  transform: translate(-50%, -50%);
}}
.date-row {{
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 10px;
  color: #C8D3CC;
  padding: 0 30px 12px;
}}
.no-data {{
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 120px;
  color: #C8D3CC;
  font-size: 13px;
}}

/* ── 右：两张卡，改成 grid ── */
.side {{
  display: grid;
  grid-template-rows: auto minmax(0, 1fr);
  gap: 14px;
  height: 100%;
  min-width: 0;
  min-height: 0;
}}
.scard {{
  background: #fff;
  border: 1px solid #E6EDE8;
  border-radius: 24px;
  padding: 24px 24px;
  min-width: 0;
}}
.cluster-card {{
  display: grid;
  align-content: start;
  row-gap: 8px;
}}
.score-card {{
  display: grid;
  grid-template-rows: auto auto auto;
  align-content: start;
  row-gap: 12px;
  min-height: 0;
}}

/* 群体卡 */
.c-tag {{
  display: inline-flex;
  align-items: center;
  width: max-content;
  font-size: 10px;
  font-weight: 700;
  color: #15803D;
  background: #F0FDF4;
  border-radius: 6px;
  padding: 4px 8px;
  letter-spacing: .3px;
}}
.c-type {{
  font-size: 11px;
  font-weight: 500;
  color: #9CA3AF;
  line-height: 1.3;
}}
.c-title {{
  font-size: 20px;
  font-weight: 800;
  color: #0A1A0F;
  line-height: 1.18;
  letter-spacing: -.3px;
  overflow-wrap: anywhere;
}}

/* 评分卡 */
.score-pair {{
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
}}
.sc {{
  border-radius: 16px;
  padding: 12px 14px;
  min-width: 0;
}}
.sc.cs {{
  background: #F5F8FF;
  border: 1px solid #DBEAFE;
}}
.sc.bs {{
  background: #F5FFF8;
  border: 1px solid #D1FAE5;
}}
.sc-lbl {{
  font-size: 10px;
  font-weight: 700;
  letter-spacing: .35px;
  margin-bottom: 8px;
  line-height: 1.2;
}}
.sc-lbl.cs {{ color: #3B82F6; }}
.sc-lbl.bs {{ color: #22C55E; }}

.sc-val {{
  font-size: 32px;
  font-weight: 800;
  line-height: 1;
  letter-spacing: -1.6px;
}}
.sc-val.cs {{ color: #2563EB; }}
.sc-val.bs {{ color: #16A34A; }}

.bars {{
  display: grid;
  row-gap: 8px;
}}
.bar-line {{
  display: grid;
  grid-template-columns: 22px minmax(0, 1fr);
  align-items: center;
  column-gap: 10px;
}}
.bar-k {{
  font-size: 10px;
  font-weight: 700;
  line-height: 1;
}}
.bar-k.cs {{ color: #3B82F6; }}
.bar-k.bs {{ color: #22C55E; }}

.bar-t {{
  height: 5px;
  background: #EEF2EE;
  border-radius: 999px;
  overflow: hidden;
}}
.bar-f {{
  height: 100%;
  border-radius: 999px;
}}
.bar-f.cs {{ background: #3B82F6; }}
.bar-f.bs {{ background: #22C55E; }}

.status-tag {{
  display: inline-flex;
  align-items: center;
  gap: 6px;
  width: max-content;
  max-width: 100%;
  font-size: 10px;
  font-weight: 700;
  line-height: 1.2;
  padding: 7px 10px;
  border-radius: 8px;
  background: {gap_bg};
  border: 1px solid {gap_bd};
  color: {gap_fg};
  justify-self: start;
}}
/* 小组件：CS-BS 象限 */
.insight-card{{
  margin-top: 12px;
  padding-top: 12px;
  border-top: 1px solid #EEF2EE;
  display: grid;
  grid-template-columns: 116px minmax(0,1fr);
  gap: 12px;
  align-items: start;
}}

.quad{{
  position: relative;
  width: 116px;
  height: 116px;
  border: 1px solid #E6EDE8;
  border-radius: 14px;
  background:
    linear-gradient(to right, transparent 49.5%, #E9EEEA 49.5%, #E9EEEA 50.5%, transparent 50.5%),
    linear-gradient(to bottom, transparent 49.5%, #E9EEEA 49.5%, #E9EEEA 50.5%, transparent 50.5%),
    #FBFCFB;
  overflow: hidden;
}}

.quad-label{{
  position: absolute;
  font-size: 9px;
  font-weight: 600;
  color: #A3AEA8;
  line-height: 1;
}}

.quad-label.tl{{ top: 8px; left: 8px; }}
.quad-label.tr{{top: 8px; right: 8px; }}
.quad-label.bl{{bottom: 8px; left: 8px; }}
.quad-label.br{{bottom: 8px; right: 8px; }}
.quad-dot{{
  position: absolute;
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: #16A34A;
  border: 2px solid #FFFFFF;
  box-shadow: 0 2px 10px rgba(34,197,94,.28);
  transform: translate(-50%, -50%);
}}

.quad-axis-x,
.quad-axis-y{{
  position: absolute;
  font-size: 9px;
  font-weight: 700;
  color: #7C8A82;
  letter-spacing: .2px;
}}

.quad-axis-x{{
  bottom: -18px;
  left: 50%;
  transform: translateX(-50%);
}}

.quad-axis-y{{
  top: 50%;
  left: -18px;
  transform: translateY(-50%) rotate(-90deg);
}}

.insight-text{{
  min-width: 0;
}}

.insight-kicker{{
  font-size: 10px;
  font-weight: 700;
  color: #9CA3AF;
  letter-spacing: .4px;
  text-transform: uppercase;
  margin-bottom: 5px;
}}

.insight-title{{
  font-size: 13px;
  font-weight: 800;
  color: #0A1A0F;
  line-height: 1.35;
  margin-bottom: 6px;
}}

.insight-desc{{
  font-size: 11px;
  font-weight: 500;
  color: #6B7280;
  line-height: 1.6;
}}

@media (max-width: 900px) {{
  .grid {{
    grid-template-columns: 1fr;
  }}
  .hero {{
    grid-template-rows: auto auto auto auto minmax(140px, 1fr) auto;
  }}
}}
</style>
</head>
<body>
<div class="frame">
  <div class="grid">

    <div class="hero">
      <div class="dorm-lbl">{selected_dorm}</div>
      <div class="kwh-num">{actual_text}</div>
      <div class="kwh-unit">千瓦时 kWh</div>
      <div class="rate-badge">{rate_text}</div>
      {chart_html}
    </div>

    <div class="side">
        <div class="scard cluster-card">
            <div class="c-tag">{cluster_tag}</div>
            <div class="c-type">群体类别 {cluster_type}</div>
            <div class="c-title">{cluster_title}</div>
        </div>

        <div class="scard score-card">
          <div class="score-pair">
            <div class="sc cs">
                <div class="sc-lbl cs">CS · 意识</div>
                <div class="sc-val cs">{cs:.2f}</div>
            </div>
            <div class="sc bs">
                <div class="sc-lbl bs">BS · 行为</div>
                <div class="sc-val bs">{bs:.2f}</div>
            </div>
        </div>

        <div class="bars">
          <div class="bar-line">
            <span class="bar-k cs">CS</span>
                <div class="bar-t">
                <div class="bar-f cs" style="width:{cs_w}%"></div>
            </div>
        </div>
        <div class="bar-line">
            <span class="bar-k bs">BS</span>
            <div class="bar-t">
                <div class="bar-f bs" style="width:{bs_w}%"></div>
            </div>
          </div>
        </div>

        <div class="status-tag">{gap_icon}&nbsp;&nbsp;{gap_label}</div>

        <div class="insight-card">
          <div class="quad">
            <div class="quad-label tl">高意低行</div>
            <div class="quad-label tr">高意高行</div>
            <div class="quad-label bl">低意低行</div>
            <div class="quad-label br">低意高行</div>

            <div class="quad-dot" style="left:{dot_x}%; top:{dot_y}%;"></div>

            <div class="quad-axis-x">CS</div>
            <div class="quad-axis-y">BS</div>
          </div>

          <div class="insight-text">
            <div class="insight-kicker">群体解释</div>
            <div class="insight-title">{quad_title}</div>
            <div class="insight-desc">{quad_desc}</div>
          </div>
        </div>
      </div>
    </div>

<script>
(function() {{
  const ALL = {pts_json};
  const IVT = {ivt_json};
  if (!ALL.length) return;

  const WEEK = 7 * 24;   // 一周 = 168 个小时点
  const canvas = document.getElementById('ec');
  const wrap   = canvas.parentElement;
  const tipEl  = document.getElementById('tip');
  const dotEl  = document.getElementById('dot');
  const d0El   = document.getElementById('d0');
  const d1El   = document.getElementById('d1');

  // ── 把全量数据切成以7天为单位的若干周 ──────────────────────────
  function buildWeeks(pts) {{
    if (!pts.length) return [pts];
    const weeks = [];
    let i = pts.length - 1;          // 从最新数据向前切
    while (i >= 0) {{
      const end   = i + 1;
      const start = Math.max(0, end - WEEK);
      weeks.unshift(pts.slice(start, end));
      i = start - 1;
    }}
    return weeks;
  }}

  const weeks = buildWeeks(ALL);
  let weekIdx = weeks.length - 1;    // 默认显示最新一周

  // ── 翻页按钮（画在 date-row 旁边，JS 动态注入） ─────────────────
  const dateRow = document.getElementById('d0').parentElement;
  dateRow.style.cssText = 'display:flex;align-items:center;justify-content:space-between;padding:6px 8px 0;';

  const btnPrev = document.createElement('button');
  const btnNext = document.createElement('button');
  const weekLabel = document.createElement('span');

  const btnStyle = `
    background:none;border:1px solid rgba(156,163,175,0.4);border-radius:6px;
    color:#6B7280;font-size:12px;padding:2px 9px;cursor:pointer;line-height:1.6;
  `;
  btnPrev.setAttribute('style', btnStyle);
  btnNext.setAttribute('style', btnStyle);
  weekLabel.style.cssText = 'font-size:11px;color:#9CA3AF;text-align:center;flex:1;';

  btnPrev.textContent = '‹ 上周';
  btnNext.textContent = '下周 ›';

  // 把 d0/d1 span 隐藏，改用 weekLabel 显示周区间
  d0El.style.display = 'none';
  d1El.style.display = 'none';

  dateRow.innerHTML = '';
  dateRow.appendChild(btnPrev);
  dateRow.appendChild(weekLabel);
  dateRow.appendChild(btnNext);

  function updateNav() {{
    btnPrev.disabled = weekIdx <= 0;
    btnNext.disabled = weekIdx >= weeks.length - 1;
    btnPrev.style.opacity = weekIdx <= 0 ? '0.3' : '1';
    btnNext.style.opacity = weekIdx >= weeks.length - 1 ? '0.3' : '1';
    const pts = weeks[weekIdx];
    if (pts.length) {{
      weekLabel.textContent = pts[0].t.slice(0,10) + ' – ' + pts[pts.length-1].t.slice(0,10);
    }}
  }}

  btnPrev.onclick = () => {{ if (weekIdx > 0) {{ weekIdx--; zoomLen=-1; zoomStart=0; drawZoom(); updateNav(); }} }};
  btnNext.onclick = () => {{ if (weekIdx < weeks.length - 1) {{ weekIdx++; zoomLen=-1; zoomStart=0; drawZoom(); updateNav(); }} }};

  // ── 主绘制函数 ───────────────────────────────────────────────────
  function draw() {{
    const pts = weeks[weekIdx];
    if (!pts || !pts.length) return;

    const dpr = window.devicePixelRatio || 1;
    const W   = Math.max(wrap.clientWidth  || 540, 320);
    const H   = Math.max(wrap.clientHeight || 180, 140);

    canvas.width  = W * dpr;
    canvas.height = H * dpr;
    canvas.style.width  = W + 'px';
    canvas.style.height = H + 'px';

    const ctx = canvas.getContext('2d');
    ctx.setTransform(1,0,0,1,0,0);
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, W, H);

    const allVals = hasBl ? [...vals, ...bls.filter(b => b != null)] : vals;
    const rawMin  = Math.min(...allVals);
    const rawMax  = Math.max(...allVals);
    // 留出上下 12% 边距，让曲线不贴边
    const span    = rawMax - rawMin || 1;
    const minV    = rawMin - span * 0.12;
    const maxV    = rawMax + span * 0.12;

    const pad = {{ l: 0, r: 0, t: 10, b: 2 }};
    const cw  = W - pad.l - pad.r;
    const ch  = H - pad.t - pad.b;
    const n   = pts.length;

    function xp(i) {{ return n === 1 ? W/2 : pad.l + (i/(n-1)) * cw; }}
    function yp(v) {{ return pad.t + (1 - (v - minV)/(maxV - minV)) * ch; }}

    // ① 节省/超耗色块（actual vs baseline 围成的区域）
    if (hasBl) {{
      for (let i = 0; i < n - 1; i++) {{
        const bC = bls[i], bN = bls[i+1];
        if (bC == null || bN == null) continue;
        const vC = vals[i], vN = vals[i+1];
        // 判断这一段的主要方向
        const saving = ((vC + vN) / 2) <= ((bC + bN) / 2);
        ctx.beginPath();
        ctx.moveTo(xp(i),   yp(vC));
        ctx.lineTo(xp(i+1), yp(vN));
        ctx.lineTo(xp(i+1), yp(bN));
        ctx.lineTo(xp(i),   yp(bC));
        ctx.closePath();
        // 节省绿色更饱和，超耗红更显眼
        ctx.fillStyle = saving ? 'rgba(22,163,74,0.22)' : 'rgba(220,38,38,0.18)';
        ctx.fill();
      }}
    }}

    // ② actual 绿色渐变面积
    const grad = ctx.createLinearGradient(0, pad.t, 0, H);
    grad.addColorStop(0, 'rgba(22,163,74,0.18)');
    grad.addColorStop(1, 'rgba(22,163,74,0)');
    ctx.beginPath();
    ctx.moveTo(xp(0), yp(vals[0]));
    for (let i = 1; i < n; i++) ctx.lineTo(xp(i), yp(vals[i]));
    ctx.lineTo(xp(n-1), H);
    ctx.lineTo(xp(0),   H);
    ctx.closePath();
    ctx.fillStyle = grad;
    ctx.fill();

    // ③ baseline 灰色虚线 —— 更粗更显眼
    if (hasBl) {{
      ctx.beginPath();
      let started = false;
      for (let i = 0; i < n; i++) {{
        if (bls[i] == null) {{ started = false; continue; }}
        if (!started) {{ ctx.moveTo(xp(i), yp(bls[i])); started = true; }}
        else ctx.lineTo(xp(i), yp(bls[i]));
      }}
      ctx.setLineDash([5, 4]);
      ctx.strokeStyle = 'rgba(120,120,130,0.70)';   // 更深的灰
      ctx.lineWidth   = 2.0;                          // 更粗
      ctx.lineJoin    = 'round';
      ctx.stroke();
      ctx.setLineDash([]);
    }}

    // ④ actual 绿色实线 —— 更粗更鲜明
    ctx.beginPath();
    ctx.moveTo(xp(0), yp(vals[0]));
    for (let i = 1; i < n; i++) ctx.lineTo(xp(i), yp(vals[i]));
    ctx.strokeStyle = '#16A34A';    // 更饱和的绿
    ctx.lineWidth   = 2.2;          // 更粗
    ctx.lineJoin    = 'round';
    ctx.lineCap     = 'round';
    ctx.stroke();

    // ⑤ y 轴参考线（基线水平值对应的位置）—— 仅在有 baseline 时画
    if (hasBl) {{
      const blAvg = bls.filter(b => b != null).reduce((a,b)=>a+b,0) / bls.filter(b=>b!=null).length;
      const yRef  = yp(blAvg);
      ctx.beginPath();
      ctx.moveTo(0, yRef); ctx.lineTo(W, yRef);
      ctx.setLineDash([2, 6]);
      ctx.strokeStyle = 'rgba(120,120,130,0.25)';
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.setLineDash([]);
    }}

    // ⑥ 干预事件标注竖线（叠加在最上层）
    const weekStart = pts[0].t;
    const weekEnd   = pts[pts.length - 1].t;
    const ivtInWeek = IVT.filter(e => e.t >= weekStart && e.t <= weekEnd);

    canvas._meta = {{ pad, cw, ch, minV, maxV, n, W, H, pts, ivtInWeek }};
  }}


// ── Tooltip 悬停 ─────────────────────────────────────────────────
let m_cache = null;
let _tipTimer = null;

// 统一显示数据点 tooltip：如果附近有策略事件，就把策略放进左侧圆圈
function showDataTip(p, px, py, nearEv = null) {{
  const blTxt = p.b != null ? ' · 基线 ' + p.b.toFixed(2) : '';

  const avatar  = document.getElementById('tip-avatar');
  const titleEl = document.getElementById('tip-title');
  const subEl   = document.getElementById('tip-sub');

  if (!avatar || !titleEl || !subEl) return;

  if (nearEv) {{
    avatar.textContent = nearEv.stype || '';
    avatar.style.background = nearEv.color || '#16A34A';

    titleEl.textContent = p.v.toFixed(2) + ' kWh';
    subEl.innerHTML =
      p.t.slice(0, 10) + blTxt + '<br>' +
      (nearEv.label || nearEv.stype || '') + '<br>' +
      (nearEv.arm || '');
  }} else {{
    avatar.textContent = '·';
    avatar.style.background = '#16A34A';

    titleEl.textContent = p.v.toFixed(2) + ' kWh';
    subEl.textContent = p.t.slice(0, 10) + blTxt;
  }}

  tipEl.style.cssText =
    'display:block;left:' + px + 'px;top:' + py + 'px;' +
    'transform:translate(-50%, calc(-100% - 10px));';

  dotEl.style.cssText =
    'display:block;left:' + px + 'px;top:' + py + 'px;';
}}

function onMove(e) {{
  const m = canvas._meta;
  if (!m) return;
  m_cache = m;

  const rect = canvas.getBoundingClientRect();
  const clientX = e.touches ? e.touches[0].clientX : e.clientX;
  const mx = clientX - rect.left;

  clearTimeout(_tipTimer);
  _tipTimer = setTimeout(() => {{
    let idx = 0;
    if (m.n > 1) {{
      idx = Math.round((mx - m.pad.l) / m.cw * (m.n - 1));
      idx = Math.max(0, Math.min(m.n - 1, idx));
    }}

    const p  = m.pts[idx];
    const px = m.n === 1 ? m.W / 2 : m.pad.l + (idx / (m.n - 1)) * m.cw;
    const py = m.pad.t + (1 - (p.v - m.minV) / (m.maxV - m.minV)) * m.ch;

    // 找当前点附近最近的干预事件（允许 1 小时误差）
    const pTime = new Date(String(p.t).replace(' ', 'T')).getTime();
    let nearEv = null;
    let bestDt = Infinity;

    (m.ivtInWeek || []).forEach(ev => {{
      const evTime = new Date(String(ev.t).replace(' ', 'T')).getTime();
      const dt = Math.abs(evTime - pTime);
      if (dt < bestDt) {{
        bestDt = dt;
        nearEv = ev;
      }}
    }});

    if (bestDt > 12 * 60 * 60 * 1000) {{
        nearEv = null;
    }}

    showDataTip(p, px, py, nearEv);
  }}, 120);
}}

function onLeave() {{
  clearTimeout(_tipTimer);
  tipEl.style.display = 'none';
  dotEl.style.display = 'none';
}}
  
  // ── 滚轮缩放（放大后在当前周内平移视窗） ─────────────────────
  let zoomStart = 0;    // 当前视窗在本周数据里的起始索引
  let zoomLen   = -1;   // -1 表示显示全部，否则为窗口长度

  function getViewPts() {{
    const full = weeks[weekIdx] || [];
    if (zoomLen <= 0 || zoomLen >= full.length) return full;
    const start = Math.max(0, Math.min(zoomStart, full.length - zoomLen));
    return full.slice(start, start + zoomLen);
  }}

  // 重写 draw 让它用 getViewPts() 代替直接用 weeks[weekIdx]
  const _drawOrig = draw;
  function drawZoom() {{
    const pts = getViewPts();
    if (!pts || !pts.length) return;

    const dpr = window.devicePixelRatio || 1;
    const W   = Math.max(wrap.clientWidth  || 540, 320);
    const H   = Math.max(wrap.clientHeight || 180, 140);
    canvas.width  = W * dpr;
    canvas.height = H * dpr;
    canvas.style.width  = W + 'px';
    canvas.style.height = H + 'px';

    const ctx = canvas.getContext('2d');
    ctx.setTransform(1,0,0,1,0,0);
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, W, H);

    const rawVals = pts.map(p => p.v);
    const bls   = pts.map(p => p.b);
    const hasBl = bls.some(b => b != null);

    const SW = 4;
    const vals = rawVals.map((_, i) => {{
      const lo  = Math.max(0, i - Math.floor(SW / 2));
      const hi  = Math.min(rawVals.length - 1, i + Math.floor(SW / 2));
      let sum = 0;
      for (let j = lo; j <= hi; j++) sum += rawVals[j];
      return sum / (hi - lo + 1);
    }});

    const allVals = hasBl ? [...vals, ...bls.filter(b => b != null)] : vals;
    const rawMin  = Math.min(...allVals);
    const rawMax  = Math.max(...allVals);
    const span    = rawMax - rawMin || 1;
    const minV    = rawMin - span * 0.12;
    const maxV    = rawMax + span * 0.12;
    const pad     = {{ l: 0, r: 0, t: 10, b: 2 }};
    const cw      = W - pad.l - pad.r;
    const ch      = H - pad.t - pad.b;
    const n       = pts.length;
    function xp(i) {{ return n === 1 ? W/2 : pad.l + (i/(n-1)) * cw; }}
    function yp(v) {{ return pad.t + (1 - (v - minV)/(maxV - minV)) * ch; }}

    // ① 节省/超耗色块
    if (hasBl) {{
      for (let i = 0; i < n-1; i++) {{
        const bC = bls[i], bN = bls[i+1];
        if (bC == null || bN == null) continue;
        const saving = ((vals[i]+vals[i+1])/2) <= ((bC+bN)/2);
        ctx.beginPath();
        ctx.moveTo(xp(i), yp(vals[i])); ctx.lineTo(xp(i+1), yp(vals[i+1]));
        ctx.lineTo(xp(i+1), yp(bN));   ctx.lineTo(xp(i), yp(bC));
        ctx.closePath();
        ctx.fillStyle = saving ? 'rgba(22,163,74,0.22)' : 'rgba(220,38,38,0.18)';
        ctx.fill();
      }}
    }}
    // ② actual 渐变面积
    const grad = ctx.createLinearGradient(0, pad.t, 0, H);
    grad.addColorStop(0, 'rgba(22,163,74,0.18)');
    grad.addColorStop(1, 'rgba(22,163,74,0)');
    ctx.beginPath();
    ctx.moveTo(xp(0), yp(vals[0]));
    for (let i=1; i<n; i++) ctx.lineTo(xp(i), yp(vals[i]));
    ctx.lineTo(xp(n-1), H); ctx.lineTo(xp(0), H); ctx.closePath();
    ctx.fillStyle = grad; ctx.fill();
    // ③ baseline 虚线
    if (hasBl) {{
      ctx.beginPath(); let started = false;
      for (let i=0; i<n; i++) {{
        if (bls[i]==null) {{ started=false; continue; }}
        if (!started) {{ ctx.moveTo(xp(i),yp(bls[i])); started=true; }}
        else ctx.lineTo(xp(i), yp(bls[i]));
      }}
      ctx.setLineDash([5,4]); ctx.strokeStyle='rgba(120,120,130,0.70)';
      ctx.lineWidth=2.0; ctx.lineJoin='round'; ctx.stroke(); ctx.setLineDash([]);
    }}
    // ④ actual 实线
    ctx.beginPath(); ctx.moveTo(xp(0), yp(vals[0]));
    for (let i=1; i<n; i++) ctx.lineTo(xp(i), yp(vals[i]));
    ctx.strokeStyle='#16A34A'; ctx.lineWidth=2.2;
    ctx.lineJoin='round'; ctx.lineCap='round'; ctx.stroke();
    // ⑤ baseline 均值参考线
    if (hasBl) {{
      const blAvg = bls.filter(b=>b!=null).reduce((a,b)=>a+b,0)/bls.filter(b=>b!=null).length;
      ctx.beginPath(); ctx.moveTo(0,yp(blAvg)); ctx.lineTo(W,yp(blAvg));
      ctx.setLineDash([2,6]); ctx.strokeStyle='rgba(120,120,130,0.25)';
      ctx.lineWidth=1; ctx.stroke(); ctx.setLineDash([]);
    }}

    // ⑥ 当前周内的干预事件：只保留数据，不再绘制顶部固定圆点
    const weekStart = pts[0].t;
    const weekEnd   = pts[pts.length - 1].t;
    const ivtInWeek = IVT.filter(e => e.t >= weekStart && e.t <= weekEnd);

    canvas._meta = {{ pad, cw, ch, minV, maxV, n, W, H, pts, ivtInWeek }};
  }}

  // 滚轮事件
  canvas.addEventListener('wheel', e => {{
    e.preventDefault();
    const full = weeks[weekIdx] || [];
    if (!full.length) return;
    if (zoomLen < 0) zoomLen = full.length;

    const rect   = canvas.getBoundingClientRect();
    const mx     = e.clientX - rect.left;
    const ratio  = Math.max(0, Math.min(1, mx / rect.width));
    const focus  = Math.round(zoomStart + ratio * zoomLen);  // 鼠标对应的数据索引

    // 向上滚 = 放大（缩短窗口），向下滚 = 缩小（拉长窗口）
    const factor = e.deltaY < 0 ? 0.75 : 1.33;
    zoomLen = Math.round(zoomLen * factor);
    zoomLen = Math.max(12, Math.min(full.length, zoomLen));   // 最小显示12个点

    // 保持鼠标位置的数据点不动
    zoomStart = Math.round(focus - ratio * zoomLen);
    zoomStart = Math.max(0, Math.min(full.length - zoomLen, zoomStart));

    drawZoom(); updateNav();
  }}, {{ passive: false }});

  // 双击重置缩放
  canvas.addEventListener('dblclick', () => {{
    zoomLen = -1; zoomStart = 0;
    drawZoom(); updateNav();
  }});

  canvas.addEventListener('mousemove', onMove);
  canvas.addEventListener('touchmove', onMove, {{ passive: true }});
  canvas.addEventListener('mouseleave', onLeave);
  canvas.addEventListener('touchend', onLeave);

  // ── 触摸左右滑动翻页 ────────────────────────────────────────────
  let touchStartX = 0;
  canvas.addEventListener('touchstart', e => {{ touchStartX = e.touches[0].clientX; }}, {{ passive: true }});
  canvas.addEventListener('touchend', e => {{
    const dx = e.changedTouches[0].clientX - touchStartX;
    if (Math.abs(dx) > 40) {{
      if (dx < 0 && weekIdx < weeks.length - 1) {{ weekIdx++; zoomLen=-1; zoomStart=0; drawZoom(); updateNav(); }}
      if (dx > 0 && weekIdx > 0)               {{ weekIdx--; zoomLen=-1; zoomStart=0; drawZoom(); updateNav(); }}
    }}
  }});
  // ── 初始化 ──────────────────────────────────────────────────────
  drawZoom();
  updateNav();
  window.addEventListener('resize', () => {{ drawZoom(); }});
}})();
</script>
</body>
</html>
"""
    components.html(full_html, height=hero_height, scrolling=False)


def _sync_hint(msg, candidate_arms, linucb_params, choose_err):
    """disabled 状态下给出最具体的原因提示。"""
    if msg is None:
        hint = "当前群体暂无匹配模板内容。"
    elif not candidate_arms:
        hint = "候选 arms 未配置 → Admin › 配置中心"
    elif linucb_params is None:
        hint = "LinUCB 参数未初始化。"
    elif choose_err:
        hint = f"选臂出错：{choose_err}"
    else:
        hint = "当前不可同步。"
    st.markdown(f'<div class="sync-hint">⚠ {hint}</div>', unsafe_allow_html=True)


def _render_sync_zone(
    selected_dorm, cluster_type, is_sim_mode,
    tip_msg, push_msg, ts_now,
    cs, bs, best_arm, best_detail,
    candidate_arms, linucb_params, choose_err,
    x, recent_diff_3h, is_weekend,
    render_slot="home_main",
):
    """
    消息预览（HTML 双栏）+ 同步按钮（st.columns）。
    session_state 写入逻辑与原版本完全一致，未做任何修改。
    """
    tip_disabled = (not candidate_arms or linucb_params is None or choose_err is not None or tip_msg is None)
    push_disabled = (not candidate_arms or linucb_params is None or choose_err is not None or push_msg is None)

    tip_html = (
        f'<div class="msg-preview">'
        f'<div class="msg-channel-tag">P · 小贴士</div>'
        f'<div class="msg-preview-title">{tip_msg.get("title", "")}</div>'
        f'<div class="msg-preview-body">{tip_msg.get("body", "")}</div>'
        f'<div class="msg-preview-meta">模板 {tip_msg.get("arm_id", "")} · 群体 {cluster_type}</div>'
        f'</div>'
    ) if tip_msg else '<div class="msg-preview" style="color:#94A3B8;font-size:13px;padding:14px;">暂无可生成的 P 小贴士</div>'

    push_html = (
        f'<div class="msg-preview push">'
        f'<div class="msg-channel-tag push">R · 推送</div>'
        f'<div class="msg-preview-title">{push_msg.get("title", "")}</div>'
        f'<div class="msg-preview-body">{push_msg.get("body", "")}</div>'
        f'<div class="msg-preview-meta">模板 {push_msg.get("arm_id", "")} · 群体 {cluster_type}</div>'
        f'</div>'
    ) if push_msg else '<div class="msg-preview push" style="color:#94A3B8;font-size:13px;padding:14px;">暂无可生成的 R 推送</div>'

    st.markdown(f'<div class="sync-row">{tip_html}{push_html}</div>', unsafe_allow_html=True)


    safe_date = str(ts_now).split(" ")[0]
    tip_arm = str(tip_msg.get("arm_id", "na")) if tip_msg else "na"
    push_arm = str(push_msg.get("arm_id", "na")) if push_msg else "na"
    safe_cluster = str(cluster_type or "na").replace(" ", "_")
    safe_dorm = str(selected_dorm or "na").replace(" ", "_")

    tip_btn_key = f"home_sync_tip_{safe_dorm}_{safe_cluster}_{tip_arm}_{safe_date}"
    push_btn_key = f"home_sync_push_{safe_dorm}_{safe_cluster}_{push_arm}_{safe_date}"
    
    st.caption(f"TIP KEY = {tip_btn_key}")
    st.caption(f"PUSH KEY = {push_btn_key}")

    btn_c1, btn_c2 = st.columns(2)

    with btn_c1:
        if st.button(
                    "同步 P 小贴士到 Messages",  
                     key=tip_btn_key,
                     disabled=tip_disabled, 
                     use_container_width=True
                     ):
            messages = st.session_state.get("messages", [])
            interaction_logs = st.session_state.get("interaction_logs", [])
            intervention_logs = st.session_state.get("intervention_logs", [])
            scores = st.session_state.get("scores", [])
            msg_key = str(tip_msg.get("message_key", "")).strip()
            arm_id_now = tip_msg.get("arm_id", "")
            if _message_exists_by_key(messages, msg_key):
                st.warning("P 小贴士今天已同步过，已跳过重复写入。")
            else:
                messages.append(tip_msg)
                intervention_logs.append({
                    "dorm_id": selected_dorm, "decision_time": ts_now,
                    "date": str(ts_now).split(" ")[0], "CS": float(cs),
                    "BS_energy_week": float(bs), "cluster_type": cluster_type,
                    "candidate_arms": list(candidate_arms), "arm_id": arm_id_now,
                    "message_key": msg_key, "message_title": tip_msg.get("title", ""),
                    "message_channel": tip_msg.get("channel", ""),
                    "message_strategy_type": tip_msg.get("strategy_type", ""),
                    "linucb_score": float(best_detail["score"]),
                    "linucb_mu": float(best_detail["mu"]),
                    "linucb_bonus": float(best_detail["bonus"]),
                    "reward_value": None, "reward_ready": 0, "window_hours": 12,
                    "assignment_mode": "LinUCB", "source_page": "home",
                    "score_basis": tip_msg.get("score_basis", "simulated" if is_sim_mode else "real"),
                    "state_vector": list(map(float, x.tolist())),
                })
                intervention_logs.append({
                    "timestamp": ts_now, "date": str(ts_now).split(" ")[0],
                    "dorm_id": selected_dorm, "cluster_type": cluster_type,
                    "arm_id": arm_id_now, "message_key": msg_key,
                    "message_channel": tip_msg.get("channel", ""),
                    "message_strategy_type": tip_msg.get("strategy_type", ""),
                    "algo_type": "LinUCB", "ucb_score": float(best_detail["score"]),
                    "p_hat": float(best_detail["mu"]), "bonus": float(best_detail["bonus"]),
                    "reward_sum_12h": None, "update_timestamp": None,
                    "decision_t0_hour": ts_now, "source_page": "home",
                    "score_basis": tip_msg.get("score_basis", "simulated" if is_sim_mode else "real"),
                    "state_json": json.dumps({
                        "CS": float(cs), "BS": float(bs),
                        "recent_diff_3h": float(recent_diff_3h), "is_weekend": int(is_weekend),
                    }, ensure_ascii=False),
                })
                interaction_logs.append({
                    "timestamp": ts_now, "date": str(ts_now).split(" ")[0],
                    "event": "home_sync_tip", "page": "home",
                    "dorm_id": selected_dorm, "cluster_type": cluster_type,
                    "arm_id": arm_id_now, "message_key": msg_key,
                    "channel": tip_msg.get("channel", ""),
                    "strategy_type": tip_msg.get("strategy_type", ""),
                    "score_basis": tip_msg.get("score_basis", "simulated" if is_sim_mode else "real"),
                })
                st.session_state["messages"] = messages
                st.session_state["interaction_logs"] = interaction_logs
                st.session_state["intervention_logs"] = intervention_logs
                st.session_state["scores"] = scores
                st.session_state["current_dorm_for_messages"] = selected_dorm
                st.success("✅ P 小贴士已同步到 Messages。")
        if tip_disabled:
            _sync_hint(tip_msg, candidate_arms, linucb_params, choose_err)

    with btn_c2:
        if st.button(
            "同步 R 推送到 Messages",
             key=push_btn_key,
             disabled=push_disabled, 
             use_container_width=True
             ):
            messages = st.session_state.get("messages", [])
            interaction_logs = st.session_state.get("interaction_logs", [])
            intervention_logs = st.session_state.get("intervention_logs", [])
            scores = st.session_state.get("scores", [])
            msg_key = str(push_msg.get("message_key", "")).strip()
            arm_id_now = push_msg.get("arm_id", "")
            if _message_exists_by_key(messages, msg_key):
                st.warning("R 推送今天已同步过，已跳过重复写入。")
            else:
                messages.append(push_msg)
                intervention_logs.append({
                    "dorm_id": selected_dorm, "decision_time": ts_now,
                    "date": str(ts_now).split(" ")[0], "CS": float(cs),
                    "BS_energy_week": float(bs), "cluster_type": cluster_type,
                    "candidate_arms": list(candidate_arms), "arm_id": arm_id_now,
                    "message_key": msg_key, "message_title": push_msg.get("title", ""),
                    "message_channel": push_msg.get("channel", ""),
                    "message_strategy_type": push_msg.get("strategy_type", ""),
                    "linucb_score": float(best_detail["score"]),
                    "linucb_mu": float(best_detail["mu"]),
                    "linucb_bonus": float(best_detail["bonus"]),
                    "reward_value": None, "reward_ready": 0, "window_hours": 12,
                    "assignment_mode": "LinUCB", "source_page": "home",
                    "score_basis": push_msg.get("score_basis", "simulated" if is_sim_mode else "real"),
                    "state_vector": list(map(float, x.tolist())),
                })
                intervention_logs.append({
                    "timestamp": ts_now, "date": str(ts_now).split(" ")[0],
                    "dorm_id": selected_dorm, "cluster_type": cluster_type,
                    "arm_id": arm_id_now, "message_key": msg_key,
                    "message_channel": push_msg.get("channel", ""),
                    "message_strategy_type": push_msg.get("strategy_type", ""),
                    "algo_type": "LinUCB", "ucb_score": float(best_detail["score"]),
                    "p_hat": float(best_detail["mu"]), "bonus": float(best_detail["bonus"]),
                    "reward_sum_12h": None, "update_timestamp": None,
                    "decision_t0_hour": ts_now, "source_page": "home",
                    "score_basis": push_msg.get("score_basis", "simulated" if is_sim_mode else "real"),
                    "state_json": json.dumps({
                        "CS": float(cs), "BS": float(bs),
                        "recent_diff_3h": float(recent_diff_3h), "is_weekend": int(is_weekend),
                    }, ensure_ascii=False),
                })
                interaction_logs.append({
                    "timestamp": ts_now, "date": str(ts_now).split(" ")[0],
                    "event": "home_sync_push", "page": "home",
                    "dorm_id": selected_dorm, "cluster_type": cluster_type,
                    "arm_id": arm_id_now, "message_key": msg_key,
                    "channel": push_msg.get("channel", ""),
                    "strategy_type": push_msg.get("strategy_type", ""),
                    "score_basis": push_msg.get("score_basis", "simulated" if is_sim_mode else "real"),
                })
                st.session_state["messages"] = messages
                st.session_state["interaction_logs"] = interaction_logs
                st.session_state["intervention_logs"] = intervention_logs
                st.session_state["scores"] = scores
                st.session_state["current_dorm_for_messages"] = selected_dorm
                st.success("✅ R 推送已同步到 Messages。")
        if push_disabled:
            _sync_hint(push_msg, candidate_arms, linucb_params, choose_err)


def _render_perf_section(dorm_out, has_outcome_data, actual, baseline, rate):
    """本周能耗 KPI mini tiles（趋势图已移至顶部英雄卡内嵌）。"""
    st.markdown('<div class="perf-section-header">本周能耗数据</div>', unsafe_allow_html=True)

    if has_outcome_data and actual is not None:
        rate_text = _safe_pct_text(rate) if not np.isnan(rate) else "N/A"
        st.markdown(
            f"""
            <div class="kpi-row">
                <div class="kpi-mini">
                    <div class="kpi-mini-label">实际用电</div>
                    <div class="kpi-mini-value">{actual:.1f}<span class="kpi-mini-unit"> kWh</span></div>
                </div>
                <div class="kpi-mini">
                    <div class="kpi-mini-label">参考基线</div>
                    <div class="kpi-mini-value">{baseline:.1f}<span class="kpi-mini-unit"> kWh</span></div>
                </div>
                <div class="kpi-mini">
                    <div class="kpi-mini-label">节电比例</div>
                    <div class="kpi-mini-value">{rate_text}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("当前宿舍暂无 outcome 数据。")

def _render_trend_strip(selected_dorm: str, actual, rate, dorm_out):
    """三指标横排：本周用电 / 节电率 / 排名超越情况。"""
    # ── 上周用电（从 outcome_logs 取前7天对比） ──────────────────
    prev_actual = None
    if dorm_out is not None and not dorm_out.empty:
        try:
            tmp = dorm_out.copy()
            tcol = "timestamp_hour" if "timestamp_hour" in tmp.columns else "timestamp"
            tmp["_t"] = pd.to_datetime(tmp.get(tcol), errors="coerce")
            vcol = "kwh" if "kwh" in tmp.columns else ("energy_kwh" if "energy_kwh" in tmp.columns else None)
            if vcol and "_t" in tmp.columns:
                tmp = tmp.dropna(subset=["_t"])
                tmp = tmp.sort_values("_t")
                cutoff = tmp["_t"].max() - pd.Timedelta(days=7)
                prev = tmp[tmp["_t"] < cutoff]
                if not prev.empty:
                    prev_actual = float(pd.to_numeric(prev[vcol], errors="coerce").fillna(0).sum())
        except Exception:
            prev_actual = None

    # ── 上周节电率（session_state 取不到则不显示） ─────────────
    prev_rate = None  # 无历史周度数据，保守处理

    # ── 排行榜数据 ───────────────────────────────────────────────
    current_rank = total_n = None
    try:
        wm = st.session_state.get("weekly_metrics_by_dorm", {})
        rdf = _build_leaderboard_df(wm)
        if not rdf.empty and str(selected_dorm) in rdf["dorm_id"].astype(str).tolist():
            cr = rdf[rdf["dorm_id"].astype(str) == str(selected_dorm)].iloc[0]
            current_rank = int(cr["rank"])
            total_n = len(rdf)
    except Exception:
        pass

    # ── 渲染 ─────────────────────────────────────────────────────
    kwh_text  = f"{actual:.1f}" if actual is not None else "N/A"
    rate_cls  = "pos" if (rate is not None and not np.isnan(rate) and rate > 0.001) else \
                "neg" if (rate is not None and not np.isnan(rate) and rate < 0) else ""
    rate_text = _safe_pct_text(rate) if (rate is not None and not np.isnan(rate)) else "N/A"

    # 用电量对比文字
    if prev_actual is not None and actual is not None:
        diff = actual - prev_actual
        if diff > 0:
            kwh_change = f'较上周 <em class="dn">↑ {diff:.1f} kWh</em>'
        elif diff < 0:
            kwh_change = f'较上周 <em class="up">↓ {abs(diff):.1f} kWh</em>'
        else:
            kwh_change = '较上周 持平'
    else:
        kwh_change = '暂无上周对比'

    # 节电率对比文字
    rate_change = '暂无上周对比'

    # 排名文字
    if current_rank is not None and total_n is not None:
        rank_text  = f'{current_rank}<span class="trend-unit">/ {total_n} 间</span>'
        rank_change = f'排名 <em class="hl">第 {current_rank}</em>，本周数据'
        rank_bar_w  = max(4, int((total_n - current_rank + 1) / total_n * 100))
    else:
        rank_text   = 'N/A'
        rank_change = '暂无排名数据'
        rank_bar_w  = 4

    # 节电率进度条（0%~5% 范围映射到 0~100%）
    rate_bar_w = 0
    if rate is not None and not np.isnan(rate):
        rate_bar_w = max(0, min(100, int(rate * 2000)))

    st.markdown(
        f"""
<div class="trend-strip">
  <div class="trend-item">
    <div class="trend-lbl">本周用电</div>
    <div class="trend-val">{kwh_text}<span class="trend-unit">kWh</span></div>
    <div class="trend-change">{kwh_change}</div>
    <div class="trend-bar-bg"><div class="trend-bar-fg" style="width:68%"></div></div>
  </div>
  <div class="trend-item">
    <div class="trend-lbl">节电率</div>
    <div class="trend-val {rate_cls}">{rate_text}</div>
    <div class="trend-change">{rate_change}</div>
    <div class="trend-bar-bg"><div class="trend-bar-fg" style="width:{rate_bar_w}%;background:#3B82F6;"></div></div>
  </div>
  <div class="trend-item">
    <div class="trend-lbl">当前排名</div>
    <div class="trend-val">{rank_text}</div>
    <div class="trend-change">{rank_change}</div>
    <div class="trend-bar-bg"><div class="trend-bar-fg" style="width:{rank_bar_w}%;background:#A78BFA;"></div></div>
  </div>
</div>""",
        unsafe_allow_html=True,
    )

def _render_week_calendar(selected_dorm: str, selected_date_str: str, dorm_out):
    """本周干预日历：7格，显示节电状态 + 是否发送干预 + 每日用电小条。"""
    try:
        today    = pd.Timestamp(selected_date_str)
        weekday  = today.dayofweek          # 0=Mon
        week_mon = today - pd.Timedelta(days=weekday)
    except Exception:
        return

    # 每日用电量
    daily_kwh: dict = {}
    if dorm_out is not None and not dorm_out.empty:
        try:
            tmp  = dorm_out.copy()
            tcol = "timestamp_hour" if "timestamp_hour" in tmp.columns else "timestamp"
            tmp["_t"] = pd.to_datetime(tmp.get(tcol), errors="coerce")
            vcol = "kwh" if "kwh" in tmp.columns else ("energy_kwh" if "energy_kwh" in tmp.columns else None)
            if vcol:
                tmp["_d"] = tmp["_t"].dt.date
                daily_kwh = tmp.groupby("_d")[vcol].apply(
                    lambda s: float(pd.to_numeric(s, errors="coerce").fillna(0).sum())
                ).to_dict()
        except Exception:
            pass

    max_kwh = max(daily_kwh.values()) if daily_kwh else 1.0

    # 今日已发送的消息日期集合
    messages    = st.session_state.get("messages", [])
    sent_dates  = set()
    for m in messages:
        if not isinstance(m, dict):
            continue
        if str(m.get("dorm_id", "")) != str(selected_dorm):
            continue
        ts = str(m.get("ts", ""))
        if len(ts) >= 10:
            sent_dates.add(ts[:10])

    day_names = ["一", "二", "三", "四", "五", "六", "日"]
    days_html = ""
    for i in range(7):
        d        = week_mon + pd.Timedelta(days=i)
        d_str    = d.strftime("%Y-%m-%d")
        d_num    = d.day
        is_today = (d_str == selected_date_str)
        is_sent  = (d_str in sent_dates)
        kwh_v    = daily_kwh.get(d.date(), 0)
        bar_w    = max(0, min(100, int(kwh_v / max_kwh * 100))) if max_kwh > 0 else 0

        if is_today:
            dot_cls = "today"
        elif is_sent:
            dot_cls = "sent"
        elif kwh_v > 0:
            dot_cls = "eco"
        else:
            dot_cls = ""

        days_html += (
            f'<div class="wday">'
            f'  <div class="wday-lbl">{day_names[i]}</div>'
            f'  <div class="wday-dot {dot_cls}">{d_num}</div>'
            f'  <div class="wday-bar"><div class="wday-bar-fg" style="width:{bar_w}%"></div></div>'
            f'</div>'
        )

    st.markdown(
        f"""
<div class="week-cal">
  <div class="week-cal-hd">
    <span class="week-cal-title">本周干预日历</span>
  </div>
  <div class="week-days">{days_html}</div>
</div>""",
        unsafe_allow_html=True,
    )


def _render_social_leaderboard(selected_dorm: str, top_n: int = 3):
    """节能排行榜：精致奖台 + 我的成绩条 + 纯文字列表 + 展开完整榜。"""
    weekly_metrics_by_dorm = st.session_state.get("weekly_metrics_by_dorm", {})
    rank_df = _build_leaderboard_df(weekly_metrics_by_dorm)
    if rank_df.empty:
        st.info("当前暂无足够的周度数据，暂时无法生成节能排行榜。")
        return
    rank_df["dorm_id"] = rank_df["dorm_id"].astype(str)
    if str(selected_dorm) not in rank_df["dorm_id"].tolist():
        st.info("当前宿舍尚未进入排行榜计算范围。")
        return

    current_row  = rank_df[rank_df["dorm_id"] == str(selected_dorm)].iloc[0]
    total_n      = len(rank_df)
    current_rank = int(current_row["rank"])
    current_rate = float(current_row["saving_rate"])
    rank_df["bar_rate"] = rank_df["saving_rate"].clip(lower=0.0)
    my_pct_cls = "neg" if current_rate < 0 else ("pos" if current_rate > 0.001 else "neu")
        # ── 同辈比较（按当前 cluster_type） ─────────────────────────
    score_map = {}
    for rec in st.session_state.get("scores", []):
        if not isinstance(rec, dict):
            continue
        did = str(rec.get("dorm_id", "")).strip()
        if not did:
            continue
        if did not in score_map or str(rec.get("ts", "")) > str(score_map[did].get("ts", "")):
            score_map[did] = rec

    cur_cluster = str(score_map.get(str(selected_dorm), {}).get("cluster_type", "")).upper()
    rank_df["peer_cluster"] = rank_df["dorm_id"].map(
        lambda x: str(score_map.get(str(x), {}).get("cluster_type", "")).upper()
    )

    peer_rank_text = "—"
    peer_delta_text = "—"
    peer_delta_cls = "neu"
    peer_tip = "社会比较模块会在这里提示你与相似宿舍的差距。"

    if cur_cluster:
        peer_df = rank_df[rank_df["peer_cluster"] == cur_cluster].copy()
        if not peer_df.empty:
            peer_df = peer_df.sort_values(
                ["saving_rate", "saving_kwh"],
                ascending=[False, False]
            ).reset_index(drop=True)
            peer_df["peer_rank"] = np.arange(1, len(peer_df) + 1)

            my_peer = peer_df[peer_df["dorm_id"] == str(selected_dorm)]
            if not my_peer.empty:
                my_peer_row = my_peer.iloc[0]
                peer_rank = int(my_peer_row["peer_rank"])
                peer_total = len(peer_df)
                peer_rank_text = f"{peer_rank}/{peer_total}"

                peer_surpass = (
                    (peer_total - peer_rank) / max(peer_total - 1, 1)
                    if peer_total > 1 else 1.0
                )
                peer_avg = float(peer_df["saving_rate"].mean())
                peer_delta = float(current_rate - peer_avg)

                peer_delta_text = f"{peer_delta*100:+.1f}%"
                peer_delta_cls = (
                    "up" if peer_delta > 0.001
                    else ("down" if peer_delta < -0.001 else "neu")
                )

                peer_headline = f"在 {cur_cluster} 类同辈中，你超过了 {peer_surpass*100:.0f}% 的宿舍"
                peer_subline = f"同群体平均节电 {_safe_pct_text(peer_avg)}，你当前 {_safe_pct_text(current_rate)}"

                if peer_rank > 1:
                    prev_peer = peer_df[peer_df["peer_rank"] == peer_rank - 1].iloc[0]
                    gap_prev = float(prev_peer["saving_rate"] - current_rate)
                    peer_tip = f"距离前一位同辈 {prev_peer['dorm_id']} 还差 {gap_prev*100:.1f}%"
                else:
                    peer_tip = "你目前已经处于该同辈组前列，适合做示范带动。"
                peer_cluster_text = f"{cur_cluster} 类同辈" if cur_cluster else "同辈参照"

                if peer_rank_text == "—":
                    peer_main_text = "当前暂无稳定的同群体参照数据"
                    peer_note_text = "补充更多同群体宿舍后，这里会显示你与相似宿舍的相对位置。"
                else:
                    peer_main_text = (
                        f"在 <strong>{peer_cluster_text}</strong> 中，"
                        f"你当前位于 <strong>{peer_rank_text}</strong>，"
                        f"相对均值 <strong class=\"lb-peer-delta {peer_delta_cls}\">{peer_delta_text}</strong>"
                    )
                    peer_note_text = peer_tip

                peer_strip_html = (
                    f'<div class="lb-peer-strip">'
                    f'  <span class="lb-peer-chip">S · 社会比较</span>'
                    f'  <span class="lb-peer-cluster-inline">{peer_cluster_text}</span>'
                    f'  <span class="lb-peer-mainline">{peer_main_text}</span>'
                    f'  <span class="lb-peer-metric">同辈排名 <strong>{peer_rank_text}</strong></span>'
                    f'  <span class="lb-peer-metric">相对均值 <strong class="lb-peer-delta {peer_delta_cls}">{peer_delta_text}</strong></span>'
                    f'  <div class="lb-peer-note">{peer_note_text}</div>'
                    f'</div>'
                )


    # ── 奖台（2nd 左 / 1st 中 / 3rd 右） ─────────────────────────
    top3    = rank_df.head(3).to_dict("records")
    p_order = [1, 0, 2]
    pod_html = '<div class="podium-row">'
    for pos in p_order:
        if pos >= len(top3):
            pod_html += '<div class="podium-col"></div>'
            continue
        row    = top3[pos]
        rank_n = int(row["rank"])
        dname  = str(row["dorm_id"])[-6:]
        r_cls  = "r1" if rank_n == 1 else (f"r{rank_n}")
        b_cls  = f"b{rank_n}"
        n_cls  = f"n{rank_n}"
        crown  = '<span class="podium-crown">✦</span>' if rank_n == 1 else ""
        mt     = "0" if rank_n == 1 else "auto"
        name_w = "font-weight:700;color:#0A1A0F;" if rank_n == 1 else ""
        pod_html += (
            f'<div class="podium-col" style="margin-top:{mt};">'
            f'  <div class="podium-avatar {r_cls}">{crown}{dname}</div>'
            f'  <div class="podium-name" style="{name_w}">{row["dorm_id"]}</div>'
            f'  <div class="podium-block {b_cls}">'
            f'    <span class="podium-num {n_cls}">{rank_n}</span>'
            f'  </div>'
            f'</div>'
        )
    pod_html += '</div>'

    # ── 行 HTML ───────────────────────────────────────────────────
    def _row_html(row, is_me=False):
        rank_n   = int(row["rank"])
        rate_v   = float(row["saving_rate"])
        pct_cls  = "neg" if rate_v < 0 else ("pos" if rate_v > 0.001 else "neu")
        idx_cls  = "top" if rank_n <= 3 else ""
        me_dot   = '<span class="lb-me-dot"></span>' if is_me else ""
        name_cls = "me" if is_me else ""
        me_chip  = '<span class="lb-me-chip">我的宿舍</span>' if is_me else ""
        return (
            f'<div class="lb-row {"me" if is_me else ""}">'
            f'  <span class="lb-rank-idx {idx_cls}">{rank_n}</span>'
            f'  {me_dot}'
            f'  <div class="lb-name-wrap">'
            f'    <span class="lb-dorm-name {name_cls}">{row["dorm_id"]}</span>'
            f'    {me_chip}'
            f'  </div>'
            f'  <span class="lb-pct {pct_cls}">{_safe_pct_text(rate_v)}</span>'
            f'</div>'
        )

    # ── 精简列表 ─────────────────────────────────────────────────
    short_rows = ""
    shown_me   = False
    for _, row in rank_df.head(top_n).iterrows():
        is_me = (str(row["dorm_id"]) == str(selected_dorm))
        if is_me:
            shown_me = True
        short_rows += _row_html(row.to_dict(), is_me=is_me)
    if not shown_me:
        short_rows += _row_html(current_row.to_dict(), is_me=True)

    # ── 完整榜分段 ───────────────────────────────────────────────
    top3_rows = ""
    rest_rows = ""
    for _, row in rank_df.iterrows():
        rank_n = int(row["rank"])
        is_me  = (str(row["dorm_id"]) == str(selected_dorm))
        h      = _row_html(row.to_dict(), is_me=is_me)
        if rank_n <= 3:
            top3_rows += h
        else:
            rest_rows += h

    # ── 渲染 ─────────────────────────────────────────────────────
    st.markdown(
        f'<div class="lb-card">'
        f'  <div class="lb-header">'
        f'    <span class="lb-title">节能排行榜</span>'
        f'    <span class="lb-total">本周 · 共 {total_n} 间宿舍</span>'
        f'  </div>'
        f'  {pod_html}'
        f'  {peer_strip_html}'
        f'  <div class="lb-my-strip">'
        f'    <div>'
        f'      <div class="lb-my-label">我的排名</div>'
        f'      <div><span class="lb-my-rank">{current_rank}</span>'
        f'      <span class="lb-my-rank-unit">/ {total_n}</span></div>'
        f'    </div>'
        f'    <div style="text-align:right;">'
        f'      <div class="lb-my-label">本周节电</div>'
        f'      <div class="lb-my-pct {my_pct_cls}">{_safe_pct_text(current_rate)}</div>'
        f'    </div>'
        f'  </div>'
        f'  <div class="lb-divider"></div>'
        f'  <div class="lb-list">{short_rows}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    with st.expander("查看完整排行榜", expanded=False):
        rest_section = (
            f'<div class="lb-section-hd">其余宿舍</div>'
            f'<div class="lb-list">{rest_rows}</div>'
        ) if rest_rows else ""
        st.markdown(
            f'<div class="lb-card" style="margin-bottom:0;">'
            f'  <div class="lb-my-strip">'
            f'    <div>'
            f'      <div class="lb-my-label">我的排名</div>'
            f'      <div><span class="lb-my-rank">{current_rank}</span>'
            f'      <span class="lb-my-rank-unit">/ {total_n}</span></div>'
            f'    </div>'
            f'    <div style="text-align:right;">'
            f'      <div class="lb-my-label">本周节电</div>'
            f'      <div class="lb-my-pct {my_pct_cls}">{_safe_pct_text(current_rate)}</div>'
            f'    </div>'
            f'  </div>'
            f'  <div class="lb-divider"></div>'
            f'  <div class="lb-section-hd">前 3 名</div>'
            f'  <div class="lb-list">{top3_rows}</div>'
            f'  {rest_section}'
            f'</div>',
            unsafe_allow_html=True,
        )


# ===================== 主页面 =====================
# 页面结构：
#   ① 宿舍选择（一行三列：来源 radio | 宿舍 selectbox | 更多设置 expander）
#   ② 顶部状态条（节电率大字 + pills，合并原 hero + toolbar）
#   ③ 双栏：群体识别卡 | CS/BS 评分对比（合并原两个独立大卡）
#   ④ 策略推荐块（精简，去长描述）
#   ⑤ 今日干预消息预览 + 同步按钮
#   ── divider ──
#   ⑥ 本周能耗表现（KPI mini + 趋势图，下沉）
#   ⑦ expander：节能排行榜（默认折叠）
#   ⑧ expander：调试信息（默认折叠）

def render_home_page(
    hourly_df,
    outcome_logs_df,
    baseline_model=None,
    load_stats=None,
    daily_cap=2,
    alpha_ucb=1.0,
    log_interaction_func=None,
    freq_guard_allow_daily_func=None,
    already_decided_today_func=None,
    mobile=False,
):
    _inject_home_style(mobile=mobile)

    if hourly_df is None or hourly_df.empty:
        st.warning("当前没有可用的小时级能耗数据。")
        return

    real_dorm_ids = sorted(hourly_df["dorm_id"].dropna().astype(str).unique().tolist())
    sim_dorm_ids = _get_sim_dorm_ids_from_scores()
    if not real_dorm_ids and not sim_dorm_ids:
        st.warning("未找到可用的宿舍 ID。")
        return

    demo_date_options = _collect_demo_date_options(hourly_df, outcome_logs_df)

    # ── ① 宿舍选择 ──────────────────────────────────
    source_real = "真实宿舍"
    source_sim = "模拟宿舍（测试）"
    source_options = [source_real] + ([source_sim] if sim_dorm_ids else [])
    if st.session_state.get("home_dorm_source") not in source_options:
        st.session_state["home_dorm_source"] = source_options[0]

    if mobile:
        dorm_source = st.session_state.get("home_dorm_source", source_options[0])
    else:
        sel_c1, sel_c2, sel_c3 = st.columns([1.2, 1.6, 1.2])
        with sel_c1:
            dorm_source = st.radio(
                "宿舍来源",
                source_options,
                horizontal=True,
                key="home_dorm_source",
            )

    selectable_dorm_ids = real_dorm_ids if dorm_source == source_real else sim_dorm_ids
    if not selectable_dorm_ids:
        st.warning("当前来源下暂无可选宿舍。")
        return

    dorm_key = f"home_selected_dorm_{'real' if dorm_source == source_real else 'sim'}"
    current_dorm_for_messages = str(st.session_state.get("current_dorm_for_messages", "")).strip()
    if dorm_key not in st.session_state:
        st.session_state[dorm_key] = (current_dorm_for_messages
                                       if current_dorm_for_messages in selectable_dorm_ids
                                       else selectable_dorm_ids[0])
    if st.session_state.get(dorm_key) not in selectable_dorm_ids:
        st.session_state[dorm_key] = (current_dorm_for_messages
                                       if current_dorm_for_messages in selectable_dorm_ids
                                       else selectable_dorm_ids[0])
    if mobile:
        current_date_str = str(
            st.session_state.get("business_date_str", pd.Timestamp.now().strftime("%Y-%m-%d"))
        )[:10]
        if (
            "mobile_demo_day_select" not in st.session_state
            or st.session_state.get("mobile_demo_day_select") not in demo_date_options
        ):
            st.session_state["mobile_demo_day_select"] = (
                current_date_str if current_date_str in demo_date_options else demo_date_options[-1]
            )

        day_label_by_date = {
            date_str: f"第 {idx + 1} 天"
            for idx, date_str in enumerate(demo_date_options)
        }
        with st.container(key="home_demo_control_bar"):
            day_col, dorm_col = st.columns(2, gap="small")
            with day_col:
                selected_date_str = st.selectbox(
                    "演示日期",
                    demo_date_options,
                    key="mobile_demo_day_select",
                    format_func=lambda d: day_label_by_date.get(str(d), str(d)),
                    label_visibility="collapsed",
                    on_change=_sync_mobile_demo_day_selection,
                )
            with dorm_col:
                selected_dorm = st.selectbox(
                    "选择宿舍",
                    selectable_dorm_ids,
                    key=dorm_key,
                    label_visibility="collapsed",
                )
        st.session_state["business_date"] = pd.to_datetime(selected_date_str).date()
        st.session_state["business_date_str"] = str(selected_date_str)
        st.session_state["current_dorm_for_messages"] = selected_dorm
        tau = float(st.session_state.get(f"tau_{selected_dorm}", 0.30))
    else:
        selected_date_str = str(
            st.session_state.get("business_date_str", pd.Timestamp.now().strftime("%Y-%m-%d"))
        )
        with sel_c2:
            selected_dorm = st.selectbox("选择宿舍", selectable_dorm_ids, key=dorm_key)
            st.session_state["current_dorm_for_messages"] = selected_dorm
        with sel_c3:
            with st.expander("更多设置"):
                tau = st.slider("BS 灵敏度 tau", 0.10, 1.00, 0.30, 0.05, key=f"tau_{selected_dorm}")
                if load_stats is not None:
                    st.write(load_stats)

    ts_now = f"{selected_date_str} 09:00:00"
    is_sim_mode = (dorm_source == source_sim)

    # ── 数据准备（不变） ─────────────────────────────────────────
    dorm_hourly = get_dorm_hourly(hourly_df, selected_dorm)
    has_hourly_data = dorm_hourly is not None and not dorm_hourly.empty
    if not has_hourly_data:
        if not is_sim_mode:
            st.info("该宿舍暂无小时级能耗数据。")
            return
        st.info("模拟宿舍测试模式：暂无小时级能耗数据，将继续展示分群识别与策略推荐结果。")

    if has_hourly_data:
        ts_col = "timestamp" if "timestamp" in dorm_hourly.columns else "timestamp_hour"
        latest_ts = (pd.to_datetime(dorm_hourly[ts_col], errors="coerce").max()
                     if ts_col in dorm_hourly.columns else pd.Timestamp.now())
    else:
        latest_ts = pd.Timestamp.now()

    dorm_out = st.session_state.get("dorm_outcome_map", {}).get(selected_dorm)
    if dorm_out is None and outcome_logs_df is not None and not outcome_logs_df.empty:
        tmp = outcome_logs_df.copy()
        if "dorm_id" in tmp.columns:
            dorm_out = tmp[tmp["dorm_id"].astype(str) == str(selected_dorm)].copy()

    has_outcome_data = dorm_out is not None and not dorm_out.empty
    if not has_outcome_data and not is_sim_mode:
        st.warning("该宿舍暂无 outcome 数据，无法展示首页核心指标。")
        return
    elif not has_outcome_data and is_sim_mode:
        st.info("模拟宿舍测试模式：暂无 outcome 数据，跳过基线图展示。")
    

    if has_outcome_data:
        wm_all = st.session_state.get("weekly_metrics_by_dorm", {})
        wm = wm_all.get(str(selected_dorm), None)

        if wm is not None and not mobile:
            actual = float(pd.to_numeric(wm.get("actual_sum", np.nan), errors="coerce"))
            baseline = float(pd.to_numeric(wm.get("baseline_sum", np.nan), errors="coerce"))
            savings = float(pd.to_numeric(wm.get("reward_sum", np.nan), errors="coerce"))
            rate = savings / baseline if baseline > 1e-9 else np.nan
        else:
            actual_col = ("kwh" if "kwh" in dorm_out.columns
                        else ("energy_kwh" if "energy_kwh" in dorm_out.columns else None))
            baseline_col = "baseline_pred" if "baseline_pred" in dorm_out.columns else None

            if actual_col is None or baseline_col is None:
                if not is_sim_mode:
                    st.warning("缺少实际值或 baseline_pred，无法展示能耗概览。")
                    return
                actual = baseline = savings = None
                rate = np.nan
            else:
                tmp = dorm_out.copy()
                tcol = "timestamp_hour" if "timestamp_hour" in tmp.columns else "timestamp"
                tmp["_t"] = pd.to_datetime(tmp[tcol], errors="coerce")

                end_dt = pd.to_datetime(selected_date_str) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                start_dt = end_dt - pd.Timedelta(days=6)

                wk = tmp[(tmp["_t"] >= start_dt) & (tmp["_t"] <= end_dt)].copy()

                actual = float(pd.to_numeric(wk[actual_col], errors="coerce").fillna(0).sum())
                baseline = float(pd.to_numeric(wk[baseline_col], errors="coerce").fillna(0).sum())
                savings = baseline - actual
                rate = savings / baseline if baseline > 1e-9 else np.nan
    else:
        actual = baseline = savings = None
        rate = np.nan
        

    # ── CS/BS/cluster 计算（不变） ───────────────────────────────
    latest_score = _latest_score_for_dorm(selected_dorm)
    if latest_score is not None and latest_score.get("CS") is not None:
        cs, cs_source = float(latest_score.get("CS", 0.5)), "latest_score"
    else:
        cs, cs_source = float(DORM_CS_PROFILE.get(selected_dorm, 0.5)), "dorm_profile"

    if is_sim_mode:
        if latest_score is not None and latest_score.get("BS_energy_week") is not None:
            bs, bs_source = float(latest_score["BS_energy_week"]), "latest_score_sim"
        elif has_outcome_data:
            bs, bs_source = compute_BS_energy_week(dorm_out, tau=tau), "outcome_data_fallback"
        else:
            bs, bs_source = 0.5, "fallback_default"
    else:
        if latest_score is not None and latest_score.get("BS_energy_week") is not None:
            bs, bs_source = float(latest_score["BS_energy_week"]), "latest_score"
        else:
            bs, bs_source = compute_BS_energy_week(dorm_out, tau=tau), "outcome_data"

    cluster_type, dist2 = classify_cluster(cs, bs)
    gap = cs - bs
    cluster_meta = _cluster_display_meta(cluster_type)

    # ── 将当前宿舍的 score 立即写回 session_state，供 Messages 自动生成使用 ──
    scores = st.session_state.get("scores", [])

    current_score = {
        "dorm_id": str(selected_dorm),
        "ts": ts_now,
        "CS": float(cs),
        "BS_energy_week": float(bs),
        "cluster_type": str(cluster_type),
        "cluster_dist2": float(dist2),
        "tau": float(tau),
    }

    # 同一宿舍 + 同一天则覆盖，避免重复堆积
    updated = False
    for i, rec in enumerate(scores):
        if not isinstance(rec, dict):
            continue
        if (
            str(rec.get("dorm_id", "")) == str(selected_dorm)
            and str(rec.get("ts", "")).startswith(selected_date_str)
        ):
            scores[i] = current_score
            updated = True
            break

    if not updated:
        scores.append(current_score)

    st.session_state["scores"] = scores
    st.session_state["current_dorm_for_messages"] = selected_dorm

    # 提前计算排行榜位置，用于顶部英雄卡显示"超过X%"
    _surpass_pct_for_hero: float | None = None
    try:
        _wm = st.session_state.get("weekly_metrics_by_dorm", {})
        _rank_df_hero = _build_leaderboard_df(_wm)
        if not _rank_df_hero.empty and str(selected_dorm) in _rank_df_hero["dorm_id"].astype(str).tolist():
            _cur = _rank_df_hero[_rank_df_hero["dorm_id"].astype(str) == str(selected_dorm)].iloc[0]
            _total = len(_rank_df_hero)
            _rank = int(_cur["rank"])
            _surpass_pct_for_hero = (_total - _rank) / max(_total - 1, 1)
    except Exception:
        _surpass_pct_for_hero = None
        
    # ── LinUCB 选臂（不变） ──────────────────────────────────────
    cluster_arms_config = st.session_state.get("cluster_arms_config")
    active_cfg = cluster_arms_config if cluster_arms_config is not None else DEFAULT_CLUSTER_ARMS
    candidate_arms = list(active_cfg.get(cluster_type, []))

    recent_diff_3h = 0.0
    is_weekend = int(pd.Timestamp(latest_ts).dayofweek >= 5)
    x = parse_state_vector(cs, bs, recent_diff_3h, is_weekend)

    linucb_params = st.session_state.get("linucb_params")
    best_arm, best_detail, details, choose_err = None, {"score": 0.0, "mu": 0.0, "bonus": 0.0}, [], None
    if candidate_arms and linucb_params is not None:
        try:
            best_arm, best_detail, details = linucb_choose_arm(linucb_params, candidate_arms, x)
        except Exception as e:
            choose_err = str(e)

    strategy_meta = _arm_display_meta(best_arm)
    if best_arm is not None:
        reason_text = _build_strategy_reason_text(cluster_type, best_arm, float(cs), float(bs))
    elif choose_err:
        reason_text = f"LinUCB 评分失败：{choose_err}"
    elif not candidate_arms:
        reason_text = "当前群体尚未配置候选 arms，请前往 Admin → 配置中心。"
    elif linucb_params is None:
        reason_text = "linucb_params 尚未初始化。"
    else:
        reason_text = "当前暂未生成推荐策略。"

    # ── 消息生成 ─────────────────────────────────────────
    tip_msg = push_msg = None
    try:
        tip_msg = _build_home_tip_message(selected_dorm, cluster_type, is_sim_mode, ts_now)
    except Exception as e:
        st.warning(f"P 小贴士生成失败：{e}")
    try:
        push_msg = _build_home_push_message(selected_dorm, cluster_type, is_sim_mode, ts_now)
    except Exception as e:
        st.warning(f"R 推送生成失败：{e}")

    if mobile:
        # ══════════════════════════════════════════════════════════
        # 手机端：三张独立 iframe 卡片
        # ══════════════════════════════════════════════════════════
        with st.container(key="home_mobile_cards"):
            _render_mobile_hero_card(
                selected_dorm=selected_dorm,
                actual=actual,
                rate=rate,
                dorm_out=dorm_out if has_outcome_data else None,
            )
            _render_mobile_cluster_card(
                cluster_type=cluster_type,
                cluster_meta=cluster_meta,
                cs=cs,
                bs=bs,
                gap=gap,
            )
            _render_mobile_social_card(selected_dorm=selected_dorm)

    else:
        # ══════════════════════════════════════════════════════════
        # 桌面端：原有渲染路径（完全不变）
        # ══════════════════════════════════════════════════════════

        # ② ③ 主视觉区：左英雄卡 + 右群体/评分双层卡
        _render_home_hero_grid(
            selected_dorm=selected_dorm,
            actual=actual,
            baseline=baseline,
            rate=rate,
            cluster_type=cluster_type,
            cluster_meta=cluster_meta,
            cs=cs,
            bs=bs,
            gap=gap,
            dorm_out=dorm_out if has_outcome_data else None,
        )

        # ④ 三指标横排（用电量 / 节电率 / 排名）
        _render_trend_strip(
            selected_dorm=selected_dorm,
            actual=actual,
            rate=rate,
            dorm_out=dorm_out if has_outcome_data else None,
        )

        # ⑥ 节能排行榜
        _render_social_leaderboard(selected_dorm=selected_dorm, top_n=3)

        # ⑦ 本周干预日历
        _render_week_calendar(
            selected_dorm=selected_dorm,
            selected_date_str=selected_date_str,
            dorm_out=dorm_out if has_outcome_data else None,
        )

        # ⑧ 调试信息
        with st.expander("🔧 调试信息", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                st.write({"CS": round(float(cs), 4), "BS": round(float(bs), 4),
                          "Gap": round(float(gap), 4), "cluster_type": cluster_type,
                          "cluster_dist2": round(float(dist2), 6), "tau": round(float(tau), 2),
                          "CS_source": cs_source, "BS_source": bs_source})
            with c2:
                st.write({"candidate_arms": candidate_arms, "best_arm": best_arm,
                          "alpha_ucb": float(alpha_ucb), "choose_err": choose_err})
            if details:
                st.dataframe(pd.DataFrame(details), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# 手机端专用渲染函数
# 每个函数输出一个完整的 components.html iframe，
# 不依赖外部 st.markdown div 拼接，避免 Streamlit 截断问题。
# 所有业务数据计算保持在 render_home_page 中不变，这里只负责展示。
# ══════════════════════════════════════════════════════════════════════════════

def _render_mobile_hero_card(selected_dorm, actual, rate, dorm_out):
    """
    手机端 F·数据反馈卡。
    - 深绿底色，kWh 大字 + 节电率 badge
    - Canvas 趋势图：复用原始小时级数据，保留真实曲线形态 + baseline 虚线
    - 全部在单个 iframe 内，无外部 st.markdown
    """
    import json
    import streamlit.components.v1 as components

    # ── 数值格式化 ──────────────────────────────────────────────────
    actual_text = "N/A" if actual is None or (isinstance(actual, float) and pd.isna(actual)) else f"{float(actual):.1f}"

    if rate is None or (isinstance(rate, float) and pd.isna(rate)):
        rate_text = "暂无数据"
        rate_bg, rate_fg = "rgba(255,255,255,0.15)", "rgba(255,255,255,0.7)"
        rate_arrow = ""
    elif float(rate) > 0.001:
        rate_text = f"{float(rate)*100:.1f}%"
        rate_bg, rate_fg = "rgba(74,222,128,0.22)", "#4ADE80"
        rate_arrow = "↓ 节约 "
    elif float(rate) < -0.001:
        rate_text = f"{abs(float(rate))*100:.1f}%"
        rate_bg, rate_fg = "rgba(251,191,36,0.22)", "#FCD34D"
        rate_arrow = "↑ 超耗 "
    else:
        rate_text = "0.0%"
        rate_bg, rate_fg = "rgba(255,255,255,0.15)", "rgba(255,255,255,0.7)"
        rate_arrow = ""

    # ── 趋势图数据：复用原有小时数据构建逻辑，保持真实性 ──────────
    chart_points = []
    if dorm_out is not None and not dorm_out.empty:
        dplot = dorm_out.copy()
        tcol = "timestamp_hour" if "timestamp_hour" in dplot.columns else "timestamp"
        dplot["_t"] = pd.to_datetime(dplot.get(tcol), errors="coerce")
        if "kwh" not in dplot.columns and "energy_kwh" in dplot.columns:
            dplot["kwh"] = pd.to_numeric(dplot["energy_kwh"], errors="coerce")
        else:
            dplot["kwh"] = pd.to_numeric(dplot.get("kwh"), errors="coerce")
        if "baseline_pred" in dplot.columns:
            dplot["_bl"] = pd.to_numeric(dplot["baseline_pred"], errors="coerce")
        else:
            dplot["_bl"] = float("nan")
        if "_t" in dplot.columns and "kwh" in dplot.columns:
            dplot = dplot.dropna(subset=["_t", "kwh"]).sort_values("_t")
            # 手机端取最近7天数据，降采样到每6小时一个点减少渲染压力，保留曲线形态
            recent = dplot.tail(7 * 24)
            # 每6条取均值（保持走势真实，减少点数）
            step = max(1, len(recent) // 42)
            for i in range(0, len(recent), step):
                chunk = recent.iloc[i:i+step]
                t_mid = chunk["_t"].iloc[len(chunk)//2]
                v_mean = float(chunk["kwh"].mean())
                b_vals = chunk["_bl"].dropna()
                b_mean = float(b_vals.mean()) if not b_vals.empty else None
                chart_points.append({
                    "t": str(t_mid)[:16],
                    "v": round(v_mean, 3),
                    "b": round(b_mean, 3) if b_mean is not None else None,
                })

    pts_json = json.dumps(chart_points, ensure_ascii=False)
    has_chart = len(chart_points) > 0
    has_baseline = has_chart and any(p["b"] is not None for p in chart_points)
    chart_height = 130 if has_chart else 0
    card_height = 220 + chart_height

    # ── baseline 图例 HTML ──────────────────────────────────────────
    legend_html = ""
    if has_baseline:
        legend_html = (
            '<div style="display:flex;gap:12px;align-items:center;'
            'margin-top:6px;padding:0 14px;">'
            '<span style="display:flex;align-items:center;gap:4px;font-size:10px;color:rgba(255,255,255,0.5);">'
            '<span style="display:inline-block;width:16px;height:2px;background:#4ADE80;border-radius:1px;"></span>'
            '实际用电</span>'
            '<span style="display:flex;align-items:center;gap:4px;font-size:10px;color:rgba(255,255,255,0.5);">'
            '<span style="display:inline-block;width:16px;height:2px;'
            'background:rgba(255,255,255,0.4);border-radius:1px;'
            'background-image:repeating-linear-gradient(to right,rgba(255,255,255,0.45) 0,rgba(255,255,255,0.45) 4px,transparent 4px,transparent 8px);'
            'background-size:8px 2px;"></span>'
            '参考基线</span>'
            '</div>'
        )

    canvas_html = ""
    if has_chart:
        canvas_html = (
            '<canvas id="mc" style="width:100%;display:block;margin-top:8px;'
            'border-radius:0 0 18px 18px;"></canvas>'
        )

    full_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
*{{box-sizing:border-box;margin:0;padding:0;}}
html,body{{
  width:100%;
  min-height:100%;
  background:transparent;
  font-family:-apple-system,'Noto Sans SC',sans-serif;
  overflow:hidden;
  box-sizing:border-box;
}}
.card{{
  background:linear-gradient(160deg,#0A2E1A 0%,#0F4023 55%,#185C32 100%);
  border-radius:20px;
  padding:18px 16px 0;
  width:100%;
  max-width:100%;
  min-height:{card_height}px;
  display:flex;
  flex-direction:column;
  box-sizing:border-box;
  overflow:hidden;
}}
.tag{{
  font-size:10px;font-weight:800;letter-spacing:0.8px;
  color:rgba(255,255,255,0.45);
  text-transform:uppercase;
  margin-bottom:10px;
}}
.kwh-row{{
  display:flex;align-items:baseline;gap:6px;
  margin-bottom:4px;
}}
.kwh-val{{
  font-size:52px;font-weight:900;
  color:#fff;letter-spacing:-2px;line-height:1;
}}
.kwh-unit{{
  font-size:14px;font-weight:600;color:rgba(255,255,255,0.55);
}}
.period{{
  font-size:11px;font-weight:600;color:rgba(255,255,255,0.4);
  margin-bottom:12px;
}}
.rate-badge{{
  display:inline-flex;align-items:center;
  padding:5px 11px;border-radius:8px;
  font-size:12px;font-weight:800;
  background:{rate_bg};
  color:{rate_fg};
  margin-bottom:6px;
  width:fit-content;
}}
.chart-wrap{{
  flex:1;min-height:{chart_height}px;
  margin:4px 0 0;
  position:relative;
  width:100%;
  min-width:0;
  overflow:hidden;
}}
</style>
</head>
<body>
<div class="card">
  <div class="tag">F · 数据反馈</div>
  <div class="kwh-row">
    <span class="kwh-val">{actual_text}</span>
    <span class="kwh-unit">kWh</span>
  </div>
  <div class="period">/ 本周</div>
  <div class="rate-badge">{rate_arrow}{rate_text}</div>
  {legend_html}
  <div class="chart-wrap">
    {canvas_html}
  </div>
</div>
<script>
(function(){{
  const PTS = {pts_json};
  if (!PTS.length) return;
  const canvas = document.getElementById('mc');
  if (!canvas) return;
  const wrap = canvas.parentElement;

  function draw() {{
    const dpr = window.devicePixelRatio || 1;
    const W = wrap.clientWidth || 340;
    const H = {chart_height};
    canvas.width  = W * dpr;
    canvas.height = H * dpr;
    canvas.style.width  = W + 'px';
    canvas.style.height = H + 'px';
    const ctx = canvas.getContext('2d');
    ctx.setTransform(1,0,0,1,0,0);
    ctx.scale(dpr, dpr);
    ctx.clearRect(0,0,W,H);

    const vals = PTS.map(p => p.v);
    const bls  = PTS.map(p => p.b);
    const hasBl = bls.some(b => b !== null && b !== undefined);

    const allVals = hasBl ? [...vals, ...bls.filter(b => b!=null)] : vals;
    const rawMin = Math.min(...allVals);
    const rawMax = Math.max(...allVals);
    const span   = rawMax - rawMin || 1;
    const minV   = rawMin - span * 0.10;
    const maxV   = rawMax + span * 0.10;
    const n = PTS.length;
    const pad = {{l:0,r:0,t:6,b:0}};
    const cw = W - pad.l - pad.r;
    const ch = H - pad.t - pad.b;

    function xp(i) {{ return n===1 ? W/2 : pad.l + (i/(n-1))*cw; }}
    function yp(v) {{ return pad.t + (1-(v-minV)/(maxV-minV))*ch; }}

    // ① 节省/超耗色块（baseline 存在时）
    if (hasBl) {{
      for (let i=0;i<n-1;i++) {{
        const bC=bls[i], bN=bls[i+1];
        if (bC==null||bN==null) continue;
        const saving=((vals[i]+vals[i+1])/2)<=((bC+bN)/2);
        ctx.beginPath();
        ctx.moveTo(xp(i),yp(vals[i]));
        ctx.lineTo(xp(i+1),yp(vals[i+1]));
        ctx.lineTo(xp(i+1),yp(bN));
        ctx.lineTo(xp(i),yp(bC));
        ctx.closePath();
        ctx.fillStyle = saving ? 'rgba(74,222,128,0.18)' : 'rgba(251,191,36,0.18)';
        ctx.fill();
      }}
    }}

    // ② actual 渐变面积
    const grad = ctx.createLinearGradient(0, pad.t, 0, H);
    grad.addColorStop(0,'rgba(74,222,128,0.25)');
    grad.addColorStop(1,'rgba(74,222,128,0)');
    ctx.beginPath();
    ctx.moveTo(xp(0),yp(vals[0]));
    for(let i=1;i<n;i++) ctx.lineTo(xp(i),yp(vals[i]));
    ctx.lineTo(xp(n-1),H);
    ctx.lineTo(xp(0),H);
    ctx.closePath();
    ctx.fillStyle=grad; ctx.fill();

    // ③ baseline 虚线
    if (hasBl) {{
      ctx.beginPath(); let started=false;
      for(let i=0;i<n;i++) {{
        if(bls[i]==null) {{started=false;continue;}}
        if(!started) {{ctx.moveTo(xp(i),yp(bls[i]));started=true;}}
        else ctx.lineTo(xp(i),yp(bls[i]));
      }}
      ctx.setLineDash([5,4]);
      ctx.strokeStyle='rgba(255,255,255,0.38)';
      ctx.lineWidth=1.5; ctx.lineJoin='round'; ctx.stroke();
      ctx.setLineDash([]);
    }}

    // ④ actual 实线
    ctx.beginPath();
    ctx.moveTo(xp(0),yp(vals[0]));
    for(let i=1;i<n;i++) ctx.lineTo(xp(i),yp(vals[i]));
    ctx.strokeStyle='#4ADE80'; ctx.lineWidth=2.0;
    ctx.lineJoin='round'; ctx.lineCap='round'; ctx.stroke();
  }}

  draw();
  window.addEventListener('resize', draw);
}})();
</script>
</body>
</html>"""

    components.html(full_html, height=card_height + 16, scrolling=False)


def _render_mobile_cluster_card(cluster_type, cluster_meta, cs, bs, gap):
    """
    手机端群体识别 + CS/BS 评分卡。
    - 白底卡片，群体标签 + 类型名称 + CS/BS 双进度条 + gap 状态标签
    - 删去象限图，保留文字描述
    - 单个 iframe，无外部 div 拼接
    """
    import streamlit.components.v1 as components

    cluster_tag   = cluster_meta.get("tag", "")
    cluster_title = cluster_meta.get("title", "")
    cluster_desc  = cluster_meta.get("desc", "")

    cs_w = max(4, min(100, int(float(cs) * 100)))
    bs_w = max(4, min(100, int(float(bs) * 100)))
    cs_pct = f"{float(cs)*100:.0f}%"
    bs_pct = f"{float(bs)*100:.0f}%"

    if float(gap) > 0.10:
        gap_bg, gap_bd, gap_fg = "#FEF9EC", "#F6CC5A", "#7A4F00"
        gap_icon, gap_label = "↑", "意识高于行为"
    elif float(gap) < -0.10:
        gap_bg, gap_bd, gap_fg = "#F0FDF4", "#86EFAC", "#15803D"
        gap_icon, gap_label = "✓", "行为表现积极"
    else:
        gap_bg, gap_bd, gap_fg = "#F5F7F5", "#D1D9D4", "#4B5563"
        gap_icon, gap_label = "●", "意识与行为均衡"

    # CS/BS 决定象限文字（替代象限图）
    if float(cs) >= 0.67 and float(bs) >= 0.67:
        quad_label = "高意识 · 高行为"
        quad_color = "#15803D"
    elif float(cs) >= 0.67 and float(bs) < 0.67:
        quad_label = "高意识 · 低行为"
        quad_color = "#B45309"
    elif float(cs) < 0.67 and float(bs) >= 0.67:
        quad_label = "低意识 · 高行为"
        quad_color = "#1D4ED8"
    else:
        quad_label = "低意识 · 低行为"
        quad_color = "#6B7280"

    full_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
*{{box-sizing:border-box;margin:0;padding:0;}}
html,body{{
  width:100%;
  min-height:100%;
  background:transparent;
  font-family:-apple-system,'Noto Sans SC',sans-serif;
  overflow:hidden;
  box-sizing:border-box;
}}
.card{{
  background:#fff;
  border:1px solid #E6EDE8;
  border-radius:18px;
  padding:16px 16px 14px;
  width:100%;
  max-width:100%;
  box-sizing:border-box;
  overflow:hidden;
}}
.tag{{
  display:inline-block;
  font-size:10px;font-weight:800;letter-spacing:0.5px;
  color:#15803D;background:#F0FDF4;
  border-radius:6px;padding:3px 9px;
  margin-bottom:10px;
}}
.cluster-label{{
  font-size:11px;font-weight:500;color:#9CA3AF;
  margin-bottom:3px;letter-spacing:0.2px;
}}
.cluster-title{{
  font-size:16px;font-weight:800;color:#0A1A0F;
  line-height:1.3;margin-bottom:10px;
}}
.divider{{
  height:1px;background:#EEF2EE;margin-bottom:12px;
}}
.score-row{{
  display:flex;gap:12px;margin-bottom:12px;
}}
.score-item{{
  flex:1;
  background:#F8FAF8;border-radius:12px;padding:10px 12px;
}}
.sc-label{{
  font-size:10px;font-weight:700;margin-bottom:4px;letter-spacing:0.3px;
}}
.sc-label.cs{{color:#3B82F6;}}
.sc-label.bs{{color:#22C55E;}}
.sc-val{{
  font-size:22px;font-weight:900;line-height:1;margin-bottom:6px;
}}
.sc-val.cs{{color:#1D4ED8;}}
.sc-val.bs{{color:#15803D;}}
.bar-bg{{
  height:5px;background:#EEF2EE;border-radius:999px;overflow:hidden;
}}
.bar-fg{{
  height:100%;border-radius:999px;
}}
.bar-fg.cs{{background:#3B82F6;}}
.bar-fg.bs{{background:#22C55E;}}
.gap-tag{{
  display:inline-flex;align-items:center;gap:5px;
  padding:5px 10px;border-radius:8px;
  border:1px solid {gap_bd};
  background:{gap_bg};color:{gap_fg};
  font-size:10px;font-weight:700;
  margin-bottom:10px;
}}
.quad-line{{
  font-size:11px;font-weight:700;margin-bottom:4px;
}}
.desc{{
  font-size:11px;color:#6B7280;line-height:1.65;
}}
</style>
</head>
<body>
<div class="card">
  <div class="tag">{cluster_tag}</div>
  <div class="cluster-label">群体类别 {cluster_type}</div>
  <div class="cluster-title">{cluster_title}</div>
  <div class="divider"></div>
  <div class="score-row">
    <div class="score-item">
      <div class="sc-label cs">CS · 意识</div>
      <div class="sc-val cs">{cs_pct}</div>
      <div class="bar-bg"><div class="bar-fg cs" style="width:{cs_w}%"></div></div>
    </div>
    <div class="score-item">
      <div class="sc-label bs">BS · 行为</div>
      <div class="sc-val bs">{bs_pct}</div>
      <div class="bar-bg"><div class="bar-fg bs" style="width:{bs_w}%"></div></div>
    </div>
  </div>
  <div class="gap-tag">{gap_icon}&nbsp;{gap_label}</div>
  <div class="quad-line" style="color:{quad_color};">{quad_label}</div>
  <div class="desc">{cluster_desc}</div>
</div>
</body>
</html>"""

    components.html(full_html, height=300, scrolling=False)


def _render_mobile_social_card(selected_dorm: str):
    """
    手机端 S·社会比较卡。
    - 显示：同辈排名、超越比例、相对均值差距
    - 排行榜前3名 + 我的成绩行
    - 单个 iframe，无外部 div 拼接
    - 数据来源：session_state["weekly_metrics_by_dorm"] + session_state["scores"]（与原 _render_social_leaderboard 完全一致）
    """
    import streamlit.components.v1 as components

    weekly_metrics_by_dorm = st.session_state.get("weekly_metrics_by_dorm", {})
    rank_df = _build_leaderboard_df(weekly_metrics_by_dorm)

    # ── 数据不足时的占位卡 ──────────────────────────────────────────
    if rank_df.empty or str(selected_dorm) not in rank_df["dorm_id"].astype(str).tolist():
        no_data_html = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
*{box-sizing:border-box;margin:0;padding:0;}
html,body{width:100%;min-height:100%;background:transparent;
  font-family:-apple-system,'Noto Sans SC',sans-serif;overflow:hidden;box-sizing:border-box;}
.card{background:#fff;border:1px solid #E6EDE8;border-radius:18px;
  padding:16px;width:100%;max-width:100%;box-sizing:border-box;min-height:80px;display:flex;align-items:center;gap:10px;overflow:hidden;}
.tag{display:inline-block;font-size:10px;font-weight:800;
  color:#F97316;background:#FFF7ED;border-radius:6px;padding:3px 9px;white-space:nowrap;}
.hint{font-size:12px;color:#9CA3AF;line-height:1.5;}
</style></head>
<body>
<div class="card">
  <div class="tag">S · 社会比较</div>
  <div class="hint">当前暂无足够的周度数据</div>
</div>
</body></html>"""
        components.html(no_data_html, height=104, scrolling=False)
        return

    rank_df["dorm_id"] = rank_df["dorm_id"].astype(str)
    current_row  = rank_df[rank_df["dorm_id"] == str(selected_dorm)].iloc[0]
    total_n      = len(rank_df)
    current_rank = int(current_row["rank"])
    current_rate = float(current_row["saving_rate"])

    # ── 同辈比较（与 _render_social_leaderboard 逻辑完全一致） ─────
    score_map = {}
    for rec in st.session_state.get("scores", []):
        if not isinstance(rec, dict):
            continue
        did = str(rec.get("dorm_id", "")).strip()
        if not did:
            continue
        if did not in score_map or str(rec.get("ts", "")) > str(score_map[did].get("ts", "")):
            score_map[did] = rec

    cur_cluster = str(score_map.get(str(selected_dorm), {}).get("cluster_type", "")).upper()
    rank_df["peer_cluster"] = rank_df["dorm_id"].map(
        lambda x: str(score_map.get(str(x), {}).get("cluster_type", "")).upper()
    )

    peer_surpass_pct = None
    peer_delta_text  = "—"
    peer_rank_str    = f"{current_rank}/{total_n}"
    peer_delta_color = "#6B7280"
    surpass_headline = f"楼层第 {current_rank} 名"

    if cur_cluster:
        peer_df = rank_df[rank_df["peer_cluster"] == cur_cluster].copy()
        if not peer_df.empty:
            peer_df = peer_df.sort_values(
                ["saving_rate", "saving_kwh"], ascending=[False, False]
            ).reset_index(drop=True)
            peer_df["peer_rank"] = np.arange(1, len(peer_df) + 1)
            my_peer = peer_df[peer_df["dorm_id"] == str(selected_dorm)]
            if not my_peer.empty:
                my_peer_row = my_peer.iloc[0]
                peer_rank   = int(my_peer_row["peer_rank"])
                peer_total  = len(peer_df)
                peer_surpass_pct = (peer_total - peer_rank) / max(peer_total - 1, 1) if peer_total > 1 else 1.0
                peer_avg    = float(peer_df["saving_rate"].mean())
                peer_delta  = float(current_rate - peer_avg)
                sign = "+" if peer_delta >= 0 else ""
                peer_delta_text  = f"{sign}{peer_delta*100:.1f}%"
                peer_delta_color = "#16A34A" if peer_delta > 0.001 else ("#D97706" if peer_delta < -0.001 else "#6B7280")
                peer_rank_str    = f"{peer_rank}/{peer_total}"
                if peer_surpass_pct is not None:
                    surpass_headline = f"节电 {_safe_pct_text(current_rate)} · 超越 {peer_surpass_pct*100:.0f}% 同学"

    # ── 节电率格式化 ───────────────────────────────────────────────
    my_rate_cls = "#16A34A" if current_rate > 0.001 else ("#D97706" if current_rate < 0 else "#6B7280")
    my_pct_text = _safe_pct_text(current_rate)

    # ── 前3名行 HTML ───────────────────────────────────────────────
    top3_rows_html = ""
    top3 = rank_df.head(3).to_dict("records")
    medal = ["🥇", "🥈", "🥉"]
    for i, row in enumerate(top3):
        is_me = (str(row["dorm_id"]) == str(selected_dorm))
        rv = float(row["saving_rate"])
        rc = "#16A34A" if rv > 0.001 else ("#D97706" if rv < 0 else "#6B7280")
        me_style = "font-weight:800;background:#F0FDF4;" if is_me else ""
        me_dot   = '<span style="display:inline-block;width:6px;height:6px;border-radius:50%;background:#22C55E;margin-right:4px;"></span>' if is_me else ""
        top3_rows_html += (
            f'<div style="display:flex;align-items:center;padding:9px 0;'
            f'border-bottom:1px solid #F0F4F1;{me_style}">'
            f'  <span style="font-size:14px;width:22px;flex-shrink:0;">{medal[i]}</span>'
            f'  {me_dot}'
            f'  <span style="flex:1;font-size:13px;font-weight:600;color:#1F2937;'
            f'overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{row["dorm_id"]}</span>'
            f'  <span style="font-size:14px;font-weight:900;color:{rc};">{_safe_pct_text(rv)}</span>'
            f'</div>'
        )

    # 当前宿舍不在前3时追加我的成绩行
    if current_rank > 3:
        top3_rows_html += (
            f'<div style="display:flex;align-items:center;padding:9px 0;'
            f'background:#F0FDF4;border-radius:8px;margin-top:4px;padding-left:6px;padding-right:6px;">'
            f'  <span style="font-size:13px;font-weight:800;width:22px;color:#9CA3AF;flex-shrink:0;">{current_rank}</span>'
            f'  <span style="display:inline-block;width:6px;height:6px;border-radius:50%;background:#22C55E;margin-right:4px;flex-shrink:0;"></span>'
            f'  <span style="flex:1;font-size:13px;font-weight:800;color:#0A1A0F;'
            f'overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{selected_dorm}</span>'
            f'  <span style="font-size:14px;font-weight:900;color:{my_rate_cls};">{my_pct_text}</span>'
            f'</div>'
        )

    card_height = 210 + len(top3) * 38 + (38 if current_rank > 3 else 0)

    full_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
*{{box-sizing:border-box;margin:0;padding:0;}}
html,body{{
  width:100%;
  min-height:100%;
  background:transparent;
  font-family:-apple-system,'Noto Sans SC',sans-serif;
  overflow:hidden;
  box-sizing:border-box;
}}
.card{{
  background:#fff;border:1px solid #E6EDE8;
  border-radius:18px;padding:16px;width:100%;max-width:100%;
  box-sizing:border-box;overflow:hidden;
}}
.top-row{{display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;}}
.tag{{
  display:inline-block;font-size:10px;font-weight:800;letter-spacing:0.5px;
  color:#F97316;background:#FFF7ED;border-radius:6px;padding:3px 9px;
}}
.rank-link{{font-size:11px;font-weight:600;color:#9CA3AF;}}
.headline{{
  font-size:15px;font-weight:800;color:#0A1A0F;
  line-height:1.4;margin-bottom:10px;
}}
.meta-row{{
  display:flex;gap:8px;margin-bottom:14px;
}}
.meta-chip{{
  flex:1;background:#F7FBF8;border-radius:10px;padding:8px 10px;
}}
.meta-chip-label{{font-size:9px;font-weight:700;color:#9CA3AF;margin-bottom:2px;letter-spacing:0.3px;}}
.meta-chip-val{{font-size:15px;font-weight:900;}}
.divider{{height:1px;background:#EEF2EE;margin-bottom:10px;}}
.lb-title{{font-size:10px;font-weight:800;color:#9CA3AF;letter-spacing:0.7px;
  text-transform:uppercase;margin-bottom:6px;}}
</style>
</head>
<body>
<div class="card">
  <div class="top-row">
    <div class="tag">S · 社会比较</div>
    <div class="rank-link">楼层第 {current_rank} 名 &rsaquo;</div>
  </div>
  <div class="headline">{surpass_headline}</div>
  <div class="meta-row">
    <div class="meta-chip">
      <div class="meta-chip-label">同辈排名</div>
      <div class="meta-chip-val" style="color:#0A1A0F;">{peer_rank_str}</div>
    </div>
    <div class="meta-chip">
      <div class="meta-chip-label">相对均值</div>
      <div class="meta-chip-val" style="color:{peer_delta_color};">{peer_delta_text}</div>
    </div>
    <div class="meta-chip">
      <div class="meta-chip-label">本周节电</div>
      <div class="meta-chip-val" style="color:{my_rate_cls};">{my_pct_text}</div>
    </div>
  </div>
  <div class="divider"></div>
  <div class="lb-title">节能排行榜</div>
  {top3_rows_html}
</div>
</body>
</html>"""

    components.html(full_html, height=card_height + 12, scrolling=False)
