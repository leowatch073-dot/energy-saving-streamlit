from __future__ import annotations

from datetime import datetime, timedelta

import streamlit as st


CHECKIN_XP = 10

BADGE_DEFS = [
    {
        "id": "first_checkin",
        "icon": "🌟",
        "name": "节能先锋",
        "bg": "#FFF8DC",
        "desc": "首次参与干预",
    },
    {
        "id": "green_habit",
        "icon": "🌿",
        "name": "绿色习惯",
        "bg": "#EAF7EF",
        "desc": "连续节能 5 天",
    },
    {
        "id": "data_expert",
        "icon": "📊",
        "name": "数据达人",
        "bg": "#EFF6FF",
        "desc": "累计获得 30 XP",
    },
    {
        "id": "energy_master",
        "icon": "🏆",
        "name": "节能达人",
        "bg": "#F3F4F6",
        "desc": "连续节能 10 天",
    },
]


def current_demo_date_str() -> str:
    return str(st.session_state.get("business_date_str", datetime.now().strftime("%Y-%m-%d")))


def init_user_progress_state() -> None:
    st.session_state.setdefault("user_progress_by_dorm", {})
    st.session_state.setdefault("tasks_checkin_log", [])
    st.session_state.setdefault("today_checked_in", False)
    st.session_state.setdefault("streak_days", 0)
    st.session_state.setdefault("weekly_checkins", [])
    st.session_state.setdefault("earned_xp", 0)
    st.session_state.setdefault("total_xp", 0)
    st.session_state.setdefault("unlocked_badges", [])


def _parse_date(date_str: str):
    return datetime.strptime(str(date_str)[:10], "%Y-%m-%d").date()


def _sort_dates(dates) -> list[str]:
    valid = []
    for item in dates:
        try:
            valid.append(_parse_date(str(item)).strftime("%Y-%m-%d"))
        except Exception:
            continue
    return sorted(set(valid))


def _legacy_checkin_dates(dorm_id: str) -> list[str]:
    logs = st.session_state.get("tasks_checkin_log", [])
    return _sort_dates(
        r.get("date", "")
        for r in logs
        if isinstance(r, dict) and str(r.get("dorm_id", "")) == str(dorm_id)
    )


def _calc_streak(dates: set[str], date_str: str) -> int:
    try:
        cursor = _parse_date(date_str)
    except Exception:
        return 0

    streak = 0
    for offset in range(365):
        d = (cursor - timedelta(days=offset)).strftime("%Y-%m-%d")
        if d in dates:
            streak += 1
        else:
            break
    return streak


def _weekly_dates(dates: set[str], date_str: str) -> list[str]:
    try:
        end = _parse_date(date_str)
    except Exception:
        return []
    week = [
        (end - timedelta(days=offset)).strftime("%Y-%m-%d")
        for offset in range(6, -1, -1)
    ]
    return [d for d in week if d in dates]


def _unlocked_badges(streak_days: int, total_xp: int, checkin_count: int) -> list[str]:
    unlocked = []
    if checkin_count >= 1:
        unlocked.append("first_checkin")
    if streak_days >= 5:
        unlocked.append("green_habit")
    if total_xp >= 30:
        unlocked.append("data_expert")
    if streak_days >= 10:
        unlocked.append("energy_master")
    return unlocked


def _mirror_current_progress(progress: dict) -> None:
    st.session_state["today_checked_in"] = bool(progress.get("today_checked_in", False))
    st.session_state["streak_days"] = int(progress.get("streak_days", 0))
    st.session_state["weekly_checkins"] = list(progress.get("weekly_checkins", []))
    st.session_state["earned_xp"] = int(progress.get("earned_xp", 0))
    st.session_state["total_xp"] = int(progress.get("total_xp", 0))
    st.session_state["unlocked_badges"] = list(progress.get("unlocked_badges", []))


def get_user_progress(dorm_id: str, date_str: str | None = None) -> dict:
    init_user_progress_state()
    dorm_key = str(dorm_id or "—")
    date_key = str(date_str or current_demo_date_str())[:10]

    by_dorm = st.session_state["user_progress_by_dorm"]
    progress = dict(by_dorm.get(dorm_key, {}))

    stored_dates = _sort_dates(progress.get("checkin_dates", []))
    legacy_dates = _legacy_checkin_dates(dorm_key)
    checkin_dates = _sort_dates([*stored_dates, *legacy_dates])
    date_set = set(checkin_dates)

    previous_total = int(progress.get("total_xp", 0) or 0)
    total_xp = max(previous_total, len(checkin_dates) * CHECKIN_XP)
    streak_days = _calc_streak(date_set, date_key)
    weekly_checkins = _weekly_dates(date_set, date_key)

    progress.update(
        {
            "dorm_id": dorm_key,
            "checkin_dates": checkin_dates,
            "today_checked_in": date_key in date_set,
            "streak_days": streak_days,
            "weekly_checkins": weekly_checkins,
            "earned_xp": int(progress.get("earned_xp", 0) or 0),
            "total_xp": total_xp,
            "unlocked_badges": _unlocked_badges(streak_days, total_xp, len(checkin_dates)),
            "last_checkin_date": progress.get("last_checkin_date", ""),
        }
    )
    by_dorm[dorm_key] = progress
    _mirror_current_progress(progress)
    return progress


def record_checkin(dorm_id: str, date_str: str | None = None, xp: int = CHECKIN_XP) -> dict:
    init_user_progress_state()
    dorm_key = str(dorm_id or "—")
    date_key = str(date_str or current_demo_date_str())[:10]
    progress = get_user_progress(dorm_key, date_key)

    checkin_dates = set(progress.get("checkin_dates", []))
    already_done = date_key in checkin_dates
    if not already_done:
        checkin_dates.add(date_key)
        progress["checkin_dates"] = _sort_dates(checkin_dates)
        progress["earned_xp"] = int(xp)
        progress["total_xp"] = int(progress.get("total_xp", 0) or 0) + int(xp)
        progress["last_checkin_date"] = date_key

        logs = st.session_state.setdefault("tasks_checkin_log", [])
        exists = any(
            isinstance(r, dict)
            and str(r.get("dorm_id", "")) == dorm_key
            and str(r.get("date", "")) == date_key
            for r in logs
        )
        if not exists:
            logs.append(
                {
                    "date": date_key,
                    "dorm_id": dorm_key,
                    "xp": int(xp),
                    "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
    else:
        progress["earned_xp"] = 0

    st.session_state["user_progress_by_dorm"][dorm_key] = progress
    return get_user_progress(dorm_key, date_key)


def get_badges(progress: dict) -> list[dict]:
    unlocked = set(progress.get("unlocked_badges", []))
    return [
        {
            **badge,
            "unlocked": badge["id"] in unlocked,
        }
        for badge in BADGE_DEFS
    ]
