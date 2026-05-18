import json
from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st

from config import STATE_DIR, STATE_FILES
from models.bandit import linucb_init


# 确保 state/ 目录存在
STATE_DIR.mkdir(exist_ok=True)


def _ndarray_to_list(x):
    """
    将 numpy.ndarray 转为可 JSON 序列化的 list。
    """
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def _list_to_ndarray(x):
    """
    将 JSON 里的 list 恢复为 numpy.ndarray。
    """
    return np.array(x, dtype=float)


def serialize_linucb_params(params: dict) -> dict:
    """
    将 LinUCB 参数转换为可写入 JSON 的结构。

    当前兼容的字段包括：
    - d
    - version
    - select_count
    - update_count
    - last_update_ts
    - A_by_arm
    - b_by_arm
    JSON 不能直接存 numpy.ndarray
    所以 A_by_arm、b_by_arm 需要 list 化
    恢复时再转回 numpy.ndarray
    还要补齐旧状态文件可能缺失的键
    """
    out = {
        "d": params.get("d", 4),
        "version": params.get("version", ""),
        "select_count": params.get("select_count", {}),
        "update_count": params.get("update_count", {}),
        "last_update_ts": params.get("last_update_ts", {}),
        "A_by_arm": {},
        "b_by_arm": {},
    }

    for arm, A in params.get("A_by_arm", {}).items():
        out["A_by_arm"][arm] = _ndarray_to_list(A)

    for arm, b in params.get("b_by_arm", {}).items():
        out["b_by_arm"][arm] = _ndarray_to_list(b)

    return out


def deserialize_linucb_params(obj: dict) -> dict:
    """
    将 JSON 结构恢复为 LinUCB 参数字典。

    恢复后会补齐缺失字段，避免旧状态文件缺键时报错。
    """
    params = linucb_init(d=int(obj.get("d", 4)))
    params["version"] = obj.get("version", params.get("version"))
    params["select_count"] = obj.get("select_count", {})
    params["update_count"] = obj.get("update_count", {})
    params["last_update_ts"] = obj.get("last_update_ts", {})

    for arm, A_list in obj.get("A_by_arm", {}).items():
        params["A_by_arm"][arm] = _list_to_ndarray(A_list)

    for arm, b_list in obj.get("b_by_arm", {}).items():
        params["b_by_arm"][arm] = _list_to_ndarray(b_list)

    for arm in list(params["A_by_arm"].keys()):
        if arm not in params["b_by_arm"]:
            params["b_by_arm"][arm] = np.zeros(params["d"])
        if arm not in params["select_count"]:
            params["select_count"][arm] = 0
        if arm not in params["update_count"]:
            params["update_count"][arm] = 0
        if arm not in params["last_update_ts"]:
            params["last_update_ts"][arm] = None

    return params


def save_json(path: Path, data):
    """
    以 UTF-8 编码保存 JSON 文件。
    """
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def load_json(path: Path):
    """
    读取 JSON 文件并返回 Python 对象。
    """
    return json.loads(path.read_text(encoding="utf-8"))


def save_app_state():
    """
    将当前 session_state 中的关键运行状态保存到 state/ 目录。

    当前保存内容：
    - linucb_params
    - cluster_arms_config
    - intervention_logs
    - interaction_logs
    - scores
    - messages
    - app_meta（saved_at, demo_time_mode）
    
    保存的是“运行状态快照”
    不是数据库持久化
    app_meta 里为什么要保存 saved_at 和 demo_time_mode
    """
    save_json(
        STATE_FILES["linucb_params"],
        serialize_linucb_params(st.session_state["linucb_params"])
    )

    save_json(
        STATE_FILES["cluster_arms_config"],
        st.session_state.get("cluster_arms_config")
    )
    save_json(
        STATE_FILES["intervention_logs"],
        st.session_state.get("intervention_logs", [])
    )
    save_json(
        STATE_FILES["interaction_logs"],
        st.session_state.get("interaction_logs", [])
    )
    save_json(
        STATE_FILES["scores"],
        st.session_state.get("scores", [])
    )
    save_json(
        STATE_FILES["messages"],
        st.session_state.get("messages", [])
    )

    meta = {
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "demo_time_mode": bool(st.session_state.get("demo_time_mode", True)),
    }
    save_json(STATE_FILES["app_meta"], meta)


def load_app_state():
    """
    从 state/ 目录加载上次保存的运行状态。

    返回
    ----
    bool
        True 表示成功加载；
        False 表示未找到必要状态文件。
    为什么把 linucb_params 作为“必要状态文件”

    为什么其他文件存在时才恢复

    返回布尔值是为了让页面按钮给出“成功/未找到”的反馈
    """
    if not STATE_FILES["linucb_params"].exists():
        return False

    st.session_state["linucb_params"] = deserialize_linucb_params(
        load_json(STATE_FILES["linucb_params"])
    )

    if STATE_FILES["cluster_arms_config"].exists():
        st.session_state["cluster_arms_config"] = load_json(
            STATE_FILES["cluster_arms_config"]
        )

    if STATE_FILES["intervention_logs"].exists():
        st.session_state["intervention_logs"] = load_json(
            STATE_FILES["intervention_logs"]
        )

    if STATE_FILES["interaction_logs"].exists():
        st.session_state["interaction_logs"] = load_json(
            STATE_FILES["interaction_logs"]
        )

    if STATE_FILES["scores"].exists():
        st.session_state["scores"] = load_json(
            STATE_FILES["scores"]
        )

    if STATE_FILES["messages"].exists():
        st.session_state["messages"] = load_json(
            STATE_FILES["messages"]
        )

    if STATE_FILES["app_meta"].exists():
        meta = load_json(STATE_FILES["app_meta"])
        st.session_state["demo_time_mode"] = bool(meta.get("demo_time_mode", True))

    return True


def clear_saved_state_files():
    """
    清空 state/ 目录下当前配置的所有状态文件。
    它只清空本地 state/ 文件
    不影响 SQLite 事件表
    不影响当前内存里的 session_state
    """
    for p in STATE_FILES.values():
        try:
            if p.exists():
                p.unlink()
        except Exception:
            pass