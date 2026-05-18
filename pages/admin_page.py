import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from config import DORM_CS_PROFILE, DEFAULT_CLUSTER_ARMS, DATA_DIR
from models.bandit import build_rl_monitor_df,parse_state_vector, linucb_update
from services.replay_service import (
    _latest_scores_map,
    clear_scores_for_all_dorms,
    batch_generate_scores_for_all_dorms,
    batch_generate_interventions_replay,
    reward_sum_for_intervention,
    get_auto_replay_range,
    load_user_pool_from_excel,
    batch_generate_scores_from_user_pool,
    batch_generate_scores_for_real_dorms_from_user_pool,
    generate_simulated_user_pool_from_base_samples,
    batch_generate_scores_from_sim_user_pool,
)

try:
    # 这些函数在部分拆分版本里可能已经存在。
    from models.bandit import linucb_update, linucb_init, parse_state_vector
except Exception:
    linucb_update = None
    linucb_init = None
    parse_state_vector = None

from services.sqlite_service import load_sqlite_intervention_events
try:
    from services.sqlite_service import (
        update_sqlite_reward,
        delete_sqlite_test_events,
        save_sqlite_intervention_event,
    )
except Exception:
    update_sqlite_reward = None
    delete_sqlite_test_events = None
    save_sqlite_intervention_event = None

from services.replay_service import (
    _latest_scores_map,
    clear_scores_for_all_dorms,
    batch_generate_scores_for_all_dorms,
    batch_generate_interventions_replay,
    reward_sum_for_intervention,
    get_auto_replay_range,
    load_user_pool_from_excel,
    batch_generate_scores_for_real_dorms_from_user_pool,
)

from utils.state_utils import save_app_state, load_app_state, clear_saved_state_files


# =========================================================
# 兼容辅助函数
# =========================================================
def _pick_ts_col(df: pd.DataFrame):
    """兼容 timestamp / timestamp_hour 两套时间列名。"""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    for c in ["timestamp", "timestamp_hour", "ts"]:
        if c in df.columns:
            return c
    return None


def _is_missing_value(v) -> bool:
    """兼容旧主程序里对空 reward 的判断。"""
    if v is None:
        return True
    try:
        if pd.isna(v):
            return True
    except Exception:
        pass
    s = str(v).strip().lower()
    return s in {"", "none", "nan"}


def _fallback_parse_state_vector(state_json_str: str):
    """当 models.bandit 未导出 parse_state_vector 时使用本地兜底。"""
    try:
        obj = json.loads(state_json_str) if state_json_str else {}
    except Exception:
        obj = {}
    cs = float(obj.get("CS", 0.0))
    bs = float(obj.get("BS_energy_week", obj.get("BS", 0.0)))
    recent = float(obj.get("recent_diff_3h", 0.0))
    is_weekend = float(obj.get("is_weekend", 0))
    return np.array([cs, bs, recent, is_weekend], dtype=float)


def _get_state_vector(log: dict):
    """兼容 state_json / state_vector_json 两种旧字段。"""
    raw = log.get("state_vector_json") or log.get("state_json") or ""
    if parse_state_vector is not None:
        try:
            return parse_state_vector(raw)
        except Exception:
            pass
    return _fallback_parse_state_vector(raw)


def _ensure_default_session_state():
    """页面运行前，补齐几个常用 session_state，减少拆分后的空值 bug。"""
    st.session_state.setdefault("scores", [])
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("interaction_logs", [])
    st.session_state.setdefault("intervention_logs", [])
    st.session_state.setdefault("cluster_arms_config", None)
    st.session_state.setdefault("demo_time_mode", True)

def _ensure_bandit_arms(params, cluster_arms_config, d=4):
    """
    若当前 linucb_params 中没有任何 arm，则按当前 cluster 配置/默认配置补齐。
    这样 Update Center 才能识别 replay 生成出来的 arm_id。
    """
    import numpy as np

    if params is None:
        params = {
            "d": int(d),
            "alpha": 1.0,
            "A_by_arm": {},
            "b_by_arm": {},
            "select_count": {},
            "update_count": {},
            "last_update_ts": {},
        }

    params.setdefault("d", int(d))
    params.setdefault("alpha", 1.0)
    params.setdefault("A_by_arm", {})
    params.setdefault("b_by_arm", {})
    params.setdefault("select_count", {})
    params.setdefault("update_count", {})
    params.setdefault("last_update_ts", {})

    cfg = cluster_arms_config or DEFAULT_CLUSTER_ARMS
    all_arms = sorted({arm for arms in cfg.values() for arm in arms})

    for arm in all_arms:
        if arm not in params["A_by_arm"]:
            params["A_by_arm"][arm] = np.eye(int(params["d"]), dtype=float)
        if arm not in params["b_by_arm"]:
            params["b_by_arm"][arm] = np.zeros(int(params["d"]), dtype=float)
        if arm not in params["select_count"]:
            params["select_count"][arm] = 0
        if arm not in params["update_count"]:
            params["update_count"][arm] = 0
        if arm not in params["last_update_ts"]:
            params["last_update_ts"][arm] = None

    return params


# =========================================================
# 页面主函数
# =========================================================
def render_admin_page(hourly_df, outcome_logs_df, baseline_model):
    """
    恢复版 Admin 页面。

    目标：
    1. 尽量保留你旧主程序里已经能跑通的 Admin 板块；
    2. 对拆分后的接口差异做兼容；
    3. 注释写清楚，方便你后续继续拆模块。
    """
    _ensure_default_session_state()

    st.title("Admin｜实验管理与结果查看")
    st.caption("本页面用于数据查看、实验控制、reward 回填、效果查看与导出。")

    st.markdown(
        """
        **推荐使用顺序：**
        1. 先看 **A. 数据查看区**，确认数据、日志与评分是否正常。
        2. 再到 **B. 实验操作区**，批量生成评分与回放事件。
        3. 然后在 **8) 更新中心** 批量回填 reward 并更新 LinUCB。
        4. 最后到 **C / D 区域** 查看效果、参数变化与导出数据。
        """
    )

    # 统一读取当前状态，避免同一页面多次写长表达式。
    scores = st.session_state.get("scores", [])
    messages = st.session_state.get("messages", [])
    interaction_logs = st.session_state.get("interaction_logs", [])
    intervention_logs = st.session_state.get("intervention_logs", [])
    linucb_params = st.session_state.get("linucb_params")

    # =========================================================
    # A. 数据查看区
    # =========================================================
    st.divider()
    st.header("A. 数据查看区")
    st.caption("用于查看基础数据、日志、评分结果以及当前状态。")

    # ---------- 0) 模型注册表 ----------
    st.subheader("0) 模型注册表")
    st.caption("查看当前 baseline / replay / RL 相关对象是否已挂载。")

    model_registry_rows = [
        {
            "module": "baseline_model",
            "status": "已加载" if baseline_model is not None else "未加载",
            "type": type(baseline_model).__name__ if baseline_model is not None else "None",
            "notes": "用于基线预测板块展示",
        },
        {
            "module": "linucb_params",
            "status": "已初始化" if linucb_params else "未初始化",
            "type": type(linucb_params).__name__ if linucb_params is not None else "None",
            "notes": "LinUCB 共享参数容器",
        },
    ]
    model_registry_df = pd.DataFrame(model_registry_rows)
    st.dataframe(model_registry_df, use_container_width=True)

    # ---------- 1) 周指标（轻量恢复版） ----------
    st.subheader("1) 周指标（按 dorm）")
    st.caption("旧主程序里有独立 weekly_metrics_by_dorm；拆分版先用当前数据动态计算一个轻量版本。")

    weekly_metrics_df = pd.DataFrame()
    try:
        base_df = outcome_logs_df if outcome_logs_df is not None and not outcome_logs_df.empty else hourly_df
        if base_df is not None and not base_df.empty and "dorm_id" in base_df.columns:
            tmp = base_df.copy()
            ts_col = _pick_ts_col(tmp)
            if ts_col is not None:
                tmp[ts_col] = pd.to_datetime(tmp[ts_col], errors="coerce")
                tmp = tmp.dropna(subset=[ts_col])
                # 只看最近 7*24 条左右，得到一个“最近一周”的轻量概览。
                rows = []
                for dorm_id, sub in tmp.groupby("dorm_id"):
                    sub = sub.sort_values(ts_col).tail(24 * 7)
                    row = {"dorm_id": dorm_id, "n_rows_recent": len(sub)}
                    for c in ["energy_kwh", "kwh", "actual_kwh", "baseline_pred", "baseline_kwh"]:
                        if c in sub.columns:
                            row[f"avg_{c}"] = round(float(pd.to_numeric(sub[c], errors="coerce").dropna().mean()), 4) if len(sub) > 0 else np.nan
                    rows.append(row)
                weekly_metrics_df = pd.DataFrame(rows)
    except Exception as e:
        st.warning(f"周指标计算失败，已跳过：{e}")

    if weekly_metrics_df.empty:
        st.info("当前无法构造周指标表，但这不影响后续 replay / 更新流程。")
    else:
        st.dataframe(weekly_metrics_df, use_container_width=True)
        st.download_button(
            "下载 weekly_metrics_by_dorm.csv",
            data=weekly_metrics_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="weekly_metrics_by_dorm.csv",
            mime="text/csv",
        )

    # ---------- 2) 结果日志 ----------
    st.subheader("2) 结果日志")
    st.caption("查看小时级 outcome 数据，是 reward 归因的重要基础。")
    if outcome_logs_df is None or outcome_logs_df.empty:
        st.info("当前 outcome_logs_df 为空。")
    else:
        st.dataframe(outcome_logs_df.tail(200), use_container_width=True)
        st.download_button(
            "下载 outcome_logs.csv",
            data=outcome_logs_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="outcome_logs.csv",
            mime="text/csv",
        )

    # ---------- 3) 干预日志 ----------
    st.subheader("3) 干预日志")
    st.caption("查看每一次干预决策记录，包括模板、状态和算法字段。")
    inter_df = pd.DataFrame(intervention_logs)
    if inter_df.empty:
        st.info("暂无 intervention_logs。")
    else:
        st.dataframe(inter_df.tail(200), use_container_width=True)
        st.download_button(
            "下载 intervention_logs.csv",
            data=inter_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="intervention_logs.csv",
            mime="text/csv",
        )

    # ---------- 4) 交互日志 ----------
    st.subheader("4) 交互日志")
    st.caption("查看平台交互层面的日志记录。")
    log_df = pd.DataFrame(interaction_logs)
    if log_df.empty:
        st.info("暂无 interaction_logs。")
    else:
        st.dataframe(log_df.tail(200), use_container_width=True)
        st.download_button(
            "下载 interaction_logs.csv",
            data=log_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="interaction_logs.csv",
            mime="text/csv",
        )

    # ---------- 5) 状态评分 ----------
    
    # ---------------------------------------------------------
    # 批量评分工具
    # 说明：
    # 1. 保留旧版“按真实小时数据生成 scores”的入口，便于对照测试
    # 2. 新增“从用户池映射到真实 dorm”入口，让 replay 真正吃到用户池画像
    # ---------------------------------------------------------
    st.subheader("5) 模拟用户池评分")
    st.caption("仅保留模式 B：以原始 670 份样本为分布依据，生成更多模拟用户，并同步生成 scores。")

    user_pool_path = os.path.join(DATA_DIR, "user_pool.xlsx")

    tau_batch = st.slider(
        "批量评分用 BS 尺度参数 τ",
        min_value=0.10,
        max_value=0.60,
        value=0.30,
        step=0.05,
        key="tau_batch_scores",
    )

    col_sim_1, col_sim_2, col_sim_3 = st.columns([1, 1, 1])

    with col_sim_1:
        sim_target_n = st.number_input(
            "模拟用户数量",
            min_value=5,
            max_value=10000,
            value=20,
            step=5,
            key="sim_target_n",
        )

    with col_sim_2:
        sim_noise_scale = st.number_input(
            "扰动强度",
            min_value=0.00,
            max_value=0.20,
            value=0.03,
            step=0.01,
            key="sim_noise_scale",
        )

    with col_sim_3:
        sim_pool_seed = st.number_input(
            "模拟池随机种子",
            min_value=0,
            max_value=9999,
            value=42,
            step=1,
            key="sim_pool_seed",
        )

    if st.button("生成更大的模拟用户池", key="gen_large_sim_pool"):
        try:
            base_user_pool_df = load_user_pool_from_excel(
                excel_path=user_pool_path,
                sheet_name="Sheet2",
            )

            sim_user_pool_df = generate_simulated_user_pool_from_base_samples(
                base_user_pool_df=base_user_pool_df,
                target_n=int(sim_target_n),
                noise_scale=float(sim_noise_scale),
                random_state=int(sim_pool_seed),
                id_prefix="SIM",
            )

            st.session_state["sim_user_pool_df"] = sim_user_pool_df

            scores_from_sim_pool, msg = batch_generate_scores_from_sim_user_pool(
                sim_user_pool_df=sim_user_pool_df,
                tau=tau_batch,
            )
            st.session_state["scores"] = scores_from_sim_pool

            st.success(f"已生成 {len(sim_user_pool_df)} 条模拟用户池记录。{msg}")

            if not sim_user_pool_df.empty:
                st.markdown("**模拟用户池预览**")
                preview_cols = [
                    c for c in ["dorm_id", "AS", "BS", "CS", "BS_energy_week", "pool_source"]
                    if c in sim_user_pool_df.columns
                ]
                st.dataframe(sim_user_pool_df[preview_cols].head(10), use_container_width=True)

            score_df = pd.DataFrame(scores_from_sim_pool)
            if not score_df.empty and "cluster_type" in score_df.columns:
                st.markdown("**模拟用户池对应的群体分布**")
                cluster_dist = (
                    score_df["cluster_type"]
                    .value_counts(dropna=False)
                    .rename_axis("cluster_type")
                    .reset_index(name="count")
                )
                st.dataframe(cluster_dist, use_container_width=True)

        except Exception as e:
            st.error(f"生成更大的模拟用户池失败：{e}")
    
    # =========================================================
    # B. 实验操作区
    # =========================================================
    st.divider()
    st.header("B. 实验操作区")
    st.caption("实验运行的主要区域：先生成事件，再回填 reward 并更新参数。")

    # ---------- 6) 时间模式 ----------
    st.subheader("6) 时间模式")
    st.caption("设置 t0 采用演示模式或真实模式。")
    st.session_state["demo_time_mode"] = st.toggle(
        "演示模式：t0 绑定数据轴（保证 reward 可更新）",
        value=bool(st.session_state.get("demo_time_mode", True)),
        key="admin_demo_time_mode",
    )
    st.caption("演示模式用于快速调试；真实模式更接近部署时机。")

    # ---------- 7) 配置中心 ----------
    st.subheader("7) 配置中心")
    st.caption("配置 cluster 到模板候选集合（arms / template_id）的映射关系。")

    use_custom = st.toggle(
        "启用自定义 cluster_arms（保存在 session_state）",
        value=st.session_state.get("cluster_arms_config") is not None,
        key="use_custom_cluster_arms",
    )

    active_cfg = st.session_state.get("cluster_arms_config")
    if active_cfg is None:
        active_cfg = DEFAULT_CLUSTER_ARMS

    rows = []
    for c in ["A", "B", "C", "D", "E"]:
        rows.append({"cluster_type": c, "arms_csv": ",".join(active_cfg.get(c, []))})
    cfg_df = pd.DataFrame(rows)
    edited_cfg_df = st.data_editor(cfg_df, use_container_width=True, num_rows="fixed", key="cluster_cfg_editor")

    colx, coly = st.columns(2)
    with colx:
        if st.button("保存配置（应用到推荐）", key="save_cluster_config"):
            new_cfg = {}
            for _, r in edited_cfg_df.iterrows():
                c = r["cluster_type"]
                arms = [x.strip() for x in str(r["arms_csv"]).split(",") if x.strip()]
                new_cfg[c] = arms
            st.session_state["cluster_arms_config"] = new_cfg if use_custom else None
            st.success("已保存 cluster_arms 配置。回 Home / Replay 即刻生效。")

    with coly:
        if st.button("恢复默认配置", key="reset_cluster_config"):
            # 与主程序的约定保持一致：None 表示使用默认 DEFAULT_CLUSTER_ARMS。
            st.session_state["cluster_arms_config"] = None
            st.success("已恢复默认 DEFAULT_CLUSTER_ARMS。")
            st.rerun()

    # ---------- 7.5) 回放工具 ----------
    st.markdown(
        """
        ### 使用说明
        1. 先在 **7.5) 回放工具** 中按日期范围批量生成 LinUCB 干预事件。
        2. 再到 **8) 更新中心** 中批量回填 reward 并更新 LinUCB 参数。
        3. 最后在本区查看 SQLite 实验事件、轻量效果概览，并按需导出 CSV。
        """
    )
    st.subheader("7.5) 回放工具")
    st.caption("回放日期自动跟随当前数据时间范围；模拟时间长度也会自动对齐回放天数。")
    st.write("当前 intervention_logs 总条数：", len(st.session_state.get("intervention_logs", [])))
    # =========================================================
    # 自动推导回放时间范围
    # 优先使用 outcome_logs_df；若不可用，则退回 hourly_df
    # 注意：这里默认按 12h reward attribution window 预留安全结束时间
    # =========================================================
    replay_start, replay_end = None, None
    attrib_hours_for_replay = 12

    try:
        replay_start, replay_end = get_auto_replay_range(outcome_logs_df)
    except Exception:
        try:
            replay_start, replay_end = get_auto_replay_range(hourly_df)
        except Exception:
            replay_start, replay_end = None, None

    # 若自动推导失败，则退回今天
    if replay_start is None:
        replay_start = pd.Timestamp.now().date()
    if replay_end is None:
        replay_end = pd.Timestamp.now().date()

    # 保存自动日期到 session_state，供其他模块复用
    st.session_state["auto_replay_start"] = pd.to_datetime(replay_start).date()
    st.session_state["auto_replay_end"] = pd.to_datetime(replay_end).date()

    # 同时也同步到 replay_start / replay_end，保持旧代码兼容
    st.session_state["replay_start"] = st.session_state["auto_replay_start"]
    st.session_state["replay_end"] = st.session_state["auto_replay_end"]

    colG1, colG2, colG3, colG4 = st.columns(4)
    with colG1:
        st.text_input(
            "回放起始日期（自动）",
            value=str(st.session_state["auto_replay_start"]),
            key="replay_start_display",
            disabled=True,
        )
    with colG2:
        st.text_input(
            "回放结束日期（自动）",
            value=str(st.session_state["auto_replay_end"]),
            key="replay_end_display",
            disabled=True,
        )
    with colG3:
        st.number_input(
            "每天发送小时（0-23）",
            min_value=0,
            max_value=23,
            value=9,
            step=1,
            key="replay_hour",
        )
    with colG4:
        st.number_input(
            "LinUCB alpha",
            min_value=0.1,
            max_value=10.0,
            value=1.2,
            step=0.1,
            key="replay_alpha",
        )

    # =========================================================
    # 自动统计回放规模
    # replay_days: 回放天数
    # dorm_count: 宿舍数
    # estimated_n: 预计生成条数
    # 同时把 sim_days 自动对齐为 replay_days
    # =========================================================
    try:
        replay_days = (
            pd.to_datetime(st.session_state["auto_replay_end"]).date()
            - pd.to_datetime(st.session_state["auto_replay_start"]).date()
        ).days + 1

        dorm_count = (
            0
            if outcome_logs_df is None
            or outcome_logs_df.empty
            or "dorm_id" not in outcome_logs_df.columns
            else outcome_logs_df["dorm_id"].dropna().astype(str).nunique()
        )

        estimated_n = max(replay_days, 0) * dorm_count

        # 自动让模拟时间长度与回放天数对齐
        if replay_days > 0:
            st.session_state["sim_days"] = int(replay_days)

    except Exception:
        replay_days, dorm_count, estimated_n = 0, 0, 0

    cA, cB, cC = st.columns(3)
    cA.metric("回放天数", replay_days)
    cB.metric("dorm 数", dorm_count)
    cC.metric("预计生成条数", estimated_n)

    if st.button("批量生成（写入 intervention_logs + SQLite，reward 先为空）", key="run_replay_batch"):
        try:
            n, msg = batch_generate_interventions_replay(
                outcome_logs_df=outcome_logs_df,
                scores=st.session_state.get("scores", []),
                linucb_params=st.session_state.get("linucb_params"),
                start_date=str(st.session_state["auto_replay_start"]),
                end_date=str(st.session_state["auto_replay_end"]),
                send_hour=int(st.session_state["replay_hour"]),
                cluster_arms_config=st.session_state.get("cluster_arms_config"),
                alpha_ucb=float(st.session_state["replay_alpha"]),
                write_sqlite=True,
            )
            if n > 0:
                st.success(f"已生成 {n} 条干预事件。{msg}")
                st.info("下一步：到 8) 更新中心 点击批量更新按钮，统一回填 reward 并更新 LinUCB。")
            else:
                st.error(f"生成失败：{msg}")
        except Exception as e:
            st.error(f"Replay 生成失败：{e}")

    st.write("当前 intervention_logs 总条数：", len(st.session_state.get("intervention_logs", [])))
    replay_preview_df = pd.DataFrame(st.session_state.get("intervention_logs", []))
    if not replay_preview_df.empty:
        st.markdown("**最近 10 条干预事件预览**")
        st.dataframe(replay_preview_df.tail(10), use_container_width=True)

    score_df_check = pd.DataFrame(st.session_state.get("scores", []))
    if not score_df_check.empty and "cluster_type" in score_df_check.columns:
        st.markdown("**当前 scores 的 cluster 分布**")
        cluster_dist_check = (
            score_df_check["cluster_type"]
            .value_counts(dropna=False)
            .rename_axis("cluster_type")
            .reset_index(name="count")
        )
        st.dataframe(cluster_dist_check, use_container_width=True)
    # =========================================================
    # 8) 更新中心
    # 从旧主程序恢复的完整版本，并在原基础上补充轻量效果概览
    # 作用：
    # 1. 找出所有待更新的 LinUCB 干预事件
    # 2. 按 reward 窗口批量回填 reward_sum_12h
    # 3. 同步更新 LinUCB 参数
    # 4. 在本区直接显示轻量效果分析（不额外开大板块）
    # =========================================================
    st.subheader("8) 更新中心")
    st.caption("批量回填 reward，并同步更新 LinUCB 参数。")

    demo_update = st.toggle(
        "演示更新：忽略等待时间，允许立即更新",
        value=True,
        key="demo_update"
    )

    attrib_hours = st.selectbox(
        "reward 归因窗口（小时）",
        [6, 12, 24],
        index=1,
        key="attrib_hours"
    )

    inter_logs = st.session_state.get("intervention_logs", [])
    pending = []
    now_dt = pd.Timestamp.now()

    # ---------------------------------------------------------
    # 找出所有“待更新”的 LinUCB 条目
    # 条件：
    # - algo_type == LinUCB
    # - reward_sum_12h 仍为空
    # - decision_t0_hour 合法
    # - 已到可更新时间，或开启了 demo_update
    # ---------------------------------------------------------
    for idx, log in enumerate(inter_logs):
        if log.get("algo_type") != "LinUCB":
            continue

        # 仅更新“还没有 reward”的记录
        try:
            already_updated = not pd.isna(log.get("reward_sum_12h"))
        except Exception:
            already_updated = (log.get("reward_sum_12h") is not None)

        if already_updated:
            continue

        t0_str = log.get("decision_t0_hour")
        if not t0_str:
            continue

        t0 = pd.to_datetime(t0_str, errors="coerce")
        if pd.isna(t0):
            continue

        due = (now_dt >= (t0 + pd.Timedelta(hours=int(attrib_hours))))
        if demo_update or due:
            pending.append((idx, log, t0))

    st.write(f"待更新条目数：**{len(pending)}**")

    if len(pending) > 0:
        with st.expander("查看待更新条目（前 50 条）"):
            pending_preview = pd.DataFrame([x[1] for x in pending[:50]])
            st.dataframe(pending_preview, use_container_width=True)

    # ---------------------------------------------------------
    # 批量更新按钮
    # 功能：
    # - 计算 reward_sum_12h
    # - 回写 intervention_logs
    # - 更新 LinUCB 参数
    # ---------------------------------------------------------
    # 先确保 bandit arms 已初始化
    
    params = st.session_state.get("linucb_params")
    params = _ensure_bandit_arms(
        params=params,
        cluster_arms_config=st.session_state.get("cluster_arms_config"),
        d=4,
    )
    st.session_state["linucb_params"] = params
    
    if st.button("批量更新所有待更新条目", key="update_all_pending"):
        if len(pending) == 0:
            st.info("没有可更新的条目（要么未发送 LinUCB，要么已更新）。")
        else:
            if linucb_params is None:
                st.error("linucb_params 尚未初始化，无法更新。")
            else:
                params = st.session_state.get("linucb_params")
                updated_count = 0
                skipped_unknown_arm = 0

                for idx, log, t0 in pending:
                    dorm_id = log.get("dorm_id")
                    arm_id = log.get("arm_id") or log.get("template_id")

                    if not dorm_id or not arm_id:
                        continue

                    # 确保当前 arm 存在于 bandit 参数中
                    valid_arms = list(params.get("A_by_arm", {}).keys()) if isinstance(params, dict) else []
                    if arm_id not in valid_arms:
                        st.warning(f"跳过更新：未知 arm_id={arm_id}，当前 bandit arms={valid_arms}")
                        skipped_unknown_arm += 1
                        continue

                    # 计算 reward
                    r_sum = reward_sum_for_intervention(
                        outcome_logs_df,
                        dorm_id=dorm_id,
                        t0=t0,
                        hours=int(attrib_hours),
                    )

                    # 解析状态向量
                    x = None
                    try:
                        state_json = log.get("state_json") or log.get("state_vector_json", "")
                        if state_json:
                            import json
                            s = json.loads(state_json)
                            cs = float(s.get("CS", 0.5))
                            bs = float(s.get("BS_energy_week", 0.5))
                            recent_diff = float(s.get("recent_diff_3h", 0.0))
                            is_weekend = int(s.get("is_weekend", 0))
                            x = parse_state_vector(cs, bs, recent_diff, is_weekend)
                    except Exception:
                        x = None

                    if x is None:
                        continue

                    update_ts = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

                    # 更新 LinUCB 参数
                    try:
                        updated_params = linucb_update(
                            params,
                            arm_id,
                            x,
                            float(r_sum),
                            update_ts=update_ts
                        )
                    except TypeError:
                        # 兼容某些旧版本 linucb_update 不支持 update_ts
                        updated_params = linucb_update(
                            params,
                            arm_id,
                            x,
                            float(r_sum)
                        )

                    if updated_params is not None:
                        params = updated_params

                    # 回写 intervention_logs
                    inter_logs[idx]["reward_sum_12h"] = round(float(r_sum), 6)
                    inter_logs[idx]["update_timestamp"] = update_ts
                    inter_logs[idx]["attrib_hours"] = int(attrib_hours)

                    updated_count += 1

                st.session_state["linucb_params"] = params
                st.session_state["intervention_logs"] = inter_logs

                if updated_count > 0:
                    st.success(
                        f"已更新 {updated_count} 条记录。"
                        + (f" 跳过未知 arm {skipped_unknown_arm} 条。" if skipped_unknown_arm > 0 else "")
                    )
                else:
                    st.warning("没有成功更新任何条目。")
    st.write("当前 bandit arms：", list((st.session_state.get("linucb_params") or {}).get("A_by_arm", {}).keys()))
    # =========================================================
    # 嵌入式轻量效果概览
    # 不新增大板块，直接放在更新中心下面
    # =========================================================
    inter_df = pd.DataFrame(st.session_state.get("intervention_logs", []))

    if not inter_df.empty and "reward_sum_12h" in inter_df.columns:
        valid_df = inter_df[inter_df["reward_sum_12h"].notna()].copy()

        if not valid_df.empty:
            st.markdown("#### 更新后效果概览")

            m1, m2, m3 = st.columns(3)
            m1.metric("已更新条数", len(valid_df))
            m2.metric("平均 reward", round(pd.to_numeric(valid_df["reward_sum_12h"], errors="coerce").fillna(0).mean(), 4))
            m3.metric(
                "正 reward 占比",
                f"{(pd.to_numeric(valid_df['reward_sum_12h'], errors='coerce').fillna(0) > 0).mean() * 100:.1f}%"
            )

            # 按策略 arm 汇总
            if "arm_id" in valid_df.columns:
                arm_summary = (
                    valid_df.groupby("arm_id", dropna=False)["reward_sum_12h"]
                    .agg(["count", "mean"])
                    .reset_index()
                    .rename(columns={
                        "arm_id": "策略",
                        "count": "次数",
                        "mean": "平均reward",
                    })
                )
                st.markdown("**按策略统计**")
                st.dataframe(arm_summary, use_container_width=True)

            # 按 cluster 汇总
            if "cluster_type" in valid_df.columns:
                cluster_summary = (
                    valid_df.groupby("cluster_type", dropna=False)["reward_sum_12h"]
                    .agg(["count", "mean"])
                    .reset_index()
                    .rename(columns={
                        "cluster_type": "群体",
                        "count": "次数",
                        "mean": "平均reward",
                    })
                )
                st.markdown("**按群体统计**")
                st.dataframe(cluster_summary, use_container_width=True)

            # 负 reward 条目排查
            neg_df = valid_df[pd.to_numeric(valid_df["reward_sum_12h"], errors="coerce").fillna(0) < 0].copy()
            if not neg_df.empty:
                st.markdown("**负 reward 条目（前 20 条）**")
                show_cols = [c for c in ["dorm_id", "cluster_type", "arm_id", "decision_t0_hour", "reward_sum_12h"] if c in neg_df.columns]
                st.dataframe(neg_df[show_cols].head(20), use_container_width=True)

    # =========================================================
    # D. 算法解释区
    # =========================================================
    st.divider()
    st.header("D. 算法解释区")
    st.caption("用于理解 LinUCB 的选择依据与参数变化，不是实验主操作区。")

    # ---------- 9) RL 可解释面板 ----------
    st.subheader("9) RL 可解释面板")
    st.caption("查看 LinUCB 当前学习状态、选择次数、更新次数与近期决策。")

    params = st.session_state.get("linucb_params", {})
    inter_df = pd.DataFrame(st.session_state.get("intervention_logs", []))

    st.markdown("**RL 参数监控**")
    if not params:
        st.info("暂无 linucb_params。")
    else:
        try:
            rl_monitor_df = build_rl_monitor_df(params)
            st.dataframe(rl_monitor_df, use_container_width=True)
        except Exception as e:
            st.warning(f"生成 RL 监控表失败：{e}")

    st.markdown("**LinUCB 决策日志统计**")
    if inter_df.empty:
        st.info("暂无 intervention_logs。请先在 Home 或 Replay 生成 LinUCB 决策。")
    else:
        if "algo_type" in inter_df.columns:
            lin = inter_df[inter_df["algo_type"] == "LinUCB"].copy()
        else:
            lin = inter_df.iloc[0:0]

        st.write("LinUCB decision 总数：", len(lin))

        if not lin.empty:
            if "reward_sum_12h" in lin.columns:
                lin["reward_sum_12h_num"] = pd.to_numeric(lin["reward_sum_12h"], errors="coerce")
            else:
                lin["reward_sum_12h_num"] = np.nan

            if "ucb_score" in lin.columns:
                lin["ucb_num"] = pd.to_numeric(lin["ucb_score"], errors="coerce")
            else:
                lin["ucb_num"] = np.nan


            if "arm_id" in lin.columns:
                # 先把时间列安全转成 datetime，避免 object 类型在 groupby max 时报错
                if "update_timestamp" in lin.columns:
                    lin["update_timestamp_dt"] = pd.to_datetime(lin["update_timestamp"], errors="coerce")
                else:
                    lin["update_timestamp_dt"] = pd.NaT

                agg_map = {
                    "n_decisions": ("arm_id", "count"),
                    "avg_reward": ("reward_sum_12h_num", "mean"),
                    "avg_ucb": ("ucb_num", "mean"),
                    "n_updated": ("update_timestamp_dt", lambda s: int(pd.Series(s).notna().sum())),
                    "last_update": ("update_timestamp_dt", "max"),
                }

                g = lin.groupby("arm_id", dropna=False).agg(**agg_map).reset_index()

                # 展示前再转回字符串，界面更清楚
                if "last_update" in g.columns:
                    g["last_update"] = pd.to_datetime(g["last_update"], errors="coerce") \
                        .dt.strftime("%Y-%m-%d %H:%M:%S") \
                        .fillna("")

                st.markdown("**每个 arm 的日志统计**")
                st.dataframe(g.sort_values("n_decisions", ascending=False), use_container_width=True)


            if "cluster_type" in lin.columns and "arm_id" in lin.columns:
                pivot_df = (
                    lin.groupby(["cluster_type", "arm_id"])
                    .size()
                    .reset_index(name="n")
                    .sort_values(["cluster_type", "n"], ascending=[True, False])
                )
                st.markdown("**不同 cluster 选择了哪些 arm**")
                st.dataframe(pivot_df, use_container_width=True)

            st.markdown("**最近 20 次 LinUCB 决策**")
            show_cols = [
                c
                for c in [
                    "timestamp",
                    "dorm_id",
                    "cluster_type",
                    "arm_id",
                    "ucb_score",
                    "p_hat",
                    "bonus",
                    "reward_sum_12h",
                    "update_timestamp",
                    "decision_t0_hour",
                ]
                if c in lin.columns
            ]
            st.dataframe(lin[show_cols].tail(20), use_container_width=True)

    # =========================================================
    # C. 结果查看区
    # =========================================================
    st.divider()
    st.header("C. 结果查看区")
    st.caption("用于查看实验事件、回填进度、轻量效果概览与导出数据。")

    # ---------- 10) SQLite 实验事件表 ----------
    st.subheader("10) SQLite 实验事件表")
    st.caption("查看实验事件总表、平均 reward、按模板汇总，并支持 CSV 导出。")

    col_sqlite_1, col_sqlite_2, col_sqlite_3 = st.columns([1, 1, 1])
    with col_sqlite_1:
        if st.button("清空 SQLite 测试事件", key="delete_sqlite_test_events"):
            if delete_sqlite_test_events is None:
                st.warning("当前 sqlite_service 未导出 delete_sqlite_test_events。")
            else:
                try:
                    deleted_count = delete_sqlite_test_events()
                    st.success(f"已删除 {deleted_count} 条 SQLite 测试事件。")
                    st.rerun()
                except Exception as e:
                    st.error(f"删除测试事件失败：{e}")

    with col_sqlite_2:
        if st.button("刷新 SQLite 事件表", key="refresh_sqlite_events"):
            st.rerun()

    with col_sqlite_3:
        if st.button("写入一条 SQLite 测试事件", key="write_sqlite_test_event"):
            if save_sqlite_intervention_event is None:
                st.warning("当前 sqlite_service 未导出 save_sqlite_intervention_event。")
            else:
                try:
                    save_sqlite_intervention_event(
                        plug_id="plug_01",
                        decision_time="2026-03-03 11:00:00",
                        current_consumption=58.0,
                        baseline_consumption=46.6667,
                        recent_diff_3h=3.0,
                        is_weekend=0,
                        time_block="morning",
                        eligible_templates=["T5", "T1", "T2", "T4"],
                        assigned_template="T1",
                        assignment_mode="rule",
                        window_hours=12,
                        reward_value=None,
                        reward_ready_flag=0,
                    )
                    st.success("已写入一条 SQLite 测试事件。")
                    st.rerun()
                except Exception as e:
                    st.error(f"写入测试事件失败：{e}")

    sqlite_events_df = load_sqlite_intervention_events()
    if isinstance(sqlite_events_df, pd.DataFrame) and "error" in sqlite_events_df.columns:
        st.error("读取 SQLite intervention_events 失败：")
        st.write(sqlite_events_df.iloc[0]["error"])
    elif sqlite_events_df is None or sqlite_events_df.empty:
        st.info("SQLite 中暂无 intervention_events 记录。")
    else:
        st.markdown("### 实验效果概览")

        total_events = len(sqlite_events_df)
        ready_df = sqlite_events_df.copy()
        if "reward_ready_flag" in ready_df.columns:
            ready_df = ready_df[pd.to_numeric(ready_df["reward_ready_flag"], errors="coerce").fillna(0) == 1].copy()
        else:
            ready_df = ready_df.iloc[0:0]
        ready_count = len(ready_df)

        avg_reward = None
        if ready_count > 0 and "reward_value" in ready_df.columns:
            reward_series = pd.to_numeric(ready_df["reward_value"], errors="coerce").dropna()
            if len(reward_series) > 0:
                avg_reward = reward_series.mean()

        c1, c2, c3 = st.columns(3)
        c1.metric("总事件数", total_events)
        c2.metric("已回填 reward 数", ready_count)
        c3.metric("平均 reward", "-" if avg_reward is None else round(float(avg_reward), 4))

        if ready_count > 0:
            st.markdown("### 按模板查看平均 reward")
            tmp = ready_df.copy()
            if "reward_value" in tmp.columns:
                tmp["reward_value"] = pd.to_numeric(tmp["reward_value"], errors="coerce")
            tmp = tmp.dropna(subset=["reward_value"]) if "reward_value" in tmp.columns else tmp.iloc[0:0]
            if not tmp.empty and "assigned_template" in tmp.columns:
                summary_df = (
                    tmp.groupby("assigned_template", dropna=False)["reward_value"]
                    .agg(["count", "mean"])
                    .reset_index()
                    .rename(columns={"assigned_template": "模板", "count": "样本数", "mean": "平均reward"})
                )
                st.dataframe(summary_df, use_container_width=True)
            else:
                st.info("当前暂无可用于模板汇总的 reward 数据。")

        st.dataframe(sqlite_events_df.head(200), use_container_width=True)
        st.download_button(
            "下载 sqlite_intervention_events.csv",
            data=sqlite_events_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="sqlite_intervention_events.csv",
            mime="text/csv",
        )

        st.markdown("**最近一次 SQLite 实验事件**")
        latest_sqlite = sqlite_events_df.iloc[0]
        for k in [
            "plug_id",
            "decision_time",
            "current_consumption",
            "baseline_consumption",
            "recent_diff_3h",
            "is_weekend",
            "time_block",
            "eligible_templates",
            "assigned_template",
            "assignment_mode",
            "window_hours",
            "reward_value",
            "reward_ready_flag",
        ]:
            if k in latest_sqlite.index:
                st.write(f"{k}：{latest_sqlite.get(k, '')}")

    # ---------- 11) 最近一次候选评分 ----------
    st.subheader("11) 最近一次候选评分")
    st.caption("查看最近一次候选 arms 的评分结果，帮助理解模板选择依据。")
    last_tbl = st.session_state.get("rl_arm_last_metrics", [])
    if not last_tbl:
        st.info("暂无评分表。请先在 Home 生成一次 LinUCB 推荐。")
    else:
        st.dataframe(pd.DataFrame(last_tbl), use_container_width=True)

    # ---------- 12) LinUCB 参数面板 ----------
    st.subheader("12) LinUCB 参数面板")
    st.caption("查看 LinUCB 共享参数状态。")
    params = st.session_state.get("linucb_params", {})
    arms = sorted(list((params.get("A_by_arm") or {}).keys())) if isinstance(params, dict) else []
    if len(arms) == 0:
        st.info("暂无 arm 参数。先在 Home 发送一次 LinUCB。")
    else:
        rows = []
        for arm in arms:
            rows.append(
                {
                    "arm_id": arm,
                    "select_count": (params.get("select_count") or {}).get(arm, 0),
                    "update_count": (params.get("update_count") or {}).get(arm, 0),
                    "last_update_ts": (params.get("last_update_ts") or {}).get(arm),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # ---------- Maintenance buttons ----------
    st.markdown("### Maintenance")
    st.caption("这些按钮主要用于测试阶段快速清理状态。")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("清空 intervention_logs（测试用）"):
            st.session_state["intervention_logs"] = []
            st.success("已清空 intervention_logs。")

    with col2:
        if st.button("清空 messages（测试用）"):
            st.session_state["messages"] = []
            st.success("已清空 messages。")

    with col3:
        if st.button("重置 LinUCB 参数（测试用）"):
            if linucb_init is None:
                st.warning("当前 models.bandit 未导出 linucb_init。")
            else:
                try:
                    # 兼容旧版本：linucb_init(d=4)
                    st.session_state["linucb_params"] = linucb_init(d=4)
                    st.success("已按旧接口重置 LinUCB 参数。")
                except TypeError:
                    try:
                        # 兼容新版本：linucb_init()
                        st.session_state["linucb_params"] = linucb_init()
                        st.success("已按当前接口重置 LinUCB 参数。")
                    except Exception as e:
                        st.error(f"重置 LinUCB 参数失败：{e}")
                except Exception as e:
                    st.error(f"重置 LinUCB 参数失败：{e}")
                    
    # ---------- 状态持久化 ----------
    st.subheader("状态持久化｜保存 / 加载（state/）")
    colA, colB, colC = st.columns(3)
    with colA:
        if st.button("保存当前状态到本地"):
            try:
                save_app_state()
                st.success("已保存到 ./state/ 目录。")
            except Exception as e:
                st.error(f"保存失败：{e}")

    with colB:
        if st.button("从本地加载状态"):
            try:
                ok = load_app_state()
                if ok:
                    st.success("已从 ./state/ 加载状态。")
                    st.rerun()
                else:
                    st.warning("未找到可加载的状态文件（state/ 为空）。")
            except Exception as e:
                st.error(f"加载失败：{e}")

    with colC:
        if st.button("清空本地状态文件"):
            try:
                clear_saved_state_files()
                st.success("已清空 ./state/ 下的状态文件。")
            except Exception as e:
                st.error(f"清空失败：{e}")
