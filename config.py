#配置中心
import os
from pathlib import Path

# ===================== 基础路径 =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
STATE_DIR = Path(BASE_DIR) / "state"

SQLITE_DB_PATH = os.path.join(DATA_DIR, "app.db")

# ===================== dorm 默认画像 =====================
# 说明：
# 这里是当前用于批量评分/演示的 dorm 默认 CS 画像值。
# 后续如果你改成真实问卷或数据库读取，这里可以被替换。
DORM_CS_PROFILE = {
    "TJ_SIPING_B3_0612": 0.72,
    "TJ_SIPING_B3_0613": 0.48,
    "TJ_SIPING_B3_0614": 0.31,
}

# ===================== Ridge baseline 配置 =====================
FEATURE_COLS = ["hour", "dow", "is_weekend"]

# ===================== 问卷评分映射 =====================
# q10: 自评行动倾向
SELF_MAP = {
    "很强，并总能付诸行动": 4,
    "较强，但有时会向舒适度妥协": 3,
    "一般，方便省事更重要": 2,
    "较弱，很少关注": 1,
}

# q8: 节能知识/做法中的“正确项”
Q8_GOOD = {
    "设定26℃及以上（夏）/20℃及以下（冬）",
    "无人时或离开前必定关闭空调",
    "使用时会紧闭门窗",
    "会定期清洁空调滤网",
}

# q12: 判断题计分
Q12_SCORE = {
    "夏季空调温度设定越低越舒适": -1,
    "长时间开空调时应关闭门窗以提高效率": 1,
    "空调运行时打开电扇辅助，能更快降温且更节能": 1,
    "定期清洁滤网对节能基本没影响": -1,
}

# q11: 主要影响因素映射到 R 分量
Q11_R = {
    "个人体感舒适度": 0.00,
    "宿舍电费分摊方式": 0.67,
    "室友的使用习惯和意见": 0.33,
    "个人的节能环保理念": 1.00,
}

# ===================== A-E分群中心 =====================
# 说明：
# A-E 五类群体在 (CS, BS) 平面上的中心点
CENTERS = {
    "A": (0.83, 0.3129),
    "B": (0.97, 0.5705),
    "C": (0.56, 0.3745),
    "D": (0.71, 0.6501),
    "E": (0.96, 0.8129),
}



# ===================== 默认干预候选策略 =====================
# 说明：
# 不同 cluster 默认允许的 template_id 候选集。
# Admin 页若启用自定义配置，可覆盖本默认值。
#
# 当前阶段：
# - Messages 页先承接 P（信息提示）/ R（定期提醒）
# - F（数据反馈）后续由 Home 页波动图承接
# - S（社会比较）后续由 Home 页排行榜承接
#
# 设计思路：
# - 保留论文中各群体的干预方向
# - 但先将其翻译为更适合 Message 页展示的 P/R 模板

DEFAULT_CLUSTER_ARMS = {
    # A 类：高意识-低行为
    # 特征：启动成本高、拖延明显
    # 当前更适合：给出清晰、低门槛、可立即执行的提示与提醒
    "A": [
        "P_tip_leave_01",       # 离室/短时外出小贴士
        "P_tip_temp_01",        # 合理温度设定建议
        "P_tip_window_01",      # 开空调前检查门窗
        "R_push_leave_01",      # 离开前确认关闭
        "R_push_setpoint_01",   # 设定温度提醒
    ],

    # B 类：中等意识-低行为
    # 特征：认知不足、习惯弱
    # 当前更适合：解释型提示 + 习惯型提醒
    "B": [
        "P_tip_temp_01",        # 温度设置知识
        "P_tip_fan_01",         # 配合风扇降低负担
        "P_tip_window_01",      # 场景性操作提示
        "R_push_habit_01",      # 连续形成节能习惯
        "R_push_setpoint_01",   # 固定温度习惯提醒
    ],

    # C 类：高意识-中等行为
    # 特征：执行不稳定
    # 当前更适合：少量说明 + 更稳定的时机提醒
    "C": [
        "P_tip_temp_01",        # 轻量知识补充
        "P_tip_filter_01",      # 使用效率相关提示
        "R_push_night_01",      # 夜间/睡前提醒
        "R_push_setpoint_01",   # 稳定温度设定
        "R_push_window_01",     # 关键场景触发提醒
    ],

    # D 类：中等意识-中等行为
    # 特征：参与动机偏低，容易受便利性影响
    # 当前更适合：低成本、轻负担、直接行动型消息
    "D": [
        "P_tip_leave_01",       # 简短易懂的行为提示
        "P_tip_window_01",      # 低成本操作建议
        "R_push_leave_01",      # 行动提醒
        "R_push_habit_01",      # 保持参与感
        "R_push_window_01",     # 场景触发
    ],

    # E 类：意识-行为一致
    # 特征：基础较好，但需避免疲劳和过度打扰
    # 当前更适合：轻量、温和、频率低的提示与提醒
    "E": [
        "P_tip_fan_01",         # 优化型小贴士
        "P_tip_temp_01",        # 精细化设置建议
        "R_push_night_01",      # 温和提醒
        "R_push_setpoint_01",   # 轻度维持
    ],
}

# ===================== template_id -> 干预模块映射 =====================
TEMPLATE_TO_MODULE = {
    # P：信息提示 / 小贴士
    "P_tip_temp_01": "P",
    "P_tip_leave_01": "P",
    "P_tip_window_01": "P",
    "P_tip_fan_01": "P",
    "P_tip_filter_01": "P",

    # R：定期提醒 / 推送消息
    "R_push_leave_01": "R",
    "R_push_night_01": "R",
    "R_push_window_01": "R",
    "R_push_setpoint_01": "R",
    "R_push_habit_01": "R",

    # F：数据反馈（后续接 Home）
    "F_feedback_weekly_01": "F",
    "F_feedback_trend_01": "F",

    # S：社会比较（后续接 Home）
    "S_social_rank_01": "S",
    "S_social_peer_01": "S",
}

# ===================== 本地状态文件 =====================
# 说明：
# 这些文件用于保存/恢复 session_state 的关键内容
STATE_FILES = {
    "linucb_params": STATE_DIR / "linucb_params.json",
    "cluster_arms_config": STATE_DIR / "cluster_arms_config.json",
    "intervention_logs": STATE_DIR / "intervention_logs.json",
    "interaction_logs": STATE_DIR / "interaction_logs.json",
    "scores": STATE_DIR / "scores.json",
    "messages": STATE_DIR / "messages.json",
    "app_meta": STATE_DIR / "app_meta.json",
}