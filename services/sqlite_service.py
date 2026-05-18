# 往 SQLite 的 intervention_events 表里读写
import sqlite3
import pandas as pd

from config import SQLITE_DB_PATH


def ensure_sqlite_intervention_events_table(db_path=SQLITE_DB_PATH):
    """
    确保 SQLite 中的 intervention_events 表存在。

    该表用于保存每一次干预决策事件，包括：
    - 决策时刻
    - 当前状态特征
    - 候选模板/最终模板
    - 分配方式
    - 观察窗口
    - reward 及其是否已回填
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS intervention_events (
        event_id INTEGER PRIMARY KEY AUTOINCREMENT,
        plug_id TEXT NOT NULL,
        decision_time TEXT NOT NULL,
        current_consumption REAL,
        baseline_consumption REAL,
        recent_diff_3h REAL,
        is_weekend INTEGER,
        time_block TEXT,
        eligible_templates TEXT,
        assigned_template TEXT,
        assignment_mode TEXT,
        window_hours INTEGER,
        reward_value REAL,
        reward_ready_flag INTEGER DEFAULT 0
    )
    """)

    conn.commit()
    conn.close()


def save_sqlite_intervention_event(
    plug_id,
    decision_time,
    current_consumption,
    baseline_consumption,
    recent_diff_3h,
    is_weekend,
    time_block,
    eligible_templates,
    assigned_template,
    assignment_mode="rule",
    window_hours=12,
    reward_value=None,
    reward_ready_flag=0,
    db_path=SQLITE_DB_PATH
):
    """
    向 intervention_events 表写入一条干预事件。

    说明：
    - eligible_templates 若为 list，会转成逗号拼接字符串存储
    - reward 初始可为空，后续由 update_sqlite_reward 回填
    - assignment_mode 可用于区分 rule / LinUCB 等来源
    
    它会先确保表存在再读取
    默认按 event_id DESC 返回，方便 Admin 看最新事件
    若读取失败，返回带 error 列的 DataFrame，方便页面直接判断
    """
    ensure_sqlite_intervention_events_table(db_path)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO intervention_events (
        plug_id,
        decision_time,
        current_consumption,
        baseline_consumption,
        recent_diff_3h,
        is_weekend,
        time_block,
        eligible_templates,
        assigned_template,
        assignment_mode,
        window_hours,
        reward_value,
        reward_ready_flag
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        str(plug_id),
        str(decision_time),
        float(current_consumption) if current_consumption is not None else None,
        float(baseline_consumption) if baseline_consumption is not None else None,
        float(recent_diff_3h) if recent_diff_3h is not None else None,
        int(is_weekend) if is_weekend is not None else 0,
        str(time_block) if time_block is not None else "",
        ",".join(eligible_templates) if isinstance(eligible_templates, list) else str(eligible_templates),
        str(assigned_template),
        str(assignment_mode),
        int(window_hours),
        float(reward_value) if reward_value is not None else None,
        int(reward_ready_flag)
    ))

    conn.commit()
    conn.close()


def update_sqlite_reward(
    plug_id,
    decision_time,
    reward_value,
    db_path=SQLITE_DB_PATH
):
    """
    根据 plug_id + decision_time 定位一条事件，并回填 reward。

    更新后会将：
    - reward_value 写入
    - reward_ready_flag 置为 1

    返回
    ----
    int
        实际更新到的记录数
    """
    ensure_sqlite_intervention_events_table(db_path)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
    UPDATE intervention_events
    SET reward_value = ?,
        reward_ready_flag = 1
    WHERE plug_id = ?
      AND decision_time = ?
    """, (
        float(reward_value) if reward_value is not None else None,
        str(plug_id),
        str(decision_time)
    ))

    conn.commit()
    affected_rows = cursor.rowcount
    conn.close()

    if affected_rows is None:
        return 0
    return int(affected_rows)


def delete_sqlite_test_events(db_path=SQLITE_DB_PATH):
    """
    删除测试写入的 SQLite 事件。

    当前规则：
    - plug_id = 'plug_01'
    - assignment_mode = 'rule'
    - reward_ready_flag = 0

    返回
    ----
    int
        删除的记录数
    """
    ensure_sqlite_intervention_events_table(db_path)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
    DELETE FROM intervention_events
    WHERE plug_id = 'plug_01'
      AND assignment_mode = 'rule'
      AND reward_ready_flag = 0
    """)

    deleted_count = cursor.rowcount
    conn.commit()
    conn.close()
    return int(deleted_count) if deleted_count is not None else 0


def load_sqlite_intervention_events(db_path=SQLITE_DB_PATH):
    """
    读取 SQLite 中的 intervention_events 总表。

    返回
    ----
    pd.DataFrame
        正常时返回事件表；
        若读取失败，返回仅含 error 列的 DataFrame。
    """
    try:
        ensure_sqlite_intervention_events_table(db_path)

        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("""
            SELECT *
            FROM intervention_events
            ORDER BY event_id DESC
        """, conn)
        conn.close()

        return df

    except Exception as e:
        return pd.DataFrame([{"error": str(e)}])