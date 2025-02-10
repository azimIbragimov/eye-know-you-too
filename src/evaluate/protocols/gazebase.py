def protocol(df, TASK_TO_NUM, task, round_n):
    is_session_1 = df["nb_session"] == 1
    is_session_2 = df["nb_session"] == 2
    is_round_1 = df["nb_round"] == 1
    is_round_n = df["nb_round"] == round_n
    is_task = df["nb_task"] == TASK_TO_NUM[task]
    
    enroll_idx = df.index[is_round_1 & is_session_1 & is_task]
    auth_idx = df.index[is_round_n & is_session_2 & is_task]
    return enroll_idx, auth_idx