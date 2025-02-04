def protocol(df, TASK_TO_NUM, task, round_n):
    """Accepts df representing test centroids, outputs authentican and enrol idx"""
    print(df["nb_round"].unique())
    is_round_1 = df["nb_round"].isin([1, 2])
    is_round_n = df["nb_round"].isin([3, 4])
    enroll_idx = df.index[is_round_1]
    auth_idx = df.index[is_round_n]
    return enroll_idx, auth_idx


