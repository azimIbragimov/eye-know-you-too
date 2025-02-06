def protocol(df, TASK_TO_NUM, task, round_n):
    """Accepts df representing test centroids, outputs authentication and enrollment indices."""
    length_of_recording = df.shape[0]    
    # Correct Boolean masks
    is_round_1 = df["nb_subsequence"] == 0
    is_round_n = df["nb_subsequence"] == 2
    
    # Extract indices using .loc[]
    enroll_idx = df.loc[is_round_1].index
    auth_idx = df.loc[is_round_n].index
    print(enroll_idx, auth_idx)

    return enroll_idx, auth_idx