
import argparse
from pathlib import Path
from typing import Sequence, Tuple
import yaml
import importlib


import numpy as np
import pandas as pd

from metrics import dprime, roc



def concatenate_embeddings(df_list: Sequence[pd.DataFrame]) -> pd.DataFrame:
    out_df = df_list[0]
    embed_cols = [x for x in out_df.columns if x.startswith("embed_dim")]
    embed_dim = len(embed_cols)

    embed_col_format = "embed_dim_{:03d}"
    join_cols = [
        "nb_round",
        "nb_subject",
        "nb_session",
        "nb_task",
        "nb_subsequence",
    ]
    drop_cols = [x for x in out_df.columns if x not in embed_cols + join_cols]

    for i, df in enumerate(df_list[1:], start=1):
        dim_start = i * embed_dim
        new_cols = [
            embed_col_format.format(x + dim_start) for x in range(embed_dim)
        ]
        cols_map = dict(zip(embed_cols, new_cols))
        new_df = df.rename(columns=cols_map).drop(columns=drop_cols)
        out_df = out_df.merge(new_df, how="outer", on=join_cols)
    return out_df


def aggregate_embeddings(df: pd.DataFrame, n: int) -> pd.DataFrame:
    group_cols = ["nb_round", "nb_subject", "nb_session", "nb_task"]
    drop_cols = ["nb_subsequence"]

    exclude = (df.loc[:, "nb_subsequence"] >= n)
    exclude_idx = df.index[exclude]
    df = df.drop(index=exclude_idx).drop(columns=drop_cols)

    embed_cols = [x for x in df.columns if x.startswith("embed_dim")]
    centroid_embeddings = (
        df.groupby(group_cols)[embed_cols].agg("mean").reset_index()
    )
    return centroid_embeddings


def pairwise_similarities(
    df: pd.DataFrame, enroll_idx: pd.Index, auth_idx: pd.Index
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    embed_cols = [x for x in df.columns if x.startswith("embed_dim")]

    enroll_subjects = df.loc[enroll_idx, "nb_subject"].to_numpy()
    enroll_mat = df.loc[enroll_idx, embed_cols].to_numpy(np.float32)
    enroll_norm = np.linalg.norm(enroll_mat, axis=1, keepdims=True)

    auth_subjects = df.loc[auth_idx, "nb_subject"].to_numpy()
    auth_mat = df.loc[auth_idx, embed_cols].to_numpy(np.float32)
    auth_norm = np.linalg.norm(auth_mat, axis=1, keepdims=True)

    similarities = np.dot(enroll_mat, auth_mat.T) / (enroll_norm * auth_norm.T)
    genuine = enroll_subjects[:, np.newaxis] == auth_subjects[np.newaxis, :]

    similarities = pd.DataFrame(
        similarities, index=enroll_subjects, columns=auth_subjects
    )
    genuine = pd.DataFrame(
        genuine, index=enroll_subjects, columns=auth_subjects
    )
    return similarities, genuine


config_parser = argparse.ArgumentParser(add_help=False)

config_parser.add_argument(
    "--config",
    default="config/lohr-22.yaml",
    type=str,
    help="Config file for the current experiment",
)
args, remaining_args = config_parser.parse_known_args()

with open(args.config) as file:
    config = yaml.safe_load(file)
print(config)

with open(config["model_config"]) as file:
    model_config = yaml.safe_load(file)
print(model_config)

with open(config["dataset_config"]) as file:
    dataset_config = yaml.safe_load(file)
print(dataset_config)


module = importlib.import_module(dataset_config["dataset_file"])
Dataset = getattr(module, "Dataset")()
TASK_TO_NUM = Dataset.TASK_TO_NUM
NUM_TO_TASK = {v: k for k, v in TASK_TO_NUM.items()}


parser = argparse.ArgumentParser(parents=[config_parser])

parser.add_argument(
    "--model",
    default="ekyt_t5000_ds1_bc16_bs16_wms10_wce01_normal",
    type=str,
    help=(
        "The common part of the embedding filenames"
        + " (i.e., the name of the model without the fold)"
    ),
)
parser.add_argument(
    "--embed_dir",
    default=model_config["embed_dir"],
    type=str,
    help="Path to directory to store embeddings"
)

parser.add_argument(
    "--plot_dir",
    default=model_config["plot_dir"],
    type=str,
    help="Directory to store plotted figures",
)
parser.add_argument(
    "--n_seq",
    default=model_config["n_seq"],
    type=int,
    help="How many subsequence embeddings to use for centroid embeddings",
)
parser.add_argument(
    "--round",
    default=model_config["round"],
    type=int,
    choices=[1, 2, 3, 4, 5, 6, 7, 8, 9],
    help="The recording round to use for authentication",
)
parser.add_argument(
    "--task",
    default=model_config["task"],
    type=str,
    choices=list(TASK_TO_NUM.keys()),
    help="The task to use for enrollment and authentication",
)
parser.add_argument(
    "--bootstrap",
    default=model_config["bootstrap"],
    action="store_true",
    help="Flag indicating to compute results involving bootstrapping",
)
parser.add_argument(
    "--judo",
    action="store_true",
    help="Flag indicating to compute results involving JuDo1000",
)
parser.add_argument(
    "--val",
    default=model_config["val"],
    action="store_true",
    help="Flag indicating to compute results involving the validation set",
)
parser.add_argument(
    "--plot",
    default=model_config["plot"],
    action="store_true",
    help="Flag indicating to plot figures",
)

args = parser.parse_args()


model_name = args.model
embed_dir = Path(args.embed_dir)
plot_dir = Path(args.plot_dir)
n_seq = args.n_seq
nb_round = args.round
task = args.task
skip_bootstrap = not args.bootstrap
skip_judo = not args.judo
skip_val = not args.val
skip_plot = not args.plot

print("Using model:", model_name)

fold_names = [model_name + f"_f{fold}.csv" for fold in range(4)]
files_dict = {
    k: [embed_dir / "_".join([k, name]) for name in fold_names]
    for k in ("test", "val", "judo")
}

print(f"\nR{nb_round} {task} ensemble, first {n_seq} subsequences")
print("-" * 20)

# Concatenate embeddings across folds
test_embeddings = [pd.read_csv(f) for f in files_dict["test"]]
test_ensemble_embeddings = concatenate_embeddings(test_embeddings)
test_ensemble_centroids = aggregate_embeddings(test_ensemble_embeddings, n_seq)

# Build enrollment and authentication sets
df = test_ensemble_centroids
is_round_1 = df["nb_round"] == 1
is_round_n = df["nb_round"] == nb_round
is_session_1 = df["nb_session"] == 1
is_session_2 = df["nb_session"] == 2
is_task = df["nb_task"] == TASK_TO_NUM[task]

enroll_idx = df.index[is_round_1 & is_session_1 & is_task]
auth_idx = df.index[is_round_n & is_session_2 & is_task]

# Compute similarity matrix
y_score, y_true = pairwise_similarities(df, enroll_idx, auth_idx)
y_score_np = y_score.to_numpy()
y_true_np = y_true.to_numpy()
print("P:", np.sum(y_true_np))
print("N:", np.sum(~y_true_np))

# Rank-1 identification rate (also called precision at 1 (P@1))
is_enrolled = y_true_np.any(axis=0)
y_score_enrolled = y_score_np[:, is_enrolled]
y_true_enrolled = y_true_np[:, is_enrolled]

sorted_indices = np.argsort(y_score_enrolled, axis=0)
sorted_indices = sorted_indices[::-1]  # descending order
score_sorted = np.take_along_axis(y_score_enrolled, sorted_indices, axis=0)
true_sorted = np.take_along_axis(y_true_enrolled, sorted_indices, axis=0)
rank1 = true_sorted[0, :].astype(int).mean()
print(f"Rank-1 IR (%): {100 * rank1:.4f}")
print(y_score_np, y_true_np)
print(sorted_indices)
print(score_sorted)
print(true_sorted)
print(len(df["nb_subject"].unique()))

metadata_1 = pd.read_csv("/blue/vemuri/sivolgina/azim/github/eye-know-you-too/data/processed/metadata.csv")
print(len(metadata_1[metadata_1['set'] == 0]['part_id'].unique()))

# Equal error rate (EER)
y_score_flat = y_score_np.flatten()
y_true_flat = y_true_np.flatten().astype(int)
fpr, tpr, thresholds = roc.build_roc(y_true_flat, y_score_flat)
eer = roc.estimate_eer(fpr, tpr)
eer_threshold = roc.estimate_threshold_at_fpr(fpr, thresholds, eer)
print(f"EER (%): {100 * eer:.4f}")
print(f"EER threshold: {eer_threshold:.4f}")

# Save values for plotting later (since we will overwrite y_* variables)
simdist_score = y_score_flat
simdist_true = y_true_flat

# Decidability index (d')
d = dprime.decidability_index(y_true_flat, y_score_flat)
print(f"d': {d:.4f}")