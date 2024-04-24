
import argparse
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import pandas as pd

from data.datasets import TASK_TO_NUM
from metrics import dprime, roc

NUM_TO_TASK = {v: k for k, v in TASK_TO_NUM.items()}


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
    drop_cols = ["nb_subsequence", "exclude"]

    exclude = (df.loc[:, "nb_subsequence"] >= n) | (df.loc[:, "exclude"] == 1)
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


parser = argparse.ArgumentParser()
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
    default="./embeddings",
    type=str,
    help="Directory where embedding files are stored",
)
parser.add_argument(
    "--plot_dir",
    default="./figures",
    type=str,
    help="Directory to store plotted figures",
)
parser.add_argument(
    "--n_seq",
    default=1,
    type=int,
    help="How many subsequence embeddings to use for centroid embeddings",
)
parser.add_argument(
    "--round",
    default=1,
    type=int,
    choices=[1, 2, 3, 4, 5, 6, 7, 8, 9],
    help="The recording round to use for authentication",
)
parser.add_argument(
    "--task",
    default="TEX",
    type=str,
    choices=list(TASK_TO_NUM.keys()),
    help="The task to use for enrollment and authentication",
)
parser.add_argument(
    "--bootstrap",
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
    action="store_true",
    help="Flag indicating to compute results involving the validation set",
)
parser.add_argument(
    "--plot",
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

fold_names = [model_name + f"_f{fold}.csv" for fold in range(1)]
#fold_names = [model_name + ".csv"]
files_dict = {
    k: [embed_dir / "_".join([k, name]) for name in fold_names]
    for k in ("val", "test", "judo")
}
has_judo = files_dict["judo"][0].exists()

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

# Compute bootstrapped similarity scores
num_samples = 1000
num_observations = 20000
genuine_scores = y_score_flat[y_true_flat == 1]
impostor_scores = y_score_flat[y_true_flat == 0]
genuine_bootstrap = np.random.choice(
    genuine_scores, size=(num_samples, num_observations), replace=True
)
impostor_bootstrap = np.random.choice(
    impostor_scores, size=(num_samples, num_observations), replace=True
)
y_score_bootstrap = np.concatenate(
    (genuine_bootstrap, impostor_bootstrap), axis=1
)
y_true_bootstrap = np.concatenate(
    (
        np.ones_like(genuine_bootstrap),
        np.zeros_like(impostor_bootstrap),
    ),
    axis=1,
)

if not skip_bootstrap:
    print("Bootstrapping...")
    eer_bootstrap = []
    dprime_bootstrap = []
    fnr_bootstrap = {fpr_inv: [] for fpr_inv in (10, 100, 1000, 10000)}
    for scores, labels in zip(y_score_bootstrap, y_true_bootstrap):
        sample_fpr, sample_tpr, _ = roc.build_roc(labels, scores)

        sample_eer = roc.estimate_eer(sample_fpr, sample_tpr)
        eer_bootstrap.append(sample_eer)

        sample_d = dprime.decidability_index(labels, scores)
        dprime_bootstrap.append(sample_d)

        for fpr_inv, fnr_list in fnr_bootstrap.items():
            fixed_fpr = 1 / fpr_inv
            sample_fnr = roc.estimate_fnr_at_fpr(
                sample_fpr, sample_tpr, fixed_fpr
            )
            fnr_list.append(sample_fnr)

    mean_eer = np.mean(eer_bootstrap)
    sd_eer = np.std(eer_bootstrap)
    print(
        "Bootstrapped EER (%):",
        f"{100 * mean_eer:.4f} +/- {100 * sd_eer:.4f}",
    )

    mean_d = np.mean(dprime_bootstrap)
    sd_d = np.std(dprime_bootstrap)
    print(f"Bootstrapped d': {mean_d:.4f} +/- {sd_d:.4f}")

    for fpr_inv, fnr_list in fnr_bootstrap.items():
        mean_fnr = np.mean(fnr_list)
        sd_fnr = np.std(fnr_list)
        print(
            f"Bootstrapped FRR (%) at 1-in-{fpr_inv} FAR:",
            f"{100 * mean_fnr:.4f} +/- {100 * sd_fnr:.4f}",
        )

if not skip_val:
    if nb_round >= 6:
        print(f"\nWarning: R{nb_round} data is not included in the val set")
    else:
        # Compute FAR, FRR on test when fitting threshold on val (per fold)
        for nb_fold in range(4):
            print(
                f"\nR{nb_round} {task} (no ensembling, val fold {nb_fold}),",
                f"first {n_seq} subsequences",
            )
            print("-" * 20)

            # Fit threshold on validation set
            val_embeddings = pd.read_csv(files_dict["val"][nb_fold])
            val_centroids = aggregate_embeddings(val_embeddings, n_seq)

            df = val_centroids
            is_round_1 = df["nb_round"] == 1
            is_round_n = df["nb_round"] == nb_round
            is_session_1 = df["nb_session"] == 1
            is_session_2 = df["nb_session"] == 2
            is_task = df["nb_task"] == TASK_TO_NUM[task]

            enroll_idx = df.index[is_round_1 & is_session_1 & is_task]
            auth_idx = df.index[is_round_n & is_session_2 & is_task]

            y_score, y_true = pairwise_similarities(df, enroll_idx, auth_idx)
            y_score_np = y_score.to_numpy()
            y_true_np = y_true.to_numpy()

            y_score_flat = y_score_np.flatten()
            y_true_flat = y_true_np.flatten().astype(int)
            fpr, tpr, thresholds = roc.build_roc(y_true_flat, y_score_flat)
            eer = roc.estimate_eer(fpr, tpr)
            val_threshold = roc.estimate_threshold_at_fpr(fpr, thresholds, eer)

            # Fit threshold on test set
            test_embeddings = pd.read_csv(files_dict["test"][nb_fold])
            test_centroids = aggregate_embeddings(test_embeddings, n_seq)

            df = test_centroids
            is_round_1 = df["nb_round"] == 1
            is_round_n = df["nb_round"] == nb_round
            is_session_1 = df["nb_session"] == 1
            is_session_2 = df["nb_session"] == 2
            is_task = df["nb_task"] == TASK_TO_NUM[task]

            enroll_idx = df.index[is_round_1 & is_session_1 & is_task]
            auth_idx = df.index[is_round_n & is_session_2 & is_task]

            y_score, y_true = pairwise_similarities(df, enroll_idx, auth_idx)
            y_score_np = y_score.to_numpy()
            y_true_np = y_true.to_numpy()

            y_score_flat = y_score_np.flatten()
            y_true_flat = y_true_np.flatten().astype(int)
            fpr, tpr, thresholds = roc.build_roc(y_true_flat, y_score_flat)
            eer = roc.estimate_eer(fpr, tpr)
            test_threshold = roc.estimate_threshold_at_fpr(
                fpr, thresholds, eer
            )

            # Results fit on test set (d', EER)
            d = dprime.decidability_index(y_true_flat, y_score_flat)
            print(f"d' on test set: {d:.4f}")
            print(f"Threshold (fit on test): {test_threshold:.4f}")
            print(f"EER (%) w/ test threshold: {100 * eer:.4f}")

            # Apply threshold from validation set (FRR, FAR)
            y_pred = y_score_flat >= val_threshold
            fp = np.sum(y_pred & ~y_true_flat)
            fn = np.sum(~y_pred & y_true_flat)
            tp = np.sum(y_pred & y_true_flat)
            tn = np.sum(~y_pred & ~y_true_flat)
            fpr = fp / (fp + tn)
            fnr = fn / (fn + tp)
            print(f"Threshold (fit on val): {val_threshold:.4f}")
            print(f"FRR (%) w/ val threshold: {100 * fnr:.4f}")
            print(f"FAR (%) w/ val threshold: {100 * fpr:.4f}")

if not skip_judo:
    if not has_judo:
        print("\nWarning: no embeddings found for JuDo1000")
    else:
        print(f"\nJuDo1000, first {n_seq} subsequences")
        print("-" * 20)

        # Concatenate embeddings across folds
        judo_embeddings = [pd.read_csv(f) for f in files_dict["judo"]]
        judo_ensemble_embeddings = concatenate_embeddings(judo_embeddings)
        judo_ensemble_centroids = aggregate_embeddings(
            judo_ensemble_embeddings, n_seq
        )

        # Build enrollment and authentication sets
        df = judo_ensemble_centroids
        is_session_1 = df["nb_session"] == 1
        is_session_2 = df["nb_session"] == 2

        enroll_idx = df.index[is_session_1]
        auth_idx = df.index[is_session_2]

        # Compute similarity matrix
        y_score, y_true = pairwise_similarities(df, enroll_idx, auth_idx)
        y_score_np = y_score.to_numpy()
        y_true_np = y_true.to_numpy()
        print("P:", np.sum(y_true_np))
        print("N:", np.sum(~y_true_np))

        # Equal error rate (EER)
        y_score_flat = y_score_np.flatten()
        y_true_flat = y_true_np.flatten().astype(int)
        fpr, tpr, thresholds = roc.build_roc(y_true_flat, y_score_flat)
        eer = roc.estimate_eer(fpr, tpr)
        eer_threshold = roc.estimate_threshold_at_fpr(fpr, thresholds, eer)
        print(f"EER (%): {100 * eer:.4f}")
        print(f"EER threshold: {eer_threshold:.4f}")

        # Decidability index (d')
        d = dprime.decidability_index(y_true_flat, y_score_flat)
        print(f"d': {d:.4f}")

        # FRR @ FAR
        for fpr_inv in (10, 100, 1000, 10000):
            fixed_fpr = 1 / fpr_inv
            fnr = roc.estimate_fnr_at_fpr(fpr, tpr, fixed_fpr)
            print(f"FRR (%) at 1-in-{fpr_inv} FAR: {100 * fnr:.4f}")

if not skip_plot:
    try:
        import matplotlib.pyplot as plt
        from umap import UMAP
    except ImportError:
        print("Warning: missing plotting libraries")
    else:
        print("\nPlotting figures")
        print("-" * 20)

        plot_dir.mkdir(parents=True, exist_ok=True)

        # Genuine vs impostor scores for the given round, task, n_seq
        # ---
        bins = np.linspace(-1.0, 1.0, 101)
        plt.figure(figsize=(3, 2), constrained_layout=True, dpi=300)
        impostor_scores = simdist_score[simdist_true == 0]
        genuine_scores = simdist_score[simdist_true == 1]
        plt.hist(
            impostor_scores,
            bins,
            density=True,
            histtype="step",
            linewidth=1,
            facecolor=(1, 0.65, 0, 0.5),
            fill=True,
            hatch="\\\\",
            edgecolor="k",
            label="Impostor scores",
        )
        plt.hist(
            genuine_scores,
            bins,
            density=True,
            histtype="step",
            linewidth=1,
            facecolor=(0, 0, 1, 0.5),
            fill=True,
            hatch="//",
            edgecolor="k",
            label="Genuine scores",
        )
        # We plot impostor scores first so they are drawn behind the
        # genuine scores, but we want the genuine label to appear first
        # in the legend for stylistic reasons
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(reversed(handles), reversed(labels), loc="upper left")
        plt.xlabel("Similarity")
        plt.ylabel("Density")
        plt.savefig(plot_dir / f"simdist_{model_name}.png")
        plt.close("all")

        # ROC curve with bootstrapping for the given round, task, n_seq
        # ---
        color_map = plt.get_cmap("tab10")
        color = color_map(0)
        mean_fpr = np.linspace(0.0, 1.0, 10001)
        plt.figure(figsize=(3, 2), constrained_layout=True, dpi=300)

        # Curve showing where FRR = FAR (i.e., EER)
        plt.plot(mean_fpr, mean_fpr, "--k", linewidth=1)
        plt.ylim(-0.01, 0.51)  # for higher max FRR, use (-0.05, 1.05)
        plt.xlim(1e-4, 1.0)
        plt.xlabel("FAR")
        plt.ylabel("FRR")

        # Curve showing mean FRR (i.e., 1 - TPR) at each FAR
        tprs = []
        for y, x in zip(y_true_bootstrap, y_score_bootstrap):
            fpr, tpr, _ = roc.build_roc(y_true=y, y_score=x)
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
        tprs = np.array(tprs)
        mean_tpr = np.mean(tprs, axis=0)
        plt.plot(mean_fpr, 1.0 - mean_tpr, "-", color=color)

        # Shade region around mean FRR curve to visualize +/- 1 SD
        sd_tpr = np.std(tprs, axis=0)
        tpr_upper = np.minimum(mean_tpr + sd_tpr, 1.0)
        tpr_lower = np.maximum(mean_tpr - sd_tpr, 0.0)
        plt.fill_between(
            mean_fpr,
            1.0 - tpr_lower,
            1.0 - tpr_upper,
            color=color,
            alpha=0.3,
            edgecolor=None,
        )

        plt.xscale("log")
        plt.savefig(plot_dir / f"roc_{model_name}.png")
        plt.close("all")

        # UMAP across all valid embeddings for 10 subjects
        # ---
        df = test_ensemble_embeddings
        ten_r9_subjects = df[df["nb_round"] == 9]["nb_subject"].unique()[:10]
        include = df["exclude"] == 0
        include &= df["nb_subject"].isin(ten_r9_subjects)
        embed_cols = [x for x in df.columns if x.startswith("embed_dim")]
        df = df.loc[include, :]
        embeddings = df[embed_cols].to_numpy()
        subjects = df["nb_subject"].to_numpy()
        labels = np.digitize(subjects, bins=ten_r9_subjects)

        mapper = UMAP(
            metric="cosine",
            n_neighbors=30,
            min_dist=0.1,
            densmap=True,
            verbose=True,
            random_state=42,  # match our figure as closely as possible
        )
        densmap_embeddings = mapper.fit_transform(embeddings)
        plt.figure(figsize=(3, 3), constrained_layout=True, dpi=300)
        plt.scatter(
            densmap_embeddings[:, 0],
            densmap_embeddings[:, 1],
            c=labels,
            cmap="tab10",
            alpha=0.5,
            marker=".",
            edgecolors="none",
        )
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.savefig(plot_dir / f"densmap_{model_name}.png")
        plt.close("all")
