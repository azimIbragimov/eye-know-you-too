import argparse
import os
from pathlib import Path
import torch

parser = argparse.ArgumentParser()
from data.gazebase import GazeBase

parser.add_argument(
    "--resume_epoch",
    default=-1,
    type=int,
    help="If resuming training from a checkpoint, the number in 'epoch=___'",
)
parser.add_argument(
    "--fold",
    default=0,
    type=int,
    choices=[0, 1, 2, 3],
    help="The fold to use as the validation set.  Must train one model per fold to enable evaluation.",
)
parser.add_argument(
    "--map_at_r",
    action="store_true",
    help="Flag indicating to compute MAP@R while training",
)
parser.add_argument(
    "--w_ms", default=1.0, type=float, help="Weight for multi-similarity loss"
)
parser.add_argument(
    "--w_ce", default=0.1, type=float, help="Weight for cross-entropy loss"
)
parser.add_argument(
    "--gazebase_dir",
    default="./data/gazebase_v3",
    type=str,
    help="Path to directory to store GazeBase data files",
)
parser.add_argument(
    "--log_dir",
    default="./logs",
    type=str,
    help="Path to directory to store Tensorboard logs",
)
parser.add_argument(
    "--ckpt_dir",
    default="./output",
    type=str,
    help="Path to directory to store model checkpoints",
)
parser.add_argument(
    "--embed_dir",
    default="./embeddings",
    type=str,
    help="Path to directory to store embeddings",
)
parser.add_argument(
    "--seq_len",
    default=5000,
    type=int,
    help="Length of input sequences (prior to downsampling)",
)
parser.add_argument(
    "--batch_classes",
    default=16,
    type=int,
    help="Number of classes sampled per minibatch",
)
parser.add_argument(
    "--batch_samples",
    default=16,
    type=int,
    help="Number of sequences sampled per class per minibatch",
)
parser.add_argument(
    "--ds",
    default=1,
    type=int,
    choices=[1, 2, 4, 8, 20, 32],
    help="Downsample factor.  Supported factors are 1 (1000 Hz), 2 (500 Hz), 4 (250 Hz), 8 (125 Hz), 20 (50 Hz), or 32 (31.25 Hz).",
)
parser.add_argument(
    "--cpu",
    action="store_true",
    help="Flag indicating to use the CPU instead of a GPU",
)

parser.add_argument(
    "--batch_size_for_testing",
    default=-1,
    type=int,
    help="Override the batch size with this value when `--mode=test`",
)
parser.add_argument(
    "--degrade_precision",
    action="store_true",
    help="Flag indicating to degrade spatial precision by adding white noise with SD=0.5 deg",
)
args = parser.parse_args()





if __name__ == "__main__": 

    # Hide all GPUs except the one we (maybe) want to use
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    
    checkpoint_stem = (
        "ekyt"
        + f"_t{args.seq_len}"
        + f"_ds{args.ds}"
        + f"_bc{args.batch_classes}"
        + f"_bs{args.batch_samples}"
        + f"_wms{round(10.0 * args.w_ms):02d}"
        + f"_wce{round(10.0 * args.w_ce):02d}"
        + ("_degraded" if args.degrade_precision else "_normal")
        + f"_f{args.fold}"
    )
    checkpoint_path = Path(args.ckpt_dir) / (checkpoint_stem + ".ckpt")


    downsample_factors_dict = {
        1: [],
        2: [2],
        4: [4],
        8: [8],
        20: [4, 5],
        32: [8, 4],
    }
    downsample_factors = downsample_factors_dict[args.ds]


    noise_sd = None
    if args.degrade_precision:
        noise_sd = 0.5


    test_batch_size = args.batch_size_for_testing
    if test_batch_size == -1:
        test_batch_size = None

    dataset = GazeBase(
        current_fold=args.fold,
        base_dir=args.gazebase_dir,
        downsample_factors=downsample_factors,
        subsequence_length_before_downsampling=args.seq_len,
        classes_per_batch=args.batch_classes,
        samples_per_class=args.batch_samples,
        compute_map_at_r=args.map_at_r,
        batch_size_for_testing=test_batch_size,
        noise_sd=noise_sd
    )


    dataset.prepare_data()
    dataset.setup(stage="fit")
    print("Train set mean:", dataset.zscore_mn)
    print("Train set SD:", dataset.zscore_sd)

    train_loader, validation_loader = dataset.train_dataloader(), dataset.test_dataloader()





