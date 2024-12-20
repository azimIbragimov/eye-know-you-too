import argparse
import os
from pathlib import Path
import torch
import tqdm
import pandas as pd
import yaml

from data.gazebase import GazeBase
from models.modules import EyeKnowYouToo


parser = argparse.ArgumentParser()

parser.add_argument("--config", default="./configs/default.yaml", type=str, help="Path to the YAML file containing default values")


parser.add_argument(
    "--fold",
    default=None,
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
    "--w_ms", default=None, type=float, help="Weight for multi-similarity loss"
)
parser.add_argument(
    "--w_ce", default=None, type=float, help="Weight for cross-entropy loss"
)
parser.add_argument(
    "--gazebase_dir",
    default="./data",
    type=str,
    help="Path to directory to store GazeBase data files",
)
parser.add_argument(
    "--log_dir",
    default=None,
    type=str,
    help="Path to directory to store Tensorboard logs",
)
parser.add_argument(
    "--ckpt_dir",
    default=None,
    type=str,
    help="Path to directory to store model checkpoints",
)

parser.add_argument(
    "--seq_len",
    default=None,
    type=int,
    help="Length of input sequences (prior to downsampling)",
)
parser.add_argument(
    "--batch_samples",
    default=None,
    type=int,
    help="Number of classes sampled per minibatch",
)
parser.add_argument(
    "--batch_classes",
    default=None,
    type=int,
    help="Number of classes sampled per minibatch",
)
parser.add_argument(
    "--batch_size_for_testing",
    default=None,
    type=int,
    help="Number of sequences sampled per class per minibatch",
)
parser.add_argument(
    "--ds",
    default=None,
    type=int,
    choices=[1, 2, 4, 8, 20, 32],
    help="Downsample factor.  Supported factors are 1 (1000 Hz), 2 (500 Hz), 4 (250 Hz), 8 (125 Hz), 20 (50 Hz), or 32 (31.25 Hz).",
)
parser.add_argument(
    "--device",
    default=None,
    type=str,
    help="Flag indicating to use the CPU instead of a GPU",
)

parser.add_argument(
    "--degrade_precision",
    default=None,
    action="store_true",
    help="Flag indicating to degrade spatial precision by adding white noise with SD=0.5 deg",
)
parser.add_argument(
    "--embed_dir",
    default=None,
    type=str,
    help="Path to directory to store embeddings",
)

# Parse CLI arguments
args = parser.parse_args()

# Load defaults from YAML configuration
with open(args.config, 'r') as f:
    defaults = yaml.safe_load(f)

args_dict = vars(args)
for key, value in defaults.items():
    if key in args_dict:
        if args_dict[key] is None:
            args_dict[key] = value
    else:
        args_dict[key] = value


# You can now access the updated arguments
print(args_dict)

if __name__ == "__main__": 

    # Hide all GPUs except the one we (maybe) want to use
    device = args.device
    
    checkpoint_stem = f"ekyt_f{args.fold}"
    checkpoint_path = Path(args.ckpt_dir) / ( f"f{args.fold}_epoch={args.epochs-1}.ckpt")

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
        classes_per_batch=16,
        samples_per_class=16,
        compute_map_at_r=args.map_at_r,
        batch_size_for_testing=test_batch_size,
        noise_sd=noise_sd,
        num_workers=args.num_workers,
        n_folds=args.n_folds

    )

    dataset.prepare_data()
    dataset.setup(stage="test")
    print("Test set mean:", dataset.zscore_mn)
    print("Test set SD:", dataset.zscore_sd)

    model = EyeKnowYouToo(
        n_classes=dataset.n_classes,
        embeddings_filename=checkpoint_stem + ".csv",
        embeddings_dir=args.embed_dir,
        w_metric_loss=args.w_ms,
        w_class_loss=args.w_ce,
        compute_map_at_r=args.map_at_r,
    ).to(device)

    model.load_state_dict(torch.load(checkpoint_path))


    test_loader, full_val_loader = dataset.test_dataloader()


    # testing 
    model.eval()
    with torch.no_grad():
        embeddings = []
        metadata = []
        with tqdm.tqdm(total=len(test_loader.dataset), desc="") as pbar:
            for batch, (x, y) in enumerate(test_loader):
                x, y = x.to(device), y.to(device)
                pred = model.embedder(x)
                pred, x, y = pred.detach().cpu(), x.detach().cpu(), y.detach().cpu()
                embeddings.append(pred)
                metadata.append(y)
    
                # Update progress bar
                pbar.update(len(x))

        embeddings = torch.cat(embeddings, dim=0).numpy()
        metadata = torch.cat(metadata, dim=0).numpy()

        print(embeddings)
        print(metadata)
        print(embeddings.shape, metadata.shape)

        embed_dim = embeddings.shape[1]
        embedding_dict = {
            f"embed_dim_{i:03d}": embeddings[:, i]
            for i in range(embed_dim)
        }
        full_dict = {
            "nb_round": metadata[:, 1],
            "nb_subject": metadata[:, 2],
            "nb_session": metadata[:, 3],
            "nb_task": metadata[:, 4],
            "nb_subsequence": metadata[:, 5],
            "exclude": metadata[:, 6],
            **embedding_dict,
        }
        df = pd.DataFrame(full_dict)
        df = df.sort_values(
            by=[
                "nb_round",
                "nb_subject",
                "nb_session",
                "nb_task",
                "nb_subsequence",
            ],
            axis=0,
            ascending=True,
        )
        path = model.embeddings_path.with_name(
            "test"  + "_" + model.embeddings_path.name
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)


    # Validation
    model.eval()
    with torch.no_grad():
        embeddings = []
        metadata = []
        with tqdm.tqdm(total=len(full_val_loader.dataset), desc="") as pbar:
            for batch, (x, y) in enumerate(full_val_loader):
                x, y = x.to(device), y.to(device)
                pred = model.embedder(x)
                pred, x, y = pred.detach().cpu(), x.detach().cpu(), y.detach().cpu()
                embeddings.append(pred)
                metadata.append(y)
    
                # Update progress bar
                pbar.update(len(x))

        embeddings = torch.cat(embeddings, dim=0).numpy()
        metadata = torch.cat(metadata, dim=0).numpy()

        print(embeddings)
        print(metadata)
        print(embeddings.shape, metadata.shape)

        embed_dim = embeddings.shape[1]
        embedding_dict = {
            f"embed_dim_{i:03d}": embeddings[:, i]
            for i in range(embed_dim)
        }
        full_dict = {
            "nb_round": metadata[:, 1],
            "nb_subject": metadata[:, 2],
            "nb_session": metadata[:, 3],
            "nb_task": metadata[:, 4],
            "nb_subsequence": metadata[:, 5],
            "exclude": metadata[:, 6],
            **embedding_dict,
        }
        df = pd.DataFrame(full_dict)
        df = df.sort_values(
            by=[
                "nb_round",
                "nb_subject",
                "nb_session",
                "nb_task",
                "nb_subsequence",
            ],
            axis=0,
            ascending=True,
        )
        path = model.embeddings_path.with_name(
            "val"  + "_" + model.embeddings_path.name
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)


