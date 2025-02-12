import argparse
import os
from pathlib import Path
import torch
import tqdm
import pandas as pd
import yaml
import importlib
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP


from src.utility.utility import get_downsample_factors_dict


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

parser = argparse.ArgumentParser(parents=[config_parser])

parser.add_argument(
    "--fold",
    default=model_config["fold"],
    type=int,
    choices=[0, 1, 2, 3],
    help="The fold to use as the validation set.  Must train one model per fold to enable evaluation.",
)
parser.add_argument(
    "--map_at_r",
    action="store_true",
    help="Flag indicating to compute MAP@R while training",
    default=model_config["map_at_r"]
)
parser.add_argument(
    "--w_ms", default=model_config["w_ms"], type=float, help="Weight for multi-similarity loss"
)
parser.add_argument(
    "--w_ce", default=model_config["w_ce"], type=float, help="Weight for cross-entropy loss"
)
parser.add_argument(
    "--dataset_dir",
    default=dataset_config["dataset_dir"],
    type=str,
    help="Path to directory to store GazeBase data files",
)
parser.add_argument(
    "--log_dir",
    default=model_config["log_dir"],
    type=str,
    help="Path to directory to store Tensorboard logs",
)
parser.add_argument(
    "--ckpt_dir",
    default=model_config["ckpt_dir"],
    type=str,
    help="Path to directory to store model checkpoints",
)

parser.add_argument(
    "--seq_len",
    default=model_config["seq_len"],
    type=int,
    help="Length of input sequences (prior to downsampling)",
)
parser.add_argument(
    "--batch_classes",
    default=model_config["batch_classes"],
    type=int,
    help="Number of classes sampled per minibatch",
)
parser.add_argument(
    "--batch_samples",
    default=model_config["batch_samples"],
    type=int,
    help="Number of sequences sampled per class per minibatch",
)
parser.add_argument(
    "--batch_size_for_testing",
    default=model_config["batch_size_for_testing"],
    type=int,
    help="Number of sequences sampled per class per minibatch",
)
parser.add_argument(
    "--ds",
    default=model_config["ds"],
    type=int,
    choices=[1, 2, 4, 8, 20, 32],
    help="Downsample factor.  Supported factors are 1 (1000 Hz), 2 (500 Hz), 4 (250 Hz), 8 (125 Hz), 20 (50 Hz), or 32 (31.25 Hz).",
)
parser.add_argument(
    "--device",
    action="store_true",
    help="Flag indicating to use the CPU instead of a GPU",
    default=model_config["device"]
)

parser.add_argument(
    "--degrade_precision",
    action="store_true",
    help="Flag indicating to degrade spatial precision by adding white noise with SD=0.5 deg",
    default=model_config["degrade_precision"]
)

parser.add_argument(
    "--embed_dir",
    default=model_config["embed_dir"],
    type=str,
    help="Path to directory to store embeddings"
)
parser.add_argument(
    "--num_workers",
    default=dataset_config["num_workers"],
    type=str,
    help="Number of dataloader workers"
)
parser.add_argument(
    "--cache_size",
    default=dataset_config["cache_size"],
    type=int,
    help="Cache size of dataloader"
)
args = parser.parse_args()

if __name__ == "__main__": 
    
    device = args.device
    module = importlib.import_module(dataset_config["dataset_file"])
    Dataset = getattr(module, "Dataset")
    
    module = importlib.import_module(model_config["model_file"])
    Model = getattr(module, "Model")

    checkpoint_stem = (
        model_config["model_name"] 
        + f"_f{args.fold}"
    )
    
    checkpoint_path = Path(args.ckpt_dir) / (checkpoint_stem + f"_epoch={model_config['epochs']-1}.ckpt")
    downsample_factors = get_downsample_factors_dict()[args.ds]
    noise_sd = None 
    if args.degrade_precision:
        noise_sd = 0.5
    
    test_batch_size = args.batch_size_for_testing

    dataset = Dataset(
        current_fold=args.fold,
        base_dir=args.dataset_dir,
        downsample_factors=downsample_factors,
        subsequence_length_before_downsampling=args.seq_len,
        classes_per_batch=model_config["batch_classes"],
        samples_per_class=model_config["batch_samples"],
        compute_map_at_r=args.map_at_r,
        batch_size_for_testing=test_batch_size,
        noise_sd=noise_sd,
        num_workers=args.num_workers,
        cache_size=args.cache_size
    )

    dataset.prepare_data()
    dataset.setup(stage="test")
    print("Test set mean:", dataset.zscore_mn)
    print("Test set SD:", dataset.zscore_sd)

    test_loader = dataset.test_dataloader()
    
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    print(f"Start running basic DDP example on rank {rank}.")
    device_id = rank % torch.cuda.device_count()
    
    model = Model(
        n_classes=dataset.n_classes,
        embeddings_filename=checkpoint_stem + ".csv",
        embeddings_dir=args.embed_dir,
        w_metric_loss=args.w_ms,
        w_class_loss=args.w_ce,
        compute_map_at_r=args.map_at_r,
    ).to(device)
    
    model = DDP(model, device_ids=[device_id])
    model.load_state_dict(torch.load(checkpoint_path, map_location=args.device))


    # testing 
    model.eval()
    with torch.no_grad():
        embeddings = []
        metadata = []
        with tqdm.tqdm(total=len(test_loader.dataset), desc="") as pbar:
            for batch, (x, y) in enumerate(test_loader):
                x, y = x.to(device), y.to(device)
                pred = model.module.embedder(x)
                pred, x, y = pred.detach().cpu(), x.detach().cpu(), y.detach().cpu()
                embeddings.append(pred)
                metadata.append(y)
    
                # Update progress bar
                pbar.update(len(x))

        embeddings = torch.cat(embeddings, dim=0).numpy()
        metadata = torch.cat(metadata, dim=0).numpy()

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
        path = model.module.embeddings_path.with_name(
            "test"  + "_" + model.module.embeddings_path.name
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)