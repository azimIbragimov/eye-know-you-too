import argparse
import importlib
import os
from pathlib import Path
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP


import torch
import tqdm

from src.utility.utility import load_config
from src.utility.utility import get_downsample_factors_dict


config_parser = argparse.ArgumentParser(add_help=False)
config_parser.add_argument(
    "--config", 
    default="config/lohr-22.yaml", 
    type=str,
    help="Config file for the current experiment"
)
args, remaining_args = config_parser.parse_known_args()

config = load_config(args.config)
model_config = load_config(config["model_config"])
dataset_config = load_config(config["dataset_config"])

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
    "--w_ms", 
    default=model_config["w_ms"], 
    type=float, 
    help="Weight for multi-similarity loss"
)
parser.add_argument(
    "--w_ce", 
    default=model_config["w_ce"], 
    type=float, 
    help="Weight for cross-entropy loss"
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
    
    checkpoint_path = Path(args.ckpt_dir) / (checkpoint_stem + ".ckpt")
    
    downsample_factors = get_downsample_factors_dict()[args.ds]


    noise_sd = None
    if args.degrade_precision:
        noise_sd = 0.5

    dataset = Dataset(
        current_fold=args.fold,
        base_dir=args.dataset_dir,
        downsample_factors=downsample_factors,
        subsequence_length_before_downsampling=args.seq_len,
        classes_per_batch=args.batch_classes,
        samples_per_class=args.batch_samples,
        compute_map_at_r=args.map_at_r,
        batch_size_for_testing=None,
        noise_sd=noise_sd,
        num_workers=args.num_workers,
        cache_size=args.cache_size
    )

    dataset.prepare_data()
    dataset.setup(stage="fit", )
    print("Train set mean:", dataset.zscore_mn)
    print("Train set SD:", dataset.zscore_sd)

    train_loader, validation_loader = dataset.train_dataloader(), dataset.val_dataloader()
    
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
    ).to(device_id)
    model = DDP(model, device_ids=[device_id])
    
    
    opt = torch.optim.Adam(model.parameters())
    sched = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=opt,
            max_lr=0.01,
            epochs=model_config["epochs"],
            steps_per_epoch=1,
            cycle_momentum=False,
            div_factor=100.0,
            final_div_factor=1000.0,
   )

    # training
    size = len(train_loader.dataset) + len(validation_loader.dataset)

    for epoch in range(model_config["epochs"]):
        model.train()
        with tqdm.tqdm(total=size, desc="") as pbar:
            for batch, (inputs, metadata) in enumerate(train_loader):
                opt.zero_grad()

                inputs, metadata = inputs.to(device), metadata.to(device)
                embeddings = model.module.embedder(inputs)
    
                labels = metadata[:, 0]
                metric_loss = model.module.metric_step(embeddings, labels)
                class_loss = model.module.class_step(embeddings, labels)
                total_loss = metric_loss + class_loss
    
                # Backpropagation
                total_loss.backward()
                opt.step()
    
                # Update progress bar
                pbar.set_description(f"Epoch {epoch+1}, Loss: {total_loss.item():.6f},  ")
                pbar.update(len(inputs))

            sched.step()

            # validation
            valid_loss = []
            model.eval()
            with torch.no_grad():
                for batch, (inputs, metadata) in enumerate(validation_loader):
                    inputs, metadata = inputs.to(device), metadata.to(device)
                    embeddings = model.module.embedder(inputs)
            
                    labels = metadata[:, 0]
                    metric_loss = model.module.metric_step(embeddings, labels)
                    class_loss = model.module.class_step(embeddings, labels)
                    total_loss = metric_loss + class_loss

                    # Update progress bar
                    valid_loss.append(total_loss)
                    pbar.update(len(inputs))


                pbar.set_description(f"Epoch {epoch+1}, Validation Loss: {sum(valid_loss)/len(valid_loss):.6f},  ")
        
        torch.save(model.state_dict(), checkpoint_path.with_name(checkpoint_stem + f"_epoch={epoch}.ckpt"))
