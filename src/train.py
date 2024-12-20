import argparse
import os
from pathlib import Path
import torch
import tqdm
import yaml

from data.gazebase import GazeBase
from models.modules import EyeKnowYouToo

# Set up argparse
parser = argparse.ArgumentParser(description="Script with YAML configuration support")

parser.add_argument("--config", default="./configs/default.yaml", type=str, help="Path to the YAML file containing default values")

# Define all arguments
parser.add_argument("--fold", default=None, type=int, choices=[0, 1, 2, 3], help="The fold to use as the validation set")
parser.add_argument("--map_at_r", default=None, action="store_true", help="Flag indicating to compute MAP@R while training")
parser.add_argument("--w_ms", default=None, type=float, help="Weight for multi-similarity loss")
parser.add_argument("--w_ce", default=None, type=float, help="Weight for cross-entropy loss")
parser.add_argument("--gazebase_dir", default=None, type=str, help="Path to directory to store GazeBase data files")
parser.add_argument("--log_dir", default=None, type=str, help="Path to directory to store Tensorboard logs")
parser.add_argument("--ckpt_dir", default=None, type=str, help="Path to directory to store model checkpoints")
parser.add_argument("--seq_len", default=None, type=int, help="Length of input sequences")
parser.add_argument("--batch_classes", default=None, type=int, help="Number of classes sampled per minibatch")
parser.add_argument("--batch_samples", default=None, type=int, help="Number of sequences sampled per class per minibatch")
parser.add_argument("--ds", default=None, type=int, choices=[1, 2, 4, 8, 20, 32], help="Downsample factor")
parser.add_argument("--device", default=None, type=str, help="Flag indicating to use the CPU instead of a GPU")
parser.add_argument("--degrade_precision", default=None, action="store_true", help="Flag indicating to degrade spatial precision")
parser.add_argument("--embed_dir", default=None, type=str, help="Path to directory to store embeddings")

# Parse CLI arguments
args = parser.parse_args()

# Load defaults from YAML configuration
with open(args.config, 'r') as f:
    defaults = yaml.safe_load(f)

# Update defaults from JSON if no argument is provided
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
    checkpoint_path = Path(args.ckpt_dir) / (checkpoint_stem + ".ckpt")
    print(checkpoint_path)


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


    dataset = GazeBase(
        current_fold=args.fold,
        base_dir=args.gazebase_dir,
        downsample_factors=downsample_factors,
        subsequence_length_before_downsampling=args.seq_len,
        classes_per_batch=args.batch_classes,
        samples_per_class=args.batch_samples,
        compute_map_at_r=args.map_at_r,
        batch_size_for_testing=None,
        noise_sd=noise_sd,
        num_workers=args.num_workers,
        n_folds=args.n_folds
    )


    dataset.prepare_data()
    dataset.setup(stage="fit")
    print("Train set mean:", dataset.zscore_mn)
    print("Train set SD:", dataset.zscore_sd)

    train_loader, validation_loader = dataset.train_dataloader(), dataset.val_dataloader()


    model = EyeKnowYouToo(
        n_classes=dataset.n_classes,
        embeddings_filename=checkpoint_stem + ".csv",
        embeddings_dir=args.embed_dir,
        w_metric_loss=args.w_ms,
        w_class_loss=args.w_ce,
        compute_map_at_r=args.map_at_r,
    ).to(device)


    opt = torch.optim.Adam(model.parameters())
    sched = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=opt,
            max_lr=0.01,
            epochs=args.epochs,
            steps_per_epoch=1,
            cycle_momentum=False,
            div_factor=100.0,
            final_div_factor=1000.0,
   )

    # training
    print(validation_loader)
    size = len(train_loader.dataset) + len(validation_loader[0].dataset)

    for epoch in range(args.epochs):
        model.train()
        with tqdm.tqdm(total=size, desc="") as pbar:
            for batch, (inputs, metadata) in enumerate(train_loader):
                opt.zero_grad()

                inputs, metadata = inputs.to(device), metadata.to(device)
                embeddings = model.embedder(inputs)
    
                labels = metadata[:, 0]
                metric_loss = model.metric_step(embeddings, labels)
                class_loss = model.class_step(embeddings, labels)
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
                for batch, (inputs, metadata) in enumerate(validation_loader[0]):
                    inputs, metadata = inputs.to(device), metadata.to(device)
                    embeddings = model.embedder(inputs)
            
                    labels = metadata[:, 0]
                    metric_loss = model.metric_step(embeddings, labels)
                    class_loss = model.class_step(embeddings, labels)
                    total_loss = metric_loss + class_loss

                    # Update progress bar
                    valid_loss.append(total_loss)
                    pbar.update(len(inputs))


                pbar.set_description(f"Epoch {epoch+1}, Validation Loss: {sum(valid_loss)/len(valid_loss):.6f},  ")
        
        torch.save(model.state_dict(),  os.path.join(args.ckpt_dir, f"f{args.fold}_epoch={epoch}.ckpt") )
