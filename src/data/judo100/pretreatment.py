import argparse
import pathlib
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from ...utility.utility import load_config, get_downsample_factors_dict


import numpy as np
from scipy.signal import savgol_filter
from ..downsample import downsample_recording
from ..assign_groups import assign_groups

config_parser = argparse.ArgumentParser(add_help=False)
config_parser.add_argument(
    "--config",
    default="config/lohr-22-judo100.yaml",
    type=str,
    help="Config file for the current experiment",
)
args, remaining_args = config_parser.parse_known_args()

config = load_config(args.config)
model_config = load_config(config["model_config"])
dataset_config = load_config(config["dataset_config"])

parser = argparse.ArgumentParser(parents=[config_parser])
parser.add_argument(
    "--raw_dir",
    default=dataset_config["raw_dir"],
    type=str,
    help="Path to raw folder",
)
parser.add_argument(
    "--processed_dir",
    default=dataset_config["processed_dir"],
    type=str,
    help="Path to processed folder",
)
parser.add_argument(
    "--downsample_factors",
    default=dataset_config["downsample_factors"],
    type=int,
    help="Downsample factor that reduces frequency",
)
parser.add_argument(
    "--initial_sampling_rate_hz",
    default=dataset_config["initial_sampling_rate_hz"],
    type=int,
    help="Initial frequency of the dataset",
)
parser.add_argument(
    "--n_folds",
    default=dataset_config["n_folds"],
    type=int,
    help="Number of folds in the dataset",
)


def extract_meta_data(filename: pathlib.Path):
    """Extract metadata from filename."""
    """Example of metadata. 
    Filename: 160_3.csv
    Participant_ID: 150 (1-3th characters)
    Round: 1 (3rd character)
    """
    metadata = filename.stem
    try:
        part_id, round_num = metadata.split("_")
        return {
            "round": int(round_num),
            "part_id": int(part_id),
            "session": -1,
            "task": -1,
            "filename": str(filename),
        }
    except IndexError as E:
        print(f"Error parsing metadata from filename: {filename}, {E}")
        return None


def featurize(filename: pathlib.Path, processed_dir: pathlib.Path, downsample_factors, initial_sampling_rate_hz):
    """Generate features for a given CSV file."""
    try:
        df = pd.read_csv(filename, sep="\t")
        # Downsample recording
        gaze, ideal_sampling_rate = downsample_recording(df, downsample_factors, initial_sampling_rate_hz, ["x_left", "x_right"])
        # Compute velocity using Savitzky-Golay filter
        vel = savgol_filter(gaze, 7, 2, deriv=1, axis=0, mode="nearest")
        vel *= ideal_sampling_rate  # Convert to deg/sec
        # Save features as a NumPy file
        output_path = processed_dir / f"{filename.stem}.npy"
        np.save(output_path, vel)
    except Exception as e:
        print(f"Error processing file {filename}: {e}")


def process_file(args):
    """Process a file for metadata and features."""
    filename, processed_dir, downsample_factors, initial_sampling_rate_hz = args
    meta_data = extract_meta_data(filename)
    featurize(filename, processed_dir, downsample_factors, initial_sampling_rate_hz)
    return meta_data


if __name__ == "__main__":
    args = parser.parse_args()

    downsample_factors = get_downsample_factors_dict()[args.downsample_factors]

    # Ensure the processed directories exist
    processed_dir = pathlib.Path(args.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Path to the metadata CSV file
    meta_data_file_path = processed_dir / "metadata.csv"

    # Get all raw CSV files
    raw_files = [f for f in pathlib.Path(args.raw_dir).rglob("*.csv") if "TrialVars" not in f.name]
    # Prepare arguments for parallel processing
    process_args = [
        (file, processed_dir, downsample_factors, args.initial_sampling_rate_hz)
        for file in raw_files
    ]

    # Use multiprocessing to process files with a progress bar
    with ProcessPoolExecutor() as executor:
        results = list(
            tqdm(
                executor.map(process_file, process_args),
                total=len(process_args),
                desc="Processing Files",
            )
        )

    # Filter out None results (from failed parsing)
    metadata_list = [res for res in results if res is not None]

    # Create a DataFrame and save to CSV
    meta_data_df = pd.DataFrame(metadata_list)
    
    print(meta_data_df)
    
    # Participants of round 4 are test set
    round_4_participants = meta_data_df["part_id"].unique()[:30]  # Remove extra brackets
    print(round_4_participants)

    test_set = meta_data_df[meta_data_df["part_id"].isin(round_4_participants)]
    train_set = meta_data_df[~meta_data_df["part_id"].isin(round_4_participants)]    
    # Count number of recordings for each participant in train set
    recordings_count = train_set.groupby("part_id").size().values
    fold_to_id, grp = assign_groups(args.n_folds, train_set["part_id"].unique(), recordings_count)
    
    meta_data_df["set"] = -1 # default value    
    for n in range(0, args.n_folds):
        meta_data_df.loc[meta_data_df["part_id"].isin(fold_to_id[n]), "set"] = n
            
    print(meta_data_df.shape)
    meta_data_df.to_csv(meta_data_file_path)
    meta_data_df = pd.read_csv(meta_data_file_path)
    print(f"Metadata written to: {meta_data_file_path}")
    print(f"Train set size: {len(meta_data_df[meta_data_df['set'] != -1]['part_id'].unique())}")
    print(f"Test set size: {len(meta_data_df[meta_data_df['set'] == -1]['part_id'].unique())}")