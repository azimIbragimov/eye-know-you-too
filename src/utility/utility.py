import yaml

def load_config(file_path):
    with open(file_path) as file:
        return yaml.safe_load(file)
    
def get_downsample_factors_dict():
    downsample_factors_dict = {
            1: [],
            2: [2],
            4: [4],
            8: [8],
            20: [4, 5],
            32: [8, 4],
        }
    return downsample_factors_dict