""""
This file is used to scale the grip actions in the dataset. The actions are scaled by dividing them by 10.0 default.
Author: Yufei Duan yufeidu@kth.se
Created On: 2025-02-08

"""

import os
import h5py
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Modify HDF5 dataset for scaling.")
    parser.add_argument("--task", type=str, required=True, help="Task name (e.g., 'square').")
    parser.add_argument("--dimension", type=str, required=True, choices=["low_dim", "image"], help="Observation dimension.")
    parser.add_argument("--demo_type", type=str, default="ph", help="Demonstration type (default: 'ph').")
    parser.add_argument("--dataset_folder", type=str, default="/proj/rep-learning-robotics/users/x_yufdu/daxia/diffusion/realtime_dp/diffusion_policy/data/robomimic/datasets/", help="Base path to the dataset folder.")
    parser.add_argument("--scale", type=float, default=10.0, help="Normalization scale for the last joint (default: 10.0).")
    return parser.parse_args() 

def modify_actions_dataset(actions: np.ndarray, scale: float) -> np.ndarray:
    """
    Normalize and unwrap discontinuities in the rotation components of the 'actions' dataset.
    """
    actions = np.copy(actions)
    actions[:, -1] /= scale

    if actions.shape[1] == 14:
        actions[:, 6] /= scale

    return actions

def copy_and_modify_group(source_group: h5py.Group, target_group: h5py.Group, scale: float):
    """
    Recursively copy and modify groups/datasets from the source HDF5 file.
    """
    for attr_name, attr_value in source_group.attrs.items():
        target_group.attrs[attr_name] = attr_value

    for key, item in tqdm(list(source_group.items()), desc="Processing HDF5 items"):
        if isinstance(item, h5py.Group):
            new_group = target_group.create_group(key)
            copy_and_modify_group(item, new_group, scale)
        elif isinstance(item, h5py.Dataset):
            if key == "actions":
                print(f"Modifying dataset: {key}")
                modified_data = modify_actions_dataset(item[...], scale)
                target_group.create_dataset(key, data=modified_data, dtype=item.dtype)
            else:
                target_group.create_dataset(key, data=item[...], dtype=item.dtype)

def modify_hdf5(input_path: str, output_path: str, scale: float):
    """
    Modify an HDF5 file, applying transformations to specific datasets.
    """
    with h5py.File(input_path, 'r') as f_in, h5py.File(output_path, 'w') as f_out:
        if "data" not in f_in:
            raise KeyError("The 'data' group is missing in the input file.")

        print("Copying and modifying 'data' group...")
        data_group = f_out.create_group("data")
        copy_and_modify_group(f_in["data"], data_group, scale)
        print("Modification complete.")

def main():
    """Main function to run the HDF5 dataset modification."""
    args = parse_args()
    
    input_file = os.path.join(args.dataset_folder, f"{args.task}/{args.demo_type}/{args.dimension}_abs.hdf5")
    output_file = os.path.join(args.dataset_folder, f"{args.task}/{args.demo_type}/{args.dimension}_abs_scaled.hdf5")
    
    assert os.path.exists(input_file), f"Input dataset not found: {input_file}"
    
    print(f"Modifying dataset for task: {args.task}, dimension: {args.dimension}")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")

    modify_hdf5(input_file, output_file, args.scale)
    print("Done.")


if __name__ == '__main__':
    main()