#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Example script demonstrating dataset tools utilities.

This script shows how to:
1. Delete episodes from a dataset
2. Split a dataset into train/val sets
3. Add/remove features
4. Merge datasets

Usage:
    python examples/dataset/use_dataset_tools.py
"""

import numpy as np

from lerobot.datasets.dataset_tools import (
    merge_datasets,
    delete_episodes
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main():
    print("Loading dataset...")
    dataset = LeRobotDataset("zp_robot/so101_test")
    # LeRobot使用元数据系统来管理数据集的结构信息
    print(f"Original dataset: {dataset.meta.total_episodes} episodes, {dataset.meta.total_frames} frames ,repo_id:{dataset.meta.repo_id}")
    print(f"Features: {list(dataset.meta.features.keys())}")
    
    print("\n4. Merging datasets...")
    dataset1 = LeRobotDataset("zp_robot/so101_test_filtered")
    print(f"Original dataset: {dataset1.meta.total_episodes} episodes, {dataset1.meta.total_frames} frames")
    # dataset2 = LeRobotDataset("zp_robot/so101_test1119")
    # print(f"Original dataset: {dataset2.meta.total_episodes} episodes, {dataset2.meta.total_frames} frames")
    
    print("\n1. Merging train and val splits back together...")
    merged = merge_datasets([dataset,dataset1], output_repo_id="zp_robot/so101_test_merged")
    print(f"Merged dataset: {merged.meta.total_episodes} episodes")

    # print("\n Deleting episodes ...")
    # filtered_dataset = delete_episodes(dataset, episode_indices=[0,1,2,3,4,5,6,7,8,18,12,29,34,38,42,44,47,50], repo_id="zp_robot/so101_test_filtered")
    # print(f"Filtered dataset: {filtered_dataset.meta.total_episodes} episodes")
    # print("\nDone! Check ~/.cache/huggingface/lerobot/ for the created datasets.")


if __name__ == "__main__":
    main()
