import torch
import torchvision
import os
from torchvision.io import read_video
from torch.utils.data import Dataset, DataLoader


class VideoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))  # Folders are the labels
        self.class_to_idx = {cls: int(cls) for cls in self.classes}  # Convert to int

        # Collect all video file paths and labels
        self.video_paths = []
        self.labels = []

        for cls in self.classes:
            class_dir = os.path.join(root_dir, cls)
            for filename in os.listdir(class_dir):
                if filename.endswith(".mp4"):  # Ensure it's a video file
                    self.video_paths.append(os.path.join(class_dir, filename))
                    self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        # Read video as (T, H, W, C) tensor
        video, _, _ = read_video(video_path, pts_unit="sec")

        # Optional transformations
        if self.transform:
            video = self.transform(video)

        return video, label


# Usage
dataset = VideoDataset(root_dir="path_to_dataset")

import numpy as np
from torch.utils.data import WeightedRandomSampler

# Get class distribution
labels = dataset.labels  # List of all labels in dataset
class_counts = np.bincount(labels)  # Number of samples per class
class_weights = 1.0 / class_counts  # Inverse class frequency

# Create weights for each sample
sample_weights = [class_weights[label] for label in labels]

# Define sampler
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

# Create DataLoader with sampler
balanced_dataloader = DataLoader(dataset, batch_size=4, sampler=sampler, num_workers=4)

