import os
import torch
import torchvision.io as io
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms


class VideoDataset(Dataset):
    def __init__(self, root_dir, num_frames=16, clip_duration=2.0):
        self.root_dir = root_dir
        self.classes = sorted(os.listdir(root_dir))
        self.video_clips = []
        self.labels = []
        self.num_frames = num_frames
        self.clip_duration = clip_duration

        for label, class_name in enumerate(self.classes):
            class_path = os.path.join(root_dir, class_name)
            for video in os.listdir(class_path):
                if video.endswith(".mp4"):
                    video_path = os.path.join(class_path, video)
                    self.video_clips.append(video_path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.video_clips)

    def __getitem__(self, idx):
        video_path = self.video_clips[idx]
        label = self.labels[idx]

        # Pre-load video and store as tensor
        video, _, _ = io.read_video(video_path, pts_unit='sec', start_pts=0, end_pts=20)
        video = video.to(torch.float32) / 255.0  # Normalize

        # Sample frames
        total_frames = video.shape[0]
        if total_frames == 0:
            raise ValueError(f"{video_path} has 0 frames")
        if video.shape[-1] != 3:
            raise ValueError(F"{video_path} has {video.shape[-1]} channels instead of 3")

        indices = torch.linspace(0, total_frames - 1, self.num_frames).long()
        frames = video[indices]  # Shape: (num_frames, H, W, C)

        # Convert to (C, H, W) format and resize
        resize_transform = transforms.Compose([
            transforms.Resize((224, 224))  # Ensure resolution is correct
        ])

        frames = [resize_transform(frame.permute(2, 0, 1)) for frame in frames]  # (H, W, C) → (C, H, W)

        # Stack frames into a tensor (C, num_frames, H, W)
        video_tensor = torch.stack(frames).permute(0, 1, 2, 3)  # Ensure correct ordering

        return video_tensor, label


transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resizing frames
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Hyperparameters
batch_size = 4
num_workers = 4


def get_dataloader():
    # Load dataset
    dataset = VideoDataset(root_dir="dataset")
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0, pin_memory=True)
    print("Loaded dataset with length: ", len(dataset))
    return dataloader


