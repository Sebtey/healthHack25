from torchvision.transforms import transforms
import torchvision.io as tvio
import torch
import io
import torch.nn.functional as F

class IModel:
    def __init__(self):
        pass

    def __call__(self, video) -> tuple[float, float, float, float, float]:
        raise NotImplementedError


class FineTunedVideoMAE(IModel):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.resizer = transforms.Compose([
            transforms.Resize((224, 224))  # Ensure resolution is correct
        ])

    def __transform(self, video):

        video_path = io.BytesIO(video)

        video, _, _ = tvio.read_video(video_path, pts_unit='sec', start_pts=0, end_pts=20)
        video = video.to(torch.float32) / 255.0  # Normalize

        # Sample frames
        total_frames = video.shape[0]
        if total_frames == 0:
            raise ValueError(f"{video_path} has 0 frames")
        if video.shape[-1] != 3:
            raise ValueError(F"{video_path} has {video.shape[-1]} channels instead of 3")

        indices = torch.linspace(0, total_frames - 1, 16).long()
        frames = video[indices]  # Shape: (num_frames, H, W, C)

        # Convert to (C, H, W) format and resize
        resize_transform = transforms.Compose([
            transforms.Resize((224, 224))  # Ensure resolution is correct
        ])

        frames = [resize_transform(frame.permute(2, 0, 1)) for frame in frames]  # (H, W, C) → (C, H, W)

        # Stack frames into a tensor (C, num_frames, H, W)
        video_tensor = torch.stack(frames).permute(0, 1, 2, 3)  # Ensure correct ordering
        video_tensor.unsqueeze(0)
        return video_tensor

    def __call__(self, video):
        tensors = self.__transform(video)
        return F.softmax(self.model(tensors.unsqueeze(0)).logits).detach().cpu().numpy().tolist()[0]



def get_model() -> IModel:
    torch_model = torch.load("machine_learning/pretrained_model.pkl", weights_only=False)
    return FineTunedVideoMAE(torch_model)

if __name__ == "__main__":
    get_model()