import torch
import torch.nn as nn
import torch.optim as optim
from transformers import VideoMAEForVideoClassification

# Load Pretrained VideoMAE Model
from machine_learning.dataset_processing import get_dataloader

model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base", num_labels=5)
print("Loaded model")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("Using:", device.type)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=5e-5)

num_epochs = 10

dataloader = get_dataloader()

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for videos, labels in dataloader:
        videos = videos.to(device)  # Shape: (batch_size, C, num_frames, H, W)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(videos).logits  # Forward pass
        loss = criterion(outputs, labels)

        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss / len(dataloader)}")

torch.save(model, "pretrained_model.pkl")

if __name__ == "__main__":
    pass