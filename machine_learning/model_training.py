from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification, TrainingArguments, Trainer
import numpy as np
import torch

video = list(np.random.randn(16, 3, 224, 224))

processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics", num_labels=5)

training_args = TrainingArguments(output_dir="models")

inputs = processor(video, return_tensors="pt")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
)

with torch.no_grad():
  outputs = model(**inputs)
  logits = outputs.logits

predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])
