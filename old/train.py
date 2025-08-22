import pickle
import torch
from utils import extract_feature
from transformers import CLIPModel, CLIPProcessor

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

features = extract_feature(model, processor, device)

with open("features.pkl", "wb") as f:
    pickle.dump(features, f)
