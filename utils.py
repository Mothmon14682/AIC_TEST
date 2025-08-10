import os
import re
import cv2
import faiss
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from numpy.linalg import norm


def extract_feature(model, processor, device):
    frame_dir = "frames"
    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith(".jpg")])
    batch_size = 16
    features = {}
    batched_images, batched_filenames = [], []

    j = 0
    for i, frame_file in enumerate(frame_files):
        image_path = os.path.join(frame_dir, frame_file)
        image = Image.open(image_path).convert("RGB")
        batched_images.append(image)
        batched_filenames.append(frame_file)

        if len(batched_images) == batch_size or i == len(frame_files) - 1:
            j += 1
            print(f"Batch [{j}]: {frame_file}")
            inputs = processor(images=batched_images, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                image_features = model.get_image_features(pixel_values=inputs["pixel_values"])
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

            for name, feature in zip(batched_filenames, image_features):
                features[name] = feature.cpu().numpy()

            batched_images.clear()
            batched_filenames.clear()

    print(f"Extracted features for {len(features)} frames.")
    return features


def extract_text_vector(text, model, processor, device):
    text_inputs = processor(text=[text], return_tensors="pt").to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    return text_features.cpu().numpy()

def extract_image_vector(image_path, model, processor, device):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        image_features = model.get_image_features(pixel_values=inputs["pixel_values"])
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
    
    return image_features.cpu().numpy()

def cosine_similarity(a, b):
    return np.dot(a, b.T) / (norm(a) * norm(b))


def create_faiss_index(features):
    features_matrix = np.array(list(features.values())).astype("float32")
    faiss.normalize_L2(features_matrix)
    index = faiss.IndexFlatIP(features_matrix.shape[1])
    index.add(features_matrix)
    return index


def similarity_text_with_video(features, text_vector, index, k=10):
    text_vector = text_vector / np.linalg.norm(text_vector)
    query = text_vector.astype("float32")

    D, I = index.search(query, k)

    similarities = []
    feature_keys = list(features.keys())
    for i, idx in enumerate(I[0]):
        frame_name = feature_keys[idx]
        score = D[0][i]
        similarities.append((frame_name, score))
    return similarities

def show_unranked_frames(unranked, frame_dir="frames", top_n=5):
    n = min(len(unranked), top_n)
    plt.figure(figsize=(15, 4))
    
    for i in range(n):
        frame_name, score = unranked[i]
        frame_path = os.path.join(frame_dir, frame_name)
        
        image = Image.open(frame_path)
        
        plt.subplot(1, n, i + 1)
        plt.imshow(image)
        plt.title(f"{frame_name}\nScore: {score:.3f}")
        plt.axis("off")
    
    plt.tight_layout()
    plt.savefig("bull_results.png")
    plt.close()
