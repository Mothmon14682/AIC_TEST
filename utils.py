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

# Load CLIP
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


def extract_feature(model, processor):
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


def extract_text_vector(text, processor):
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
    index = faiss.IndexFlatL2(features_matrix.shape[1])
    index.add(features_matrix)
    return index


def similarity_text_with_video(features, text_vector, index, k=10):
    text_vector = text_vector / np.linalg.norm(text_vector)
    query = np.expand_dims(text_vector.flatten(), axis=0)
    D, I = index.search(query, k)

    similarities = []
    for i, idx in enumerate(I[0]):
        frame_name = list(features.keys())[idx]
        score = 1 - 0.5 * D[0][i]  # L2 to cosine approximation
        similarities.append((frame_name, score))
    return similarities


def extract_orb_features(img):
    orb = cv2.ORB_create(1000)
    keypoints, descriptors = orb.detectAndCompute(img, None)
    return keypoints, descriptors


def match_descriptors(desc1, desc2):
    if desc1 is None or desc2 is None:
        return []
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(desc1, desc2)
    return sorted(matches, key=lambda x: x.distance)


def nosac_filter(matches, kp1, kp2, angle_thresh=20, scale_thresh=0.5):
    inliers = []
    for m in matches:
        angle1 = kp1[m.queryIdx].angle
        angle2 = kp2[m.trainIdx].angle
        if angle1 == -1 or angle2 == -1 or abs(angle1 - angle2) > angle_thresh:
            continue
        size1 = kp1[m.queryIdx].size
        size2 = kp2[m.trainIdx].size
        if min(size1, size2) / max(size1, size2) < scale_thresh:
            continue
        inliers.append(m)
    return inliers


def extract_video_and_frame(fname):
    match = re.match(r"(.+?)_(\d+)\.jpg", fname)
    return (match.group(1), int(match.group(2))) if match else (None, -1)

def compute_combined_score(clip_score, inliers, max_clip, min_clip, max_inliers, alpha=0.6, beta=0.4):
    norm_clip = (clip_score - min_clip) / (max_clip - min_clip + 1e-6)
    norm_inlier = inliers / (max_inliers + 1e-6)
    return alpha * norm_clip + beta * norm_inlier

def rerank_with_nosac(
    top_k_results,
    frame_dir="frames",
    alpha=0.7,
    beta=0.3,
    min_gap=10,
    penalty=0.05,
    bonus=0.05,
    top_n=10,
):
    if not top_k_results:
        return []

    top_frame = top_k_results[0][0]
    query_img = cv2.imread(os.path.join(frame_dir, top_frame))
    q_kp, q_des = extract_orb_features(query_img)

    scores_raw = []

    max_inliers = 0
    for fname, clip_score in top_k_results:
        img = cv2.imread(os.path.join(frame_dir, fname))
        if img is None:
            scores_raw.append((fname, clip_score, 0))
            continue

        kp, des = extract_orb_features(img)
        matches = match_descriptors(q_des, des)
        inliers = nosac_filter(matches, q_kp, kp)
        inlier_count = len(inliers)
        max_inliers = max(max_inliers, inlier_count)
        scores_raw.append((fname, clip_score, inlier_count))

    clip_scores = [score for _, score, _ in scores_raw]
    min_clip = min(clip_scores)
    max_clip = max(clip_scores)

    reranked = []
    for fname, clip_score, inlier_count in scores_raw:
        final_score = compute_combined_score(
            clip_score, inlier_count, max_clip, min_clip, max_inliers, alpha, beta
        )
        reranked.append((fname, final_score, inlier_count))

    reranked.sort(key=lambda x: x[1], reverse=True)

    final = []
    used_frames = set()

    for i, (fname, score, inliers) in enumerate(reranked):
        video_id, frame_num = extract_video_and_frame(fname)

        if (video_id, frame_num) in used_frames:
            continue

        adjusted_score = score + (bonus if i == 0 else 0.0)

        for j in range(i + 1, len(reranked)):
            other_fname, other_score, other_inliers = reranked[j]
            other_video, other_frame = extract_video_and_frame(other_fname)

            if video_id == other_video and abs(frame_num - other_frame) < min_gap:
                reranked[j] = (
                    other_fname,
                    max(0.0, other_score - penalty),
                    other_inliers,
                )
                used_frames.add((other_video, other_frame))

        final.append((fname, adjusted_score, inliers))

    selected_names = {f[0] for f in final}
    untouched = [r for r in reranked if r[0] not in selected_names]
    full_list = sorted(final + untouched, key=lambda x: x[1], reverse=True)

    return full_list[:top_n]


def show_reranked_frames(reranked, frame_dir="frames", top_n=5):
    plt.figure(figsize=(15, 4))
    for i, (frame_name, score, _) in enumerate(reranked[:top_n]):
        frame_path = os.path.join(frame_dir, frame_name)
        img = Image.open(frame_path)
        plt.subplot(1, top_n, i + 1)
        plt.imshow(img)
        plt.title(f"{frame_name}\nScore: {score:.3f}")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig("reranked_results.png")
    print("Saved as reranked_results.png")


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
