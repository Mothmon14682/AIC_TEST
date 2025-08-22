import os
import numpy as np
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
import json
import torch
from sklearn.metrics.pairwise import cosine_similarity

# Load the CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load video features
def load_video_features(dir):
    print("Starting to load video features...")
    feature_file_names = os.listdir(dir)
    videos_feature = {}
    for file_name in tqdm(feature_file_names):
        video_name = os.path.splitext(file_name)[0]
        try:
            video_features = np.load(os.path.join(dir, file_name))
            videos_feature[video_name] = video_features
        except Exception as e:
            print(f"Error loading {file_name}: {e}")
    return videos_feature

# Load metadata information (could be video metadata)
def load_video_metadata(video_dir):
    print("Starting to load video metadata...")
    metadata = {}
    for file_name in tqdm(os.listdir(video_dir)):
        if file_name.endswith('.json'):
            try:
                with open(os.path.join(video_dir, file_name)) as f:
                    metadata[file_name.replace('.json', '')] = json.load(f)
            except Exception as e:
                print(f"Error loading {file_name}: {e}")
    return metadata

# Load object detection results (objects detected in the keyframes)
def load_object_detection_results(keyframe_dir):
    print("Starting to load object detection results...")
    object_detection = {}
    for video_name in tqdm(os.listdir(keyframe_dir)):
        video_dir = os.path.join(keyframe_dir, video_name)
        if os.path.isdir(video_dir):
            object_detection[video_name] = {}
            for image_file in os.listdir(video_dir):
                if image_file.endswith('.json'):
                    try:
                        with open(os.path.join(video_dir, image_file)) as f:
                            object_detection[video_name][image_file.replace('.json', '')] = json.load(f)
                    except Exception as e:
                        print(f"Error loading {image_file}: {e}")
    return object_detection

# Assuming keyframe objects are stored in a directory
object_detection_results = load_object_detection_results("/mnt/wrar/data/objects")

# Query processing function with object and metadata filtering
def search(query, video_features, metadata, object_detection_results, top_k=5):
    print(f"Processing search query: '{query}'...")
    # Convert the query to CLIP embedding
    inputs = processor(text=[query], return_tensors="pt", padding=True)
    with torch.no_grad():
        query_features = model.get_text_features(**inputs)

    # Prepare list to store results with their scores and metadata
    all_similarities = {}

    for video_name, feature_vectors in video_features.items():
        print(f"Evaluating video: {video_name}")
        # Calculate cosine similarity between the query and the video keyframes
        similarities = cosine_similarity(query_features.cpu().numpy(), feature_vectors)
        avg_similarity = np.mean(similarities)  # Aggregate similarity for all keyframes
        print(f"Average Similarity for {video_name}: {avg_similarity}")

        # Object-based filtering
        matched_objects = []
        if video_name in object_detection_results:
            for frame_id, detected_objects in object_detection_results[video_name].items():
                # Check if any object in the keyframe matches the query
                for obj in detected_objects.get('objects', []):
                    if obj.lower() in query.lower():
                        matched_objects.append(frame_id)

        # Metadata-based filtering
        matched_metadata = False
        if 'title' in metadata.get(video_name, {}) and query.lower() in metadata[video_name].get('title', '').lower():
            matched_metadata = True
        elif 'description' in metadata.get(video_name, {}) and query.lower() in metadata[video_name].get('description', '').lower():
            matched_metadata = True

        # If no metadata matches, we continue with the similarity score but also prioritize metadata
        if not matched_metadata:
            avg_similarity *= 0.8  # Give slightly lower weight if metadata doesn't match
            print(f"Adjusted similarity score for {video_name} due to lack of metadata match: {avg_similarity}")

        # Store similarity scores along with metadata (if matched)
        all_similarities[video_name] = {
            'score': avg_similarity,
            'metadata': metadata.get(video_name, {}),
            'matched_objects': matched_objects if matched_objects else None
        }

    # Sort the results by the highest similarity score
    sorted_videos = sorted(all_similarities.items(), key=lambda x: x[1]['score'], reverse=True)

    # Fetch top_k results
    results = []
    for video_name, data in sorted_videos[:top_k]:
        results.append({
            'video_name': video_name,
            'score': data['score'],
            'metadata': data['metadata'],
            'matched_objects': data['matched_objects']
        })

    return results

video_features = load_video_features("/mnt/wrar/data/clip-features-32")
metadata = load_video_metadata("/mnt/wrar/data/media-info")
while True:
    query = input("Enter a prompt: ") 
    top_results = search(query, video_features, metadata, object_detection_results)

    if top_results:
        for idx, result in enumerate(top_results):
            print(f"Result #{idx + 1}:")
            print(f"Video: {result['video_name']}")
            print(f"Similarity Score: {result['score']}")
            print(f"Metadata: {result['metadata']}")
    else:
        print("No results found.")

