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
    print(f"Loaded {len(videos_feature)} video features.")
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
    print(f"Loaded {len(metadata)} metadata files.")
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
    print(f"Loaded object detection results for {len(object_detection)} videos.")
    return object_detection

object_detection_results = load_object_detection_results("/mnt/wrar/data/objects")

# Query processing function with dynamic object and metadata filtering
def search(query, video_features, metadata, object_detection_results, top_k=5):
    print(f"Processing search query: '{query}'...")

    # Truncate query if it exceeds the token limit (77 tokens for CLIP model)
    inputs = processor(text=[query], return_tensors="pt", padding=True, truncation=True, max_length=77)
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

        # Object-based filtering without manual synonyms (using CLIP for semantic understanding)
        matched_objects = []
        if video_name in object_detection_results:
            for frame_id, detected_objects in object_detection_results[video_name].items():
                for obj in detected_objects.get('objects', []):
                    # Use CLIP model to get embeddings of both the object and query
                    object_inputs = processor(text=[obj], return_tensors="pt", padding=True, truncation=True, max_length=77)
                    with torch.no_grad():
                        object_features = model.get_text_features(**object_inputs)
                    
                    # Compare cosine similarity between the object and the query
                    object_similarity = cosine_similarity(query_features.cpu().numpy(), object_features.cpu().numpy())
                    if object_similarity >= 0.3:  # Adjust threshold as necessary
                        matched_objects.append(frame_id)

        # Metadata-based filtering with semantic similarity (using CLIP model)
        matched_metadata = False
        metadata_score = 0
        if 'title' in metadata.get(video_name, {}):
            title = metadata[video_name].get('title', '')
            # Compare title with query using CLIP embeddings
            title_inputs = processor(text=[title], return_tensors="pt", padding=True, truncation=True, max_length=77)
            with torch.no_grad():
                title_features = model.get_text_features(**title_inputs)
            title_similarity = cosine_similarity(query_features.cpu().numpy(), title_features.cpu().numpy())
            if title_similarity >= 0.4:  # Adjust threshold as necessary
                matched_metadata = True
                metadata_score = title_similarity

        if 'description' in metadata.get(video_name, {}):
            description = metadata[video_name].get('description', '')
            # Compare description with query using CLIP embeddings
            description_inputs = processor(text=[description], return_tensors="pt", padding=True, truncation=True, max_length=77)
            with torch.no_grad():
                description_features = model.get_text_features(**description_inputs)
            description_similarity = cosine_similarity(query_features.cpu().numpy(), description_features.cpu().numpy())
            if description_similarity >= 0.3:  # Adjust threshold as necessary
                matched_metadata = True
                metadata_score = max(metadata_score, description_similarity)

        # Adjust similarity score based on metadata match
        if matched_metadata:
            avg_similarity += metadata_score  # Increase similarity if metadata matches
        else:
            avg_similarity *= 0.8  # Lower similarity if no metadata match

        # Prioritize videos with object matches by adjusting similarity score
        if matched_objects:
            avg_similarity *= 1.2  # Increase similarity for object matches
            print(f"Adjusted similarity score for {video_name} due to object matches: {avg_similarity}")

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

# Load video features and metadata
video_features = load_video_features("/mnt/wrar/data/clip-features-32")
metadata = load_video_metadata("/mnt/wrar/data/media-info")

# Query loop
while True:
    query = input("Enter a prompt: ") 
    top_results = search(query, video_features, metadata, object_detection_results)

    if top_results:
        for idx, result in enumerate(top_results):
            print(f"Result #{idx + 1}:")
            print(f"Video: {result['video_name']}")
            print(f"Similarity Score: {result['score']}")
            print(f"Metadata: {result['metadata']}")
            if result['matched_objects']:
                print(f"Matched Objects in Keyframes: {', '.join(result['matched_objects'])}")
    else:
        print("No results found.")
