import joblib
import torch
from transformers import CLIPProcessor, CLIPModel
from utils import create_faiss_index, extract_text_vector, extract_image_vector, show_unranked_frames, similarity_text_with_video, rerank_with_nosac, show_reranked_frames

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

features = joblib.load("features.pkl")
text_vector = extract_text_vector("A man and a woman sitting on a bicycle who are not students", processor)
image_vector = extract_image_vector("frames/dailomt_2099.jpg", model, processor, device)
index = create_faiss_index(features)
top_sim = similarity_text_with_video(features, image_vector, index, 100) 

print("------Before reranked------")
i = 1
for frame, score in top_sim:
    if i > 5:
        break 
    print(f"{frame}: score={score}")
    i += 1

i = 1
print("---------Reranked----------")
reranked = rerank_with_nosac(top_sim)
for frame, score, inliers in reranked:
    if i > 5:
        break
    print(f"{frame}: score={score:.4f}, inliers={inliers}")
    i += 1

show_unranked_frames(top_sim)
show_reranked_frames(reranked, top_n = 5)
