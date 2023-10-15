import os
import pickle
import numpy as np
from tqdm import tqdm
from nltk.tokenize import word_tokenize 


id2disease = [
    "adhd",
    "anxiety",
    "bipolar",
    "depression",
    "eating",
    "ocd",
    "ptsd"
]

topK = 16
pool_types = ["sum", "max"]
for pool_type in pool_types:
    group = f"symptom_{pool_type}"
    os.makedirs(f"processed/{group}_top{topK}", exist_ok=True)

for split in ["train", "val", "test"]:
    with open(f"../smhd/{split}_data.pkl", "rb") as f:
        data = pickle.load(f)
    with open(f"../smhd/symp_dataset/{split}_rm_feats.pkl", "rb") as f:
        symp_data = pickle.load(f)
    selected = {pool_type: [] for pool_type in pool_types}
    for i, record in enumerate(tqdm(data)):
        if len(record['diseases']):
            uid = 'P' + str(record['id'])
        else:
            uid = 'C' + str(record['id'])
        symp_probs = symp_data[uid]
        user_posts = record['posts']
        for pool_type in pool_types:
            if pool_type == "sum":
                post_scores = symp_probs.sum(1)
            elif pool_type == "max":
                post_scores = symp_probs.max(1)
            top_ids = post_scores.argsort()[-topK:]
            top_ids = np.sort(top_ids)  # sort in time order
            sel_posts = [user_posts[ii] for ii in top_ids]
            selected[pool_type].append({'id': record['id'], 'diseases': record['diseases'], 
            'selected_posts': sel_posts, 'symp_probs': symp_probs})

    for pool_type in pool_types:
        group = f"symptom_{pool_type}"
        with open(f"processed/{group}_top{topK}/{split}.pkl", "wb") as f:
            pickle.dump(selected[pool_type], f)