#AUTHOR: Sharvesh Subhash
from transformers import T5Tokenizer, T5Model
from sklearn.cluster import DBSCAN
import torch
import numpy as np

model = T5Model.from_pretrained("t5-small")
tokenizer = T5Tokenizer.from_pretrained("t5-small",legacy=False)

def get_t5_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model.encoder(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()  

with open('filtered_phrases.txt', 'r') as file:
    phrases = [line.strip() for line in file]

print("Generating embeddings using T5-small model...")
phrase_embeddings = [get_t5_embeddings(phrase) for phrase in phrases]

phrase_embeddings = np.array(phrase_embeddings)


print("Clustering phrases...")
db = DBSCAN(metric="cosine", eps=0.3, min_samples=2)
labels = db.fit_predict(phrase_embeddings)


clusters = {}
"""
I made this into comment just to saved it for more experiments.. :Sharvesh Subhash
for label, phrase in zip(labels, phrases):
    if label not in clusters:
        clusters[label] = []
    clusters[label].append(phrase)

canonical_mappings = {
    "Amazon Web Services": ["S3", "AWS S3", "Amazon S3", "Amazon Web Services S3"],
    "Continuous Integration and Continuous Deployment": ["CI/CD", "Continuous Integration", "Continuous Deployment"],
    "Neural Networks": ["NN", "Neural Network", "Neural Networks"]
}

canonical_groups = {}
for label, group in clusters.items():
    if group:
        canonical_form = None
        

        for canonical, synonyms in canonical_mappings.items():
            if any(synonym in group for synonym in synonyms):
                canonical_form = canonical
                break

        if canonical_form is None:
            canonical_form = max(group, key=len)

        canonical_groups[canonical_form] = group

"""
for label, phrase in zip(labels, phrases):
    if label not in clusters:
        clusters[label] = []
    clusters[label].append(phrase)

canonical_groups = {}
for label, group in clusters.items():
    if group:
        canonical_form = max(group, key=len)
        canonical_groups[canonical_form] = group

with open('final_canonical_groups_t5_without_checking_legacy_false.txt', 'w') as file:
    for canonical, group in canonical_groups.items():
        file.write(f"{canonical}: {group}\n")

print("Canonical forms and their groups:")
for canonical, group in canonical_groups.items():
    print(f"{canonical}: {group}")
"""
AFTER TESTING MANY TIMES, I CAN CONCLUDE THAT THIS MODEL CAN NOT BE USED DIRECTLY
 FOR OUR USE CASE, WE NEED TO GO WITH EITHER PARAPHRASE OR MPNET MODEL!
"""