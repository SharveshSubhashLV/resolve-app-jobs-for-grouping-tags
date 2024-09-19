#AUTHOR: Sharvesh Subhash
from txtai.embeddings import Embeddings
from sklearn.cluster import DBSCAN


model_name ="sentence-transformers/paraphrase-MiniLM-L6-v2"

#"sentence-transformers/all-mpnet-base-v2" 

embeddings = Embeddings({"path": model_name})

with open('filtered_phrases.txt', 'r') as file:
    phrases = [line.strip() for line in file]


embeddings.index([(i, phrase, None) for i, phrase in enumerate(phrases)])

print(f"Generating embeddings using model {model_name}...")
phrase_embeddings = [embeddings.transform(phrase) for phrase in phrases]

print("Clustering phrases...")
db = DBSCAN(metric="cosine", eps=0.3, min_samples=2)
labels = db.fit_predict(phrase_embeddings)

clusters = {}
for label, phrase in zip(labels, phrases):
    if label not in clusters:
        clusters[label] = []
    clusters[label].append(phrase)
"""
canonical_mappings = {
    "Amazon Web Services S3": ["S3", "AWS S3", "Amazon S3", "amazon web services s3"],
    "Continuous Integration and Continuous Deployment": ["CI/CD", "Continuous Integration", "Continuous Deployment"],
    "Neural Networks": ["NN", "Neural Network"]
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
        
with open(f"final_canonical_groups_model_M62_without_checking.txt", 'w') as file:
    for canonical, group in canonical_groups.items():
        file.write(f"{canonical}: {group}\n")

print("Canonical forms and their groups:")
for canonical, group in canonical_groups.items():
    print(f"{canonical}: {group}")
