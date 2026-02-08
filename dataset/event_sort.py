import json
import numpy as np
from sentence_transformers import SentenceTransformer
from collections import defaultdict
import os


VALUES_JSON = "values.json"
EVENTS_JSON = "event.json"
OUTPUT_JSON = "top3_events_per_value.json"


model = SentenceTransformer('bge-large-zh-v1.5')

with open(VALUES_JSON, "r", encoding="utf-8") as f:
    values_dict = json.load(f)

value_names = list(values_dict.keys())
value_descs = list(values_dict.values())

print(f"load {len(value_names)} value")


value_embeddings = model.encode(
    value_descs,
    normalize_embeddings=True,
    show_progress_bar=True
)


with open(EVENTS_JSON, "r", encoding="utf-8") as f:
    events = json.load(f)

print(f"load {len(events)} events")


event_ids = [e["event_id"] for e in events]
event_descs = [e["event"] for e in events]


event_embeddings = model.encode(
    event_descs,
    normalize_embeddings=True,
    show_progress_bar=True
)


similarity_matrix = event_embeddings @ value_embeddings.T

top3_per_value = {}

for j, val_name in enumerate(value_names):
    sims = similarity_matrix[:, j]
    top3_indices = np.argsort(sims)[-3:][::-1]

    top3_events = []
    for idx in top3_indices:
        top3_events.append({
            "event_id": event_ids[idx],
            "event": event_descs[idx],
            "similarity": float(sims[idx])
        })

    top3_per_value[val_name] = top3_events

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(top3_per_value, f, ensure_ascii=False, indent=2)
