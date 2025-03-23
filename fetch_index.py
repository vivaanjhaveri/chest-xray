"""
Authors: Vivaan Jhaveri, Erfan Javed, Charity G. 
Editors: Joel Bonnie, Ethan Rajkumar

The purpose of this script is to fetch the vectors for the image indexes from PineCone. 

"""


import os
# from pinecone import Pinecone, ServerlessSpec, PodSpec
import time
import pandas as pd
from PIL import Image
import numpy as np
import json
import itertools

from pinecone.grpc import PineconeGRPC as Pinecone

# 1 Initialize a Pinecone client with your API key
pc = Pinecone(api_key="pcsk_31421K_HbRUu2Sy5uMCgKvqcEgoe9UnLvnk3dacgBPEbBfacggA51qrvUg4LqPaiphXzJJ")

# To get the unique host for an index, 
# see https://docs.pinecone.io/guides/data/target-an-index
index = pc.Index(host="https://nih-xray-2025-th0zwic.svc.aped-4627-b74a.pinecone.io")

# Load the CSV that contains the image indexes
data_df = pd.read_csv('data/Data_Entry_2017.csv')

# Assuming the image indexes are in a column named "Image Index"
# Get unique image indexes to avoid duplicates
image_indexes = data_df['Image Index'].unique().tolist()
p

batch_size = 1000
vectors_dict = {}

# Loop through image_indexes in batches of 1000
for i in range(0, len(image_indexes), batch_size):
    print(f'Current Index: {i}')
    
    batch_ids = image_indexes[i: i + batch_size]
    # Fetch the vectors for the current batch
    response = index.fetch(ids=batch_ids, namespace="")
    
    # Map each image index in the batch to its vector values (or None if missing)
    for img_id in batch_ids:
        # if img_id in response.get('vectors', {}):
        if img_id in response.vectors:
            # vectors_dict[img_id] = response['vectors'][img_id]['values']
            vectors_dict[img_id] = response.vectors[img_id].values
        else:
            vectors_dict[img_id] = None

# Optionally, convert the dictionary to a DataFrame and save it to CSV
df_vectors = pd.DataFrame(list(vectors_dict.items()), columns=["Image Index", "Vector"])
df_vectors.to_csv('vectors.csv', index=False)

print("Completed fetching vectors for {} image indexes.".format(len(vectors_dict)))

