import os
from pinecone import Pinecone, ServerlessSpec, PodSpec
import time
import pandas as pd
from PIL import Image
import numpy as np
import json
import itertools
from process_image import process_image

# ===========================
# 1 Initialize a Pinecone client with your API key
pc = Pinecone(api_key="pcsk_31421K_HbRUu2Sy5uMCgKvqcEgoe9UnLvnk3dacgBPEbBfacggA51qrvUg4LqPaiphXzJJ")

spec = ServerlessSpec(
    cloud="aws",
    region="us-east-1"
  )

index_name = 'nih-xray-2025'
# index_name = 'test'

# if index_name in pc.list_indexes().names():
#     pc.delete_index(index_name)
     
# dimension = 2048 # limit 4194304 bytes
# pc.create_index(
#     name = index_name,
#     dimension = dimension,
#     metric = 'cosine',
#     spec = spec)

# ===========================
# 2 PROCESS vector images
#    https://developers.google.com/drive/v2/web/search-parameters
def open_images(folder_dir: str):
    iterable = []
    i = 0
    for image in os.listdir(folder_dir):
        i += 1
        if i % 50 == 0:
            print('image', i)
        if (image.endswith(".png")):
            vector = process_image(folder_dir + r"/" + image) #.numpy().astype(float)
            iterable.append((image, vector))
    return iterable

# Example generator that generates many (id, vector) pairs
folder_dir = r'C:\Users\Hello\.cache\kagglehub\datasets\nih-chest-xrays\data\versions\3\images_001\images'
image_vectors = open_images(folder_dir)

# ===========================
# wait for INDEX to be ready
while not pc.describe_index(index_name).status['ready']:
    time.sleep(1)
print('----Pinecone index is ready!')
index= pc.Index(index_name)


def chunks(iterable, batch_size=200):
    """A helper function to break an iterable into chunks of size batch_size."""
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))


start_time = time.time()

# index.upsert(vectors=image_vectors) 

# # Upsert data with 200 vectors per upsert request
for ids_vectors_chunk in chunks(image_vectors, batch_size=200):
    index.upsert(vectors=ids_vectors_chunk, batch_size=1) 
print("--- %s seconds ---" % (time.time() - start_time))
