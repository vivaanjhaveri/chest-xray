import os
from pinecone import Pinecone, ServerlessSpec, PodSpec
import time
import pandas as pd
from PIL import Image
import numpy as np
import json
import itertools
from process_image import process_image
import torch


# ===========================
# 1 Initialize a Pinecone client with your API key
pc = Pinecone(api_key="pcsk_31421K_HbRUu2Sy5uMCgKvqcEgoe9UnLvnk3dacgBPEbBfacggA51qrvUg4LqPaiphXzJJ")

spec = ServerlessSpec(
    cloud="aws",
    region="us-east-1"
  )

index_name = 'nih-xray-2025'
# index_name = 'test'
     
## Create a new index
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
        # if i < 200:
        #     continue
        # elif i > :
        #     break
        if i % 500 == 0:
            print('image', i)
        if (image.endswith(".png")):
            vector = process_image(folder_dir + r"/" + image) #.numpy().astype(float)
            iterable.append((image, vector))
    return iterable

# Example generator that generates many (id, vector) pairs

start_time = time.time()
folder_dir = r'/Users/ethanelliotrajkumar/Downloads/chest-xray-main/images_004/images'
image_vectors = open_images(folder_dir)
print("--- %s seconds ---" % (time.time() - start_time))


# ===========================
# wait for INDEX to be ready
while not pc.describe_index(index_name).status['ready']:
    time.sleep(1)
print('----Pinecone index is ready!')


def chunks(iterable, batch_size=200):
    """A helper function to break an iterable into chunks of size batch_size."""
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))



# index.upsert(vectors=image_vectors) 

start_time = time.time()
# # Upsert data with 200 vectors per upsert request
# for ids_vectors_chunk in chunks(image_vectors, batch_size=200):
#     index.upsert(vectors=ids_vectors_chunk) 

# Parallalize
host = r'https://nih-xray-2025-th0zwic.svc.aped-4627-b74a.pinecone.io'
with pc.Index(host=host, pool_threads=30) as index:
    # Send requests in parallel
    async_results = [
        index.upsert(vectors=ids_vectors_chunk, async_req=True)
        for ids_vectors_chunk in chunks(image_vectors, batch_size=200)
    ]
    # Wait for and retrieve responses (this raises in case of error)
    [async_result.get() for async_result in async_results]
print("--- %s seconds ---" % (time.time() - start_time))
