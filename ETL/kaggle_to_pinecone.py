import os
from pinecone import Pinecone, ServerlessSpec, PodSpec
import time
import pandas as pd
from PIL import Image
import numpy as np
import json
import itertools

# ===========================
# 1 Initialize a Pinecone client with your API key
pc = Pinecone(api_key="key")

spec = ServerlessSpec(
    cloud="aws",
    region="us-east-1"
  )

# index_name = 'nih-xray-2025'
index_name = 'test'

if index_name in pc.list_indexes().names():
    pc.delete_index(index_name)
     
dimension = 20000 # limit 4194304 bytes
pc.create_index(
    name = index_name,
    dimension = dimension,
    metric = 'cosine',
    spec = spec)

# ===========================
# 2 PROCESS vector images
#    https://developers.google.com/drive/v2/web/search-parameters
def open_images(folder_dir: str):
    iterable = []
    for image in os.listdir(folder_dir):
        if (image.endswith(".png")):
            pic = Image.open(folder_dir + r"/" + image).convert("L")
            vector = np.array(pic)
            print(vector)
            vector = vector.flatten().astype(np.float32)
            print(image)
            # print(vector)
            
            iterable.append((image, vector))
    return iterable

# Example generator that generates many (id, vector) pairs
folder_dir = r''
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
obj = {"id": image_vectors[0][0], "values": image_vectors[0][1]}

out_file = open("myfile.json", "w")
out_file.write(str(obj))
print(obj)
index.upsert(vectors=[obj]) 

# # Upsert data with 200 vectors per upsert request
# for ids_vectors_chunk in chunks(image_vectors, batch_size=200):
#     index.upsert(vectors=ids_vectors_chunk, batch_size=1) 
print("--- %s seconds ---" % (time.time() - start_time))
