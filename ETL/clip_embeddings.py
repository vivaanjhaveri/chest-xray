# model imports
import torch
from openai import OpenAI
import torch.nn as nn
from torch.utils.data import DataLoader
import clip
key = 'clip key TODO'
client = OpenAI(api_key=key)

# db imports
from pinecone import Pinecone, ServerlessSpec, PodSpec

# helper imports
from tqdm import tqdm
import time
import json
import os
import numpy as np
import pickle
import itertools
import pandas as pd
from typing import List, Union, Tuple

# visualisation imports
from PIL import Image
import base64

     
# =======================
device = "cpu"
model, preprocess = clip.load("ViT-B/32",device=device)

def get_all_image_embeddings_from_folder(folder_dir):
    image_paths = []

    for image_p in os.listdir(folder_dir):
      if (image_p.endswith(".png")):
        image_paths.append(image_p)
    print('length: ', len(image_paths))

    image_features = []
    images = [preprocess(Image.open(folder_dir + r"/" + image_p)) for image_p in image_paths]
    image_input = torch.tensor(np.stack(images))
    with torch.no_grad():
        image_features = model.encode_image(image_input).float()
    return image_paths, image_features.numpy().astype(float)


start_time = time.time()
folder_dir = r'.cache\kagglehub\datasets\nih-chest-xrays\data\versions\3\images_002\images'
image_paths, img_embeddings = get_all_image_embeddings_from_folder(folder_dir)
print("--- tensor clip %s seconds ---" % (time.time() - start_time))

# # PINECONE
pc = Pinecone(api_key="pinecone key TODO")
spec = ServerlessSpec(
    cloud="aws",
    region="us-east-1"
  )
# index_name = 'nih-xray-2025'

index_name = 'test-clip' #USING TEST CLIP
# if index_name in pc.list_indexes().names():
#     pc.delete_index(index_name)
# dimension = 512 # limit 4194304 bytes
# pc.create_index(
#     name = index_name,
#     dimension = dimension,
#     metric = 'cosine',
#     spec = spec)

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

objs = [(image_paths[i], img_embeddings[i]) for i in range(len(image_paths))]
index.upsert(vectors=objs) 

# # Upsert data with 200 vectors per upsert request
# for ids_vectors_chunk in chunks(image_vectors, batch_size=200):
#     index.upsert(vectors=ids_vectors_chunk, batch_size=1) 
print("--- pinecone %s seconds ---" % (time.time() - start_time))