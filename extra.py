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

#index.fetch(ids=["00000001_001.png"], namespace="nih-xray-2025")

print(index.fetch(ids=["00000001_001.png"], namespace=""))