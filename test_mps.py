"""
Authors: Joel Bonnie 
The purpose of this script is to test the MPS functionality of PyTorch with Apple silicon. 
""" 


import os
from pinecone import Pinecone, ServerlessSpec, PodSpec
import time
import pandas as pd
from PIL import Image
import numpy as np
import json
import itertools
import torch
from process_image import process_image

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(device)