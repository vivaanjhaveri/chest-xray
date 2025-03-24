import os
from kaggle.api.kaggle_api_extended import KaggleApi

path = ''
os.chdir(path)
# Setting the variabeles
os.environ['KAGGLE_USERNAME'] = ""
os.environ['KAGGLE_KEY'] = ""

# Connecting to Kaggle
api = KaggleApi()
api.authenticate()

import kagglehub

# Download the NIH Chest X-ray dataset via KaggleHub
dataset_path = kagglehub.dataset_download("nih-chest-xrays/data")

print("Dataset downloaded successfully.")
print("Path to dataset files:", dataset_path)
