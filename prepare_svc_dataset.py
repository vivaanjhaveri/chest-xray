# Import libraries: 
import pandas as pd
from PIL import Image
import numpy as np
import json

image_feature_df = pd.read_csv('vectors.csv')
print("LOG: Vectors dataframe read")


# The elements in the Vector column is currently a string. Parse this to create a list of lists 
feature_vector_list = [json.loads(curr_str_list) for curr_str_list in image_feature_df['Vector'].tolist()]
print("LOG: Converted features to a list")


print(f" The number of features: {len(feature_vector_list)}")
print(f" The dimension of each feature: {len(feature_vector_list[0])}")

# convert Vector column to an 2048 input variables 
features_expanded_df = array_df = pd.DataFrame(feature_vector_list, 
                       columns=[f'x{i}' for i in range(len(feature_vector_list[0]))])
image_feature_df_wide = pd.concat([image_feature_df.drop('Vector', axis=1), features_expanded_df], axis=1)
print("LOG: Input features ready")


# Adding new columns to the dataframe for labels 
classes = [
    "Atelectasis", "Consolidation", "Infiltration", "Pneumothorax", "Edema", 
    "Emphysema", "Fibrosis", "Effusion", "Pneumonia", "Pleural_Thickening", 
    "Cardiomegaly", "Nodule", "Mass", "Hernia"
    
]
# Initializing as 0
image_feature_df_wide[classes] = 0

data_df = pd.read_csv('data/Data_Entry_2017.csv')

print("LOG: Parsing Labels")

image_indexes = data_df['Image Index'].unique().tolist()
for curr_index, current_image in enumerate(image_indexes):
    if curr_index % 5000 == 0:
       print(f"Current Index Parsed: {curr_index}")
    classes_detected = data_df[data_df["Image Index"] == current_image]["Finding Labels"].iloc[0]
    split_classes = classes_detected.split("|")
    mask = image_feature_df_wide["Image Index"] == current_image
    for curr_class in split_classes:
       if curr_class != "No Finding":
        image_feature_df_wide.loc[mask, curr_class] = 1

print("LOG: Labels added to dataframe")

image_feature_df_wide.to_csv('xray_input.csv', index=False)
print("LOG: Dataframe saved!")

