import torch
from torch.nn import functional as F
from torchvision import transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import numpy.linalg as norm


efficientnet = models.efficientnet_b5(pretrained=True)  # Load ResNet-50
efficientnet = torch.nn.Sequential(*list(efficientnet.children())[:-1])  # Remove the last layer
efficientnet.eval()  # Set the model to evaluation mode

# Preprocess the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match ResNet input size
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
])

# Process Image:
def process_image(image_name):
  image = Image.open(image_name).convert("RGB")
  image_tensor = transform(image).unsqueeze(0)
  with torch.no_grad():
    features = efficientnet(image_tensor)
  features = features.flatten()
  return features


