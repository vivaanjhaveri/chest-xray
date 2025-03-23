# Efficient X-Ray Representations For Classifying Diseases
## UBC Medicine Datathon 2025 (Team 8)
### Team Members:
- Ethan Rajkumar
- Joel Bonnie
- Pushya Jain
- Erhan Javed
- Vivaan Jhaveri 
- Charity Grey

<br>
Chest X-rays, a primary diagnostic tool for identifying thoracic diseases, are the focus of our project, which aims to develop an efficient X-ray representation for automating and benchmarking weakly supervised classification and localization of common thoracic diseases. By leveraging clinical metadata, bounding box annotations, and efficient vector retrieval via Pinecone, we address challenges posed by large-scale datasets (~112k X-rays), annotation constraints, and weak supervision, and build robust classification models using Support Vector Machines (SVM) and Random Forest classifiers.

---
<br>

### Conda Environments: 
#### Feature Eng
Create the conda environment: 
`conda env create -f environment.yml`

Activate the conda environment: 
`conda activate xray-analysis`
<br>

#### SVC Processing
Create the conda environment: 
`conda env create -f inference_env.yml`

Activate the conda environment: 

`conda activate xray-svc`
