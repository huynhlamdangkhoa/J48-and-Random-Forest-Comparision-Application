# File: download_dataset.py (ở desktop hoặc Downloads folder)
import kagglehub

# Download dataset
path = kagglehub.dataset_download("oktayrdeki/heart-disease")
print("Path to dataset files:", path)
print("\nDataset downloaded! Now copy the CSV file to your Java project.")