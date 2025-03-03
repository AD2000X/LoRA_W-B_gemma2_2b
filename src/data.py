"""
Data processing functions
"""

import json
import os
import requests
from tqdm import tqdm

def download_file(url, save_path):
    """Download file and show progress bar"""
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'wb') as f:
        with tqdm(
            total=total_size, 
            unit='B', 
            unit_scale=True, 
            desc=f"Downloading {os.path.basename(save_path)}"
        ) as pbar:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    return save_path

def load_dolly_dataset(file_path=None, max_samples=None):
    """Load Databricks Dolly dataset"""
    
    if file_path is None or not os.path.exists(file_path):
        print("Dolly dataset not found, downloading...")
        url = "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl"
        file_path = "data/databricks-dolly-15k.jsonl"
        download_file(url, file_path)
    
    data = []
    
    print(f"Loading Dolly dataset from {file_path}...")
    with open(file_path) as file:
        for line in file:
            features = json.loads(line)
            # Filter out examples with context, to keep it simple
            if features["context"]:
                continue
                
            # Format the entire example as a single string
            template = "Instruction:\n{instruction}\n\nResponse:\n{response}"
            data.append(template.format(**features))
            
            if max_samples and len(data) >= max_samples:
                break
    
    print(f"Loaded {len(data)} examples")
    return data

if __name__ == "__main__":
    # Usage demonstration
    data = load_dolly_dataset(max_samples=5)
    print("\nSample data:")
    print("-" * 50)
    print(data[0])
    print("-" * 50)