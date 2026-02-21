
import os
import zipfile
import subprocess

def download_strawberry():
    dataset_slug = "noulam/strawberry-dataset" # Common strawberry dataset
    target_dir = "data/New_Fruits/Strawberry"
    
    print(f"Attempting to download {dataset_slug} to {target_dir}...")
    
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        
    try:
        # Using kaggle CLI
        cmd = f"kaggle datasets download -d {dataset_slug} -p {target_dir}"
        subprocess.check_call(cmd, shell=True)
        
        # Extract
        for file in os.listdir(target_dir):
            if file.endswith(".zip"):
                zip_path = os.path.join(target_dir, file)
                print(f"Extracting {file}...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(target_dir)
                os.remove(zip_path)
                print("Extraction complete.")
                
    except subprocess.CalledProcessError as e:
        print(f"Error downloading dataset: {e}")
        print("Please ensure you have the Kaggle API configured.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    download_strawberry()
