import os
import zipfile
import subprocess
import shutil
import json
import sys

# Constants
KAGGLE_KEY_CANDIDATE = "6168a881c93d5412fa61c19e033a481d" # Extracted from user input
DATASETS = [
    "raghavrpotdar/fresh-and-stale-images-of-fruits-and-vegetables",
    "kritikseth/fruit-and-vegetable-image-recognition"
]
TARGET_FRUITS = ["Bitter Gourd", "Capsicum", "Strawberry", "Cucumber"]
BASE_DATA_DIR = os.path.join("data", "New_Fruits")

def setup_kaggle_credentials():
    # Set Kaggle Config Dir to project folder (avoid C: drive)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    kaggle_dir = os.path.join(project_root, ".kaggle")
    os.environ['KAGGLE_CONFIG_DIR'] = kaggle_dir
    
    kaggle_json_path = os.path.join(kaggle_dir, "kaggle.json")
    
    if os.path.exists(kaggle_json_path):
        print(f"✅ Found Kaggle credentials at {kaggle_json_path}")
        return True
    
    print("⚠️ Kaggle credentials not found locally.")
    print(f"I have the API Key: {KAGGLE_KEY_CANDIDATE}")
    username = input("Please enter your Kaggle Username (from your account settings): ").strip()
    
    if not username:
        print("❌ Username is required to download datasets.")
        return False
        
    os.makedirs(kaggle_dir, exist_ok=True)
    with open(kaggle_json_path, 'w') as f:
        json.dump({"username": username, "key": KAGGLE_KEY_CANDIDATE}, f)
    
    print(f"✅ Created {kaggle_json_path}")
    return True

def download_and_extract():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    download_dir = os.path.join(project_root, "data", "downloads")
    
    for dataset in DATASETS:
        print(f"\n⬇️ Downloading {dataset} to {download_dir}...")
        try:
            # Use kaggle cli
            subprocess.run(["kaggle", "datasets", "download", "-d", dataset, "-p", download_dir], check=True)
            
            # Unzip
            zip_name = dataset.split("/")[-1] + ".zip"
            zip_path = os.path.join(download_dir, zip_name)
            extract_path = os.path.join(download_dir, dataset.split("/")[-1])
            
            print(f"📦 Extracting {zip_name}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
                
            # Search for target fruits and move them
            print(f"🔍 Searching for {TARGET_FRUITS} in {extract_path}...")
            for root, dirs, files in os.walk(extract_path):
                for d in dirs:
                    # Normalize name
                    dir_name_clean = d.lower().replace("_", " ").replace("-", " ")
                    
                    for target in TARGET_FRUITS:
                        if target.lower() in dir_name_clean:
                            src_path = os.path.join(root, d)
                            dst_path = os.path.join(BASE_DATA_DIR, target.replace(" ", "_")) # e.g. Bitter_Gourd
                            
                            print(f"   found {d} -> moving to {dst_path}")
                            if os.path.exists(dst_path):
                                print(f"   Merging into existing {dst_path}")
                                # Recursive merge needed or just copytree with ignore
                                _merge_dirs(src_path, dst_path)
                            else:
                                shutil.copytree(src_path, dst_path)
                                
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to download {dataset}: {e}")
        except Exception as e:
            print(f"❌ Error processing {dataset}: {e}")

def _merge_dirs(src, dst):
    # Simple merge: Copy files from src to dst, creating subdirs if needed
    for root, dirs, files in os.walk(src):
        rel_path = os.path.relpath(root, src)
        dest_root = os.path.join(dst, rel_path)
        os.makedirs(dest_root, exist_ok=True)
        for f in files:
            try:
                shutil.copy2(os.path.join(root, f), os.path.join(dest_root, f))
            except:
                pass

def main():
    print("🚀 Starting Auto-Expansion Pipeline...")
    
    # 1. Setup Kaggle
    if not setup_kaggle_credentials():
        return

    # 2. Download Data
    os.makedirs("temp_downloads", exist_ok=True)
    download_and_extract()
    
    # 3. Generate Metadata
    print("\n📝 Generating Metadata...")
    subprocess.run([sys.executable, "scripts/generate_metadata.py", "--data_dir", BASE_DATA_DIR, "--output", "data/metadata_new.json"])
    
    # 4. Train
    print("\n🚂 Starting Training...")
    # Using existing sequential training script, but pointing to the new metadata
    # We assume 'stage4_new_fruits' as the stage name
    subprocess.run([sys.executable, "scripts/train_sequential.py", "--stage_name", "stage4_new_fruits", "--metadata_file", "data/metadata_new.json", "--epochs", "5"])
    
    print("\n✅ Pipeline Completed! Restart the backend to see changes.")

if __name__ == "__main__":
    main()
