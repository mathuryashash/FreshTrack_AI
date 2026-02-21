import os
import zipfile

with open("debug_start.txt", "w") as f:
    f.write("Started\n")

# 1. Install the kaggle library if you haven't already
# !pip install kaggle

def install_kaggle_datasets():
    # Define the datasets to download
    datasets = {
        "FruitNet_Indian": "shashwatwork/fruitnet-indian-fruits-dataset-with-quality",
        "Fruits_360": "moltean/fruits",
        "Fruit_Quality_Classification": "ryandpark/fruit-quality-classification"
    }

    # Check for kaggle.json
    kaggle_json_path = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(kaggle_json_path):
        print(f"Error: kaggle.json not found at {kaggle_json_path}")
        print("Please ensure you have placed your Kaggle API key in the correct location.")
        return

    for folder_name, dataset_slug in datasets.items():
        print(f"\n--- Installing: {folder_name} ---")
        
        # Create a specific directory for the dataset
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        
        # Download the dataset using the Kaggle API
        # This assumes kaggle.json is in the correct path (~/.kaggle/kaggle.json)
        os.system(f"kaggle datasets download -d {dataset_slug} -p {folder_name}")
        
        # Find the downloaded zip file and extract it
        for file in os.listdir(folder_name):
            if file.endswith(".zip"):
                zip_path = os.path.join(folder_name, file)
                print(f"Extracting {file}...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(folder_name)
                os.remove(zip_path) # Clean up the zip file to save space

    print("\nAll difficult datasets have been installed and extracted.")

if __name__ == "__main__":
    install_kaggle_datasets()
