import requests
import sys
import os

def test_predict(image_path):
    url = "http://127.0.0.1:8000/predict"
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    print(f"Sending {image_path} to {url}...")
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(url, files=files)
        
        if response.status_code == 200:
            print("Success!")
            print(response.json())
        else:
            print(f"Failed with status {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Use a sample image from the dataset if available
    # Or just use a dummy path and let user know
    import glob
    images = glob.glob("data/downloads/**/*.jpg", recursive=True)
    if not images:
        images = glob.glob("data/**/*.jpg", recursive=True)
        
    if images:
        test_predict(images[0])
    else:
        print("No images found to test.")
