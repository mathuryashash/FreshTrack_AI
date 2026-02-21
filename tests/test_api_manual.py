import requests
import os

# Configuration
API_URL = "http://127.0.0.1:8000/predict"
IMAGE_PATH = "data/New_Fruits/Apple/Fresh/IMG_20200821_190011.jpg"  # Replace with a valid path if needed

def test_api():
    # Find a valid image to test
    image_path = IMAGE_PATH
    if not os.path.exists(image_path):
        # Fallback to finding first jpg in data directory
        for root, dirs, files in os.walk("data"):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(root, file)
                    break
            if os.path.exists(image_path):
                break
    
    if not os.path.exists(image_path):
        print("Error: No image found to test.")
        return

    print(f"Testing API with image: {image_path}")

    try:
        with open(image_path, "rb") as f:
            files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
            response = requests.post(API_URL, files=files)
        
        if response.status_code == 200:
            print("\n✅ API Test Passed!")
            print("Response:", response.json())
        else:
            print(f"\n❌ API Test Failed with status code: {response.status_code}")
            print("Response:", response.text)

    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to the API. Make sure it's running (uvicorn src.api.main:app).")

if __name__ == "__main__":
    test_api()
