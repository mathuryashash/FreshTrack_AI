import json
import os

def get_fruit_types(filename):
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        
        # Extract fruit names from paths
        # Example path: FruitNet_Indian\Processed Images_Fruits\Bad Quality_Fruits\Apple_Bad\IMG...
        # We can look for keywords or directory structure
        
        fruit_types = set()
        for item in data:
            path = item['image_path']
            # Normalize path separators
            path = path.replace('\\', '/')
            parts = path.split('/')
            
            # Simple heuristic: Look for known fruit names in the path parts
            # or just collect the parent folder names and clean them up
            for part in parts:
                lower_part = part.lower()
                if 'apple' in lower_part: fruit_types.add('Apple')
                elif 'banana' in lower_part: fruit_types.add('Banana')
                elif 'guava' in lower_part: fruit_types.add('Guava')
                elif 'lime' in lower_part: fruit_types.add('Lime')
                elif 'orange' in lower_part: fruit_types.add('Orange')
                elif 'pomegranate' in lower_part: fruit_types.add('Pomegranate')
                elif 'grape' in lower_part: fruit_types.add('Grape')
                elif 'mango' in lower_part: fruit_types.add('Mango')
                elif 'potato' in lower_part: fruit_types.add('Potato')
                elif 'tomato' in lower_part: fruit_types.add('Tomato')
        
        return fruit_types

    except FileNotFoundError:
        return set()

fruitnet_fruits = get_fruit_types("data/metadata_fruitnet.json")
# We focused on FruitNet for the robust checkpoint
print("Supported Fruits (FruitNet):", sorted(list(fruitnet_fruits)))
