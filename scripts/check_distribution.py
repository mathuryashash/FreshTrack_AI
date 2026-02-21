import json
from collections import Counter

def check_distribution(filename):
    print(f"Checking {filename}...")
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        
        freshness_counts = Counter(item['freshness'] for item in data)
        print("Freshness Distribution:", dict(freshness_counts))
        
        quality_counts = Counter(item.get('quality', 'N/A') for item in data)
        print("Quality Distribution:", dict(quality_counts))
        
    except FileNotFoundError:
        print(f"File not found: {filename}")

check_distribution("data/metadata_fruitnet.json")
check_distribution("data/metadata_fruits360.json")
