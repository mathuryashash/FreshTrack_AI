import os
import requests

api_key = os.environ.get("OPENWEATHERMAP_API_KEY", "")
cities_to_test = ["Bengaluru", "Bangalore", "Bengaluru,IN", "Mumbai", "London"]

for city in cities_to_test:
    print(f"Testing: {city}...")
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        res = requests.get(url, timeout=5)
        print(f"Status Code: {res.status_code}")
        print(f"Response: {res.json()}")
    except Exception as e:
        print(f"Error: {e}")
    print("-" * 20)
