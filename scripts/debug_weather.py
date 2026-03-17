import os
import requests

api_key = os.environ.get("OPENWEATHERMAP_API_KEY", "")
cities = ["delhi", "Delhi", "Delhi,IN", "New Delhi"]

print(f"Testing API Key: {api_key}")

for city in cities:
    print(f"\n--- Testing City: {city} ---")
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    try:
        res = requests.get(url, timeout=10)
        print(f"Status Code: {res.status_code}")
        if res.status_code == 200:
            print("Success!")
            print(res.json())
        else:
            print("Failed!")
            print(res.text)
    except Exception as e:
        print(f"Exception: {e}")
