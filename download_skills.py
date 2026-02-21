import urllib.request
import zipfile
import io
import os

url = 'https://github.com/google-labs-code/stitch-skills/archive/refs/heads/main.zip'
print("Downloading...")
r = urllib.request.urlopen(url)
print("Extracting...")
with zipfile.ZipFile(io.BytesIO(r.read())) as z:
    z.extractall('.agents/skills')
print("Done.")
