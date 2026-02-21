
import os
from pathlib import Path
import sys

print("Debug script started")
print(f"CWD: {os.getcwd()}")
print(f"Python executable: {sys.executable}")

data_dir = Path("data")
if data_dir.exists():
    print(f"Data dir exists: {data_dir.absolute()}")
    try:
        test_file = data_dir / "debug_output.txt"
        with open(test_file, "w") as f:
            f.write("Hello from debug script\n")
        print(f"Created {test_file}")
    except Exception as e:
        print(f"Failed to create file: {e}")
else:
    print("Data dir does not exist")

print("Checking datasets:")
for d in ["FruitNet_Indian", "Fruits_360", "Fruit_Quality_Classification"]:
    p = Path(d)
    print(f"  {d}: {p.exists()} (Abs: {p.absolute()})")
