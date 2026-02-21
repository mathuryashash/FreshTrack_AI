import os

# Use absolute path to be sure
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, "data", "New_Fruits")

dirs = [
    os.path.join(data_dir, "Strawberry", "Fresh"),
    os.path.join(data_dir, "Strawberry", "Rotten")
]

print(f"Base Dir: {base_dir}")

for d in dirs:
    try:
        os.makedirs(d, exist_ok=True)
        print(f"Created: {d}")
    except Exception as e:
        print(f"Failed to create {d}: {e}")
