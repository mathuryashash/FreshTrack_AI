import os
import argparse

def setup_data_directories(base_path='data'):
    """
    Sets up the directory structure for the project's data.
    """
    subdirs = [
        'raw',
        'processed',
        'external',
        'interim'
    ]
    
    for subdir in subdirs:
        path = os.path.join(base_path, subdir)
        os.makedirs(path, exist_ok=True)
        # Create a .gitkeep file to ensure the directory is tracked by git if empty
        with open(os.path.join(path, '.gitkeep'), 'w') as f:
            pass
        print(f"Created/Verified directory: {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup data directories for FreshTrack AI")
    parser.add_argument("--path", type=str, default="data", help="Base path for data directory")
    args = parser.parse_args()
    
    setup_data_directories(args.path)
