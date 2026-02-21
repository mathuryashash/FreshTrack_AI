import os
import shutil
import time

def safe_remove(path):
    try:
        if os.path.isfile(path) or os.path.islink(path):
            os.remove(path)
            print(f"Deleted file: {path}")
        elif os.path.isdir(path):
            shutil.rmtree(path)
            print(f"Deleted directory: {path}")
    except Exception as e:
        print(f"Error deleting {path}: {e}")

def safe_move(src, dst):
    try:
        if os.path.exists(src):
            shutil.move(src, dst)
            print(f"Moved {src} to {dst}")
    except Exception as e:
        print(f"Error moving {src} to {dst}: {e}")

def main():
    root_dir = os.getcwd() # Should be project root
    print(f"Cleaning up in: {root_dir}")
    
    # 1. Archive Datasets
    archive_dir = os.path.join(root_dir, "archive_datasets")
    if not os.path.exists(archive_dir):
        os.makedirs(archive_dir)
        print(f"Created archive directory: {archive_dir}")

    datasets_to_move = [
        "FruitNet_Indian",
        "Fruit_Quality_Classification",
        "Fruits_360",
        "temp_downloads"
    ]
    
    for item in datasets_to_move:
        safe_move(os.path.join(root_dir, item), os.path.join(archive_dir, item))

    # 2. Move Scripts
    scripts_dir = os.path.join(root_dir, "scripts")
    if not os.path.exists(scripts_dir):
        os.makedirs(scripts_dir)
    
    scripts_to_move = [
        "test_run.py",
        "test_weather.py",
        "test_write.py",
        "check_distribution.py",
        "check_fruits.py",
        "test_ckpt_discovery.py"
    ]
    
    for item in scripts_to_move:
        safe_move(os.path.join(root_dir, item), os.path.join(scripts_dir, item))

    # 3. Delete Files
    files_to_delete = [
        "debug_start.txt",
        "install_log.txt",
        "run_auto_train.bat",
        "run_debug.bat",
        "run_install.bat",
        "run_stage4.bat",
        "cleanup.bat" 
    ]
    
    for item in files_to_delete:
        safe_remove(os.path.join(root_dir, item))

if __name__ == "__main__":
    main()
