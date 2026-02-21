import os
import shutil
import sys

def safe_remove(path):
    print(f"Attempting to delete: {path}", flush=True)
    try:
        if os.path.isfile(path) or os.path.islink(path):
            os.remove(path)
            print(f"Deleted file: {path}", flush=True)
        elif os.path.isdir(path):
            shutil.rmtree(path)
            print(f"Deleted directory: {path}", flush=True)
        else:
            print(f"Path not found: {path}", flush=True)
    except Exception as e:
        print(f"Error deleting {path}: {e}", flush=True)

def safe_move(src, dst):
    print(f"Attempting to move: {src} -> {dst}", flush=True)
    try:
        if os.path.exists(src):
            shutil.move(src, dst)
            print(f"Moved {src} to {dst}", flush=True)
        else:
            print(f"Source not found: {src}", flush=True)
    except Exception as e:
        print(f"Error moving {src} to {dst}: {e}", flush=True)

if __name__ == "__main__":
    print(f"CWD: {os.getcwd()}", flush=True)
    
    # 1. Archive Datasets
    if not os.path.exists("archive_datasets"):
        os.makedirs("archive_datasets")
        
    safe_move("FruitNet_Indian", "archive_datasets/FruitNet_Indian")
    safe_move("Fruit_Quality_Classification", "archive_datasets/Fruit_Quality_Classification")
    safe_move("Fruits_360", "archive_datasets/Fruits_360")
    safe_move("temp_downloads", "archive_datasets/temp_downloads")

    # 2. Move Scripts
    if not os.path.exists("scripts"):
        os.makedirs("scripts")
        
    safe_move("test_run.py", "scripts/test_run.py")
    safe_move("test_weather.py", "scripts/test_weather.py")
    safe_move("test_write.py", "scripts/test_write.py")
    safe_move("check_distribution.py", "scripts/check_distribution.py")
    safe_move("check_fruits.py", "scripts/check_fruits.py")
    safe_move("test_ckpt_discovery.py", "scripts/test_ckpt_discovery.py")

    # 3. Delete Files
    safe_remove("debug_start.txt")
    safe_remove("install_log.txt")
    safe_remove("run_auto_train.bat")
    safe_remove("run_debug.bat")
    safe_remove("run_install.bat")
    safe_remove("run_stage4.bat")
    safe_remove("cleanup.bat") 
    
    print("Cleanup script finished.", flush=True)
