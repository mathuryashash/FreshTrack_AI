import os
import glob

def rename_checkpoint():
    checkpoint_dir = os.path.join("models", "checkpoints")
    # Find the stage 3 checkpoint
    pattern = os.path.join(checkpoint_dir, "stage3_fruits360_*.ckpt")
    files = glob.glob(pattern)
    
    if not files:
        # Check if already renamed
        if os.path.exists(os.path.join(checkpoint_dir, "stage3_base.ckpt")):
            print("Checkpoint already renamed to stage3_base.ckpt")
            return
        
        # Check for stage3_latest.ckpt
        if os.path.exists(os.path.join(checkpoint_dir, "stage3_latest.ckpt")):
             os.rename(os.path.join(checkpoint_dir, "stage3_latest.ckpt"), os.path.join(checkpoint_dir, "stage3_base.ckpt"))
             print("Renamed stage3_latest.ckpt to stage3_base.ckpt")
             return

        print("No Stage 3 checkpoint found to rename.")
        print(f"Files in {checkpoint_dir}: {os.listdir(checkpoint_dir)}")
        return

    # Take the latest one if multiple
    files.sort(key=os.path.getmtime, reverse=True)
    target_file = files[0]
    new_name = os.path.join(checkpoint_dir, "stage3_base.ckpt")
    
    try:
        os.rename(target_file, new_name)
        print(f"Successfully renamed '{target_file}' to '{new_name}'")
    except OSError as e:
        print(f"Error renaming file: {e}")

if __name__ == "__main__":
    rename_checkpoint()
