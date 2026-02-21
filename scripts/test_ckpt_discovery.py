from pathlib import Path
import os

def get_best_checkpoint(run_name):
    # Find the best checkpoint for the given run name
    checkpoint_dir = Path("models/checkpoints")
    if not checkpoint_dir.exists():
        print(f"Directory not found: {checkpoint_dir}")
        return None
        
    # Pattern: run_name_epoch=XX_val_loss=YY.ckpt
    pattern = f"{run_name}_*.ckpt"
    print(f"Searching for pattern: {pattern} in {checkpoint_dir.absolute()}")
    checkpoints = list(checkpoint_dir.glob(pattern))
    
    if not checkpoints:
        print("No checkpoints found.")
        return None
        
    # Sort by modification time (most recent first)
    checkpoints.sort(key=os.path.getmtime, reverse=True)
    
    print(f"Found {len(checkpoints)} checkpoints:")
    for ckpt in checkpoints:
        print(f" - {ckpt.name}")
    
    return str(checkpoints[0])

print("Testing get_best_checkpoint for 'stage1_fruitnet'...")
ckpt = get_best_checkpoint("stage1_fruitnet")
print(f"Result: {ckpt}")
