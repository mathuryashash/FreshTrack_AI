import subprocess
import os
import sys
from pathlib import Path

def run_training_stage(stage_name, metadata_file, resume_from=None, epochs=10):
    print(f"\n{'='*50}")
    print(f"Starting Stage: {stage_name}")
    print(f"Metadata: {metadata_file}")
    print(f"Resume Checkpoint: {resume_from}")
    print(f"{'='*50}\n")
    
    cmd = [
        sys.executable, "src/training/train.py",
        "--metadata", metadata_file,
        "--epochs", str(epochs),
        "--name", stage_name
    ]
    
    if resume_from:
        cmd.extend(["--resume_from", resume_from])
        
    try:
        # Run and stream output
        subprocess.check_call(cmd)
        print(f"\nStage {stage_name} completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"\nStage {stage_name} failed with error code {e.returncode}.")
        sys.exit(e.returncode)

def get_best_checkpoint(run_name):
    # Find the best checkpoint for the given run name
    checkpoint_dir = Path("models/checkpoints")
    if not checkpoint_dir.exists():
        return None
        
    # Pattern: run_name_epoch=XX_val_loss=YY.ckpt
    checkpoints = list(checkpoint_dir.glob(f"{run_name}_*.ckpt"))
    
    if not checkpoints:
        return None
        
    # Sort by modification time (most recent first) or parse loss?
    # PyTorch Lightning ModelCheckpoint saves the best one, so taking the latest created is usually safe if save_top_k=1
    # But let's verify by sorting by time
    checkpoints.sort(key=os.path.getmtime, reverse=True)
    
    return str(checkpoints[0])

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage_name', type=str, help='Name of the stage to run', default=None)
    parser.add_argument('--metadata_file', type=str, help='Path to metadata file', default=None)
    parser.add_argument('--epochs', type=int, help='Number of epochs', default=None)
    parser.add_argument('--resume_from', type=str, help='Checkpoint to resume from', default=None)
    
    args = parser.parse_args()
    
    # If arguments are provided, run specific stage
    if args.stage_name and args.metadata_file:
        print(f"Running specific stage: {args.stage_name}")
        run_training_stage(
            stage_name=args.stage_name,
            metadata_file=args.metadata_file,
            resume_from=args.resume_from,
            epochs=args.epochs if args.epochs else 10
        )
        return

    # Default sequential execution (if no args provided)
    # Define stages
    # 1. FruitNet (Base)
    # 2. Fruit Quality (Fine-tune)
    # 3. Fruits 360 (Fine-tune)
    
    # Stage 1: FruitNet
    print("Stage 1: Training on FruitNet Indian Dataset...")
    ckpt_stage1_existing = get_best_checkpoint("stage1_fruitnet")
    
    if ckpt_stage1_existing:
        print(f"Found and resuming from existing checkpoint for Stage 1: {ckpt_stage1_existing}")
    
    run_training_stage(
        stage_name="stage1_fruitnet",
        metadata_file="data/metadata_fruitnet.json",
        epochs=5, # Small number for testing/initial run
        resume_from=ckpt_stage1_existing
    )
    
    # Find checkpoint from Stage 1
    ckpt_stage1 = get_best_checkpoint("stage1_fruitnet")
    if not ckpt_stage1:
        print("Error: Could not find checkpoint from Stage 1.")
        sys.exit(1)
        
    # Stage 2: Fruit Quality
    ckpt_stage2_existing = get_best_checkpoint("stage2_quality")
    resume_checkpoint_stage2 = ckpt_stage2_existing if ckpt_stage2_existing else ckpt_stage1
    
    print(f"\nStage 2: Fine-tuning on Fruit Quality Dataset (starting from {resume_checkpoint_stage2})...")
    
    run_training_stage(
        stage_name="stage2_quality",
        metadata_file="data/metadata_fruitquality.json",
        resume_from=resume_checkpoint_stage2,
        epochs=10 # Cumulative: 5 (Stage 1) + 5 (Stage 2)
    )
    
    # Find checkpoint from Stage 2
    ckpt_stage2 = get_best_checkpoint("stage2_quality")
    if not ckpt_stage2:
        print("Error: Could not find checkpoint from Stage 2.")
        sys.exit(1)

    # Stage 3: Fruits 360
    ckpt_stage3_existing = get_best_checkpoint("stage3_fruits360")
    resume_checkpoint_stage3 = ckpt_stage3_existing if ckpt_stage3_existing else ckpt_stage2

    print(f"\nStage 3: Fine-tuning on Fruits 360 Dataset (starting from {resume_checkpoint_stage3})...")
    run_training_stage(
        stage_name="stage3_fruits360",
        metadata_file="data/metadata_fruits360.json",
        resume_from=resume_checkpoint_stage3,
        epochs=15 # Cumulative: 10 (Stage 2) + 5 (Stage 3)
    )
    
    # Find checkpoint from Stage 3
    ckpt_stage3 = get_best_checkpoint("stage3_fruits360")
    
    # Stage 4: New Fruits
    # If Stage 3 failed or didn't produce a checkpoint, try to use Stage 2 or whatever we used for Stage 3
    resume_checkpoint_stage4 = ckpt_stage3 if ckpt_stage3 else resume_checkpoint_stage3
    
    print(f"\nStage 4: Fine-tuning on New Fruits Dataset (starting from {resume_checkpoint_stage4})...")
    
    # Use metadata_stage4_fixed.json if available
    meta_file = "data/metadata_stage4_fixed.json"
    if not os.path.exists(meta_file):
         print(f"Warning: {meta_file} not found. Using data/metadata_new.json")
         meta_file = "data/metadata_new.json"

    run_training_stage(
        stage_name="stage4_newfruits",
        metadata_file=meta_file,
        resume_from=resume_checkpoint_stage4,
        epochs=20 # Cumulative: 15 (Stage 3) + 5 (Stage 4)
    )

    print("\nAll training stages completed successfully!")

if __name__ == "__main__":
    main()
