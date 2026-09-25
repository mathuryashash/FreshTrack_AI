#!/usr/bin/env python3
"""
FreshTrack AI - Cleanup Script
Removes build artifacts, cache, and development-only files.
Preserves: source code, production checkpoints, training results, documentation.
"""

import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).parent

# Files/directories to DELETE (safe to remove)
TO_DELETE = [
    # Python cache
    "**/__pycache__",
    ".pytest_cache",
    
    # Flutter build artifacts (huge)
    "mobile_app/build",
    "mobile_app/**/ephemeral",
    "mobile_app/windows/flutter/ephemeral/cpp_client_wrapper",
    
    # Training scripts (dev only)
    "scripts",
    
    # Experiment tracking
    "wandb",
    "experiment_results",
    
    # Raw training data (GBs) - keep only if you need to retrain
    # "data",  # COMMENTED OUT - uncomment only if you want to delete raw images
    
    # Old checkpoints (keep only production one in models/checkpoints/)
    "checkpoints",
    
    # Model variants not used in production (compiled .pyc files)
    "src/models/freshtrack_model_b2.pyc",
    "src/models/freshtrack_model_mobilenet.pyc",
    
    # Test files (not needed in production container)
    "tests",
    "mobile_app/test",
    
    # One-time utility scripts
    "download_model.py",
    "upload_model_to_hf.py",
    "run_backend.bat",
    "run_prediction.bat",
    "run_ui.bat",
    "render.yaml",
    "requirements-api.txt",
    
    # Empty placeholder files
    "**/.gitkeep",
    
    # Platform-specific desktop builds (keep only if targeting desktop)
    # "mobile_app/windows",
    # "mobile_app/macos",
    # "mobile_app/linux",
]

# Files to KEEP explicitly (protection list)
PROTECTED = [
    "src",
    "mobile_app/lib",
    "mobile_app/android",
    "mobile_app/ios",
    "mobile_app/web",
    "mobile_app/pubspec.yaml",
    "models/checkpoints/freshtrack_epoch=04_val_loss=0.01-v1.ckpt",
    "models/checkpoints/b0_70_30",
    "MODEL_ANALYSIS_REPORT.md",
    "freshtrack_ai_expanded_paper.tex",
    "freshtrack_ai_prd.md",
    "freshtrack_workflow.md",
    "README.md",
    "AGENTS.md",
    "CLAUDE.md",
    "requirements.txt",
    "Dockerfile",
    ".env",
    ".env.example",
    ".gitignore",
    ".dockerignore",
    ".mcp.json",
    "DECISIONS.md",
    "data/metadata.json",  # if exists
]

def is_protected(path: Path) -> bool:
    """Check if path matches any protected pattern."""
    try:
        rel = path.relative_to(ROOT)
    except ValueError:
        return False
    
    for pattern in PROTECTED:
        if rel.match(pattern) or str(rel).startswith(pattern.rstrip("*")):
            return True
    return False

def get_delete_targets() -> list[Path]:
    """Get all paths matching delete patterns, excluding protected."""
    targets = []
    for pattern in TO_DELETE:
        for path in ROOT.glob(pattern):
            if path.exists() and not is_protected(path):
                targets.append(path)
    # Deduplicate (some patterns overlap)
    unique = []
    seen = set()
    for t in sorted(targets, key=lambda p: len(p.parts), reverse=True):
        if t not in seen and not any(t.is_relative_to(s) for s in seen):
            unique.append(t)
            seen.add(t)
    return unique

def format_size(path: Path) -> str:
    """Get human-readable size."""
    if path.is_file():
        size = path.stat().st_size
    else:
        size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"

def main():
    print("=" * 70)
    print("FreshTrack AI - Cleanup Script")
    print("=" * 70)
    
    targets = get_delete_targets()
    
    if not targets:
        print("Nothing to clean up.")
        return 0
    
    print(f"\nFound {len(targets)} item(s) to delete:\n")
    
    total_size = 0
    for t in targets:
        size = format_size(t)
        total_size += t.stat().st_size if t.is_file() else sum(f.stat().st_size for f in t.rglob("*") if f.is_file())
        print(f"  {size:>10}  {t.relative_to(ROOT)}")
    
    # Calculate total properly
    total_bytes = sum(
        t.stat().st_size if t.is_file() else sum(f.stat().st_size for f in t.rglob("*") if f.is_file())
        for t in targets
    )
    print(f"Total space to reclaim: ", end="")
    # Format manually
    size = total_bytes
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            print(f"{size:.1f} {unit}")
            break
        size /= 1024
    
    print("\nProtected (will NOT be deleted):")
    for p in PROTECTED:
        if (ROOT / p).exists():
            print(f"  [OK] {p}")
    
    print("\n" + "=" * 70)
    
    # Auto-confirm if --yes flag
    if "--yes" in sys.argv or "-y" in sys.argv:
        confirm = "y"
    else:
        confirm = input("Proceed with deletion? [y/N]: ").strip().lower()
    
    if confirm != "y":
        print("Aborted.")
        return 1
    
    print("\nDeleting...")
    errors = []
    for t in targets:
        try:
            if t.is_file():
                t.unlink()
            else:
                shutil.rmtree(t)
            print(f"  [OK] Deleted: {t.relative_to(ROOT)}")
        except Exception as e:
            errors.append((t, e))
            print(f"  [FAIL] Failed: {t.relative_to(ROOT)} - {e}")
    
    print(f"\nDone. {len(targets) - len(errors)} deleted, {len(errors)} errors.")
    return 0 if not errors else 1

if __name__ == "__main__":
    sys.exit(main())