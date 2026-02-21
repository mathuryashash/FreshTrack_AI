# FreshTrack AI

FreshTrack AI is an intelligent fruit quality assessment system that uses multi-task deep learning to classify freshness, grade quality, and predict shelf life.

## Project Structure
```
freshtrack-ai/
├── src/
│   ├── models/         # PyTorch Lightning model definitions
│   ├── data/           # Dataset loading and transforms
│   ├── training/       # Training scripts
│   ├── api/            # Inference API (To be implemented)
│   └── utils/          # Utility functions
├── scripts/            # Helper scripts (data prep, splits)
├── tests/              # Unit and integration tests
├── configs/            # Configuration files
├── docs/               # Documentation
└── archive/            # Raw dataset
```

## Setup
1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: Ensure you have `torch`, `torchvision`, `pytorch-lightning`, `wandb`, `albumentations`, `opencv-python`, `timm` installed.*

2. **Prepare Dataset**:
   The project expects the dataset in `archive/dataset`. Run the preparation script to generate metadata:
   ```bash
   python scripts/prepare_dataset.py
   ```
   This will create `data/metadata.json`.

3. **Create Splits**:
   Generate train/val/test splits:
   ```bash
   python scripts/create_splits.py
   ```
   This will create `data/metadata_with_splits.json`.

4. **Train Model**:
   Start training with:
   ```bash
   python src/training/train.py
   ```

## Known Issues
- **Python Environment**: During setup, there were issues executing python scripts from the terminal. Please verify your python installation and path if scripts fail to run.
- **Dataset**: The current dataset implementation supports "Fresh" and "Rotten" classes for Apples, Bananas, and Oranges. "Semi-ripe" and "Overripe" labels are placeholders until a more granular dataset is added.
