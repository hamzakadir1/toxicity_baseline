# Drug Toxicity Prediction — Baseline (sklearn vs PyTorch)

Compare classic ML (LogReg/SVM/RF/KNN) with a small PyTorch MLP for binary toxicity.  
Includes stratified 5-fold CV, fixed seeds, ROC-AUC & PR-AUC (for class imbalance), duplicate-ID leakage warning, and a calibration (reliability) curve.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/train_sklearn.py --data_csv data/sample_tox21.csv --id_col id --label_col y --model_all --kfold 5 --seed 42
python src/train_pytorch.py --data_csv data/sample_tox21.csv --id_col id --label_col y --epochs 50 --seed 42
