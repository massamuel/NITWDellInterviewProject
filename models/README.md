# Models

The supported classifier lives in `keras/train.py`. Run it from the repository root:

```sh
python -m models.keras.train --epochs 15
```

It uses `data/raw/mbti_1.csv` by default and saves `model.keras` and `metadata.json` to `artifacts/`. Override with `--data` and `--output` when needed. Shared preprocessing lives in `models/preprocessing.py` and is also used by the FastAPI service.

`legacy/` preserves the interview experiments, grouped by framework:

| Directory | Experiment | Dataset |
| --- | --- | --- |
| `legacy/scikit_learn/` | SGD, SVC and logistic regression | Raw MBTI posts |
| `legacy/keras/` | Original LSTM and processed-text experiments | Raw posts / processed samples |
| `legacy/pytorch/` | Original dense neural network | Raw MBTI posts |
| `legacy/spark/` | Original Spark pipeline draft | Raw MBTI posts |

These historical scripts are preserved as reference; relocating them does not modernize their algorithms or dependency APIs. Their dataset paths now resolve to the data directory, and evaluation output goes to the console instead of the deleted log files.
