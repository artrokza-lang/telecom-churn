# 📁 data/

Recommended location for the dataset.

## Expected file

Place **`telecom_users.csv`** in this folder:

```
telecom/
├── data/
│   └── telecom_users.csv   ← dataset
└── main.py
```

The pipeline looks here **first**, then falls back to the project root (backward compatibility).

## Where to download the dataset

### Option 1 — bundled copy (fastest)

The repo already ships with `telecom_users.csv`. Move it here:

```powershell
Move-Item ..\telecom_users.csv .\telecom_users.csv
```

### Option 2 — Kaggle

Original **Telco Customer Churn** dataset (IBM sample):

🔗 https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Steps:

1. Sign in to Kaggle (free account)
2. Click **Download** (or use CLI: `kaggle datasets download -d blastchar/telco-customer-churn`)
3. Unzip → rename the CSV to **`telecom_users.csv`**
4. Drop it into this folder

> ⚠️ **Schema difference:** the Kaggle version has a slightly different structure — no `Provider` / `Age` columns, uses `SeniorCitizen` instead. The pipeline tolerates missing columns when dropping them (`errors="ignore"`), but **performance numbers may differ slightly** from the table in the root README.

## Files

| File | Tracked in git? | Purpose |
|---|---|---|
| `.gitkeep` | ✅ | Keeps the folder visible in git |
| `telecom_users.csv` | ✅ (default) | Bundled dataset for reproducibility |
| `README.md` | ✅ | This file |

If you'd rather **not** commit the CSV (e.g., swapping datasets often), add to the root `.gitignore`:

```gitignore
data/*.csv
```

`data/README.md` and `data/.gitkeep` will still be tracked — only the CSV is skipped.