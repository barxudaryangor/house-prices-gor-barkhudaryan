# House Prices — Advanced Regression Techniques

**Author:** Gor Barxudaryan
**Course:** Picsart AI/ML — Classical Machine Learning, final project

## Project overview

The goal is to predict the sale price of houses in Ames, Iowa, for the Kaggle competition
[House Prices — Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques).
The training set has 1,460 houses with known prices and the test set has 1,459 houses whose prices must be predicted.
Each house is described by 79 features (size, quality, location, age, …).

Kaggle scores the predictions with the RMSE between the logarithm of the predicted and the real price, so all models
predict `log(1 + SalePrice)` and are compared with 5-fold cross-validation on that scale.

## Project structure

```text
house-prices-gor-barxudaryan/
├── data/
│   ├── raw/                    # original Kaggle files (not modified)
│   └── processed/              # train_processed.csv, test_processed.csv (created by 01_EDA)
├── notebooks/
│   ├── 01_EDA.ipynb            # Part B: exploration, cleaning, new features, encoding
│   └── 02_Models.ipynb         # Part C: models, tuning, comparison, error analysis, submission
├── results/
│   ├── figures/                # 9 saved plots
│   └── model_comparison.csv    # CV results of all models
├── submissions/
│   ├── submission_lasso.csv    # predictions uploaded to Kaggle
│   └── kaggle_score.png        # screenshot of the Kaggle score
└── README.md
```

## How to run

1. Install Python 3.10+ and the packages:
   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn jupyter
   ```
2. Make sure the four Kaggle files (`train.csv`, `test.csv`, `data_description.txt`, `sample_submission.csv`)
   are in `data/raw/`.
3. Open `notebooks/01_EDA.ipynb` and run all cells. It creates the processed datasets and figures 01–06.
4. Open `notebooks/02_Models.ipynb` and run all cells (about 1 minute). It creates `results/model_comparison.csv`,
   figures 07–09 and `submissions/submission_lasso.csv`.

All paths are relative to the `notebooks/` folder and `random_state=42` is used everywhere.

## Data preparation (01_EDA)

- **Target:** `SalePrice` is right-skewed (skewness 1.88); after `log1p` it is almost symmetric (0.12).
- **Missing values:** for most columns an empty value means that the house does not have the feature (no pool,
  garage, basement …), so I filled them with `"None"` or `0`. `LotFrontage` was filled with the median of the
  neighborhood, and the few other gaps with the most frequent value or median. All values were computed on the
  training set only.
- **Outliers:** I removed 2 training houses (Ids 524 and 1299) with more than 4,000 sq ft that sold for less than
  \$200,000 (both partial sales). No test rows were removed.
- **New features:** `TotalSF`, `HouseAge`, `RemodAge`, `TotalBathrooms`, `TotalPorchSF`, `HasGarage`.
- **Encoding:** quality columns (Po < Fa < TA < Gd < Ex) were encoded as numbers 1–5 and the other text columns were
  one-hot encoded. The test set was aligned to the same 260 columns as the training set.

## Models and validation (02_Models)

- All models use the same `KFold(n_splits=5, shuffle=True, random_state=42)` folds.
- `DummyRegressor` (always predicts the mean) is the baseline.
- Models: Linear Regression, Ridge, Lasso, Random Forest and Gradient Boosting. Linear models use a pipeline with
  `StandardScaler`, so the scaler is fitted only on the training part of each fold.
- Ridge (`alpha`), Lasso (`alpha`) and Gradient Boosting (`n_estimators`, `learning_rate`, `max_depth`) were tuned
  with `GridSearchCV` on the same folds.

## Results

| Rank | Model | Stage | CV RMSE | Std | Parameters |
|---:|---|---|---:|---:|---|
| 1 | **Lasso** | tuned | **0.1128** | 0.0067 | alpha=0.005 |
| 2 | Gradient Boosting | tuned | 0.1162 | 0.0054 | learning_rate=0.03, max_depth=3, n_estimators=1000 |
| 3 | Ridge | tuned | 0.1187 | 0.0085 | alpha=500 |
| 4 | Lasso | initial | 0.1197 | 0.0089 | alpha=0.001 |
| 5 | Gradient Boosting | initial | 0.1232 | 0.0062 | defaults |
| 6 | Ridge | initial | 0.1304 | 0.0129 | alpha=1 |
| 7 | Linear Regression | initial | 0.1309 | 0.0130 | — |
| 8 | Random Forest | initial | 0.1352 | 0.0060 | n_estimators=300 |
| 9 | Baseline | mean price | 0.3993 | 0.0168 | — |

**Selected model: Lasso (`alpha=0.005`), CV RMSE 0.1128.** It keeps 76 of the 260 features. On the log scale the
price is close to a linear function of quality, size, age and location, and the L1 penalty removes the many weak
dummy columns.

**Kaggle public score:** _pending_ — see `submissions/kaggle_score.png`.

## Error analysis

Out-of-fold predictions of the Lasso model have an RMSE of 0.1130 and R² 0.92 on the log scale; the median error
is about 6% of the price. The largest errors are non-normal sales (family, abnormal and partial sales; 7 of the 10
largest errors) and cheap old houses that the model overestimates, for example Id 633 (actual \$82,500, predicted
\$183,532). The neighborhood with the highest error is `IDOTRR`.

One test house (Id 2550, 5,095 sq ft) got an unrealistic prediction of about \$1.6 million, so the predictions are
clipped to the range of training prices (\$34,900 – \$755,000). This affects only that house.

## Limitations

- The tuned scores are slightly optimistic, because the hyperparameters were chosen and scored on the same folds.
- `LotFrontage` medians were computed on the whole training set before cross-validation (a very small leak).
- The differences between the best models are small compared to the variation between folds.


