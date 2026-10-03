# MMR Property Valuation Estimator

Machine-learning price estimation for residential property in the **Mumbai Metropolitan Region (MMR)** — Mumbai, Thane, Navi Mumbai, Panvel, Mira-Bhayandar, Kalyan-Dombivli and Vasai-Virar.

Given a locality, optional project name, BHK, area and a few listing attributes, the app returns an estimated market price, a likely range, the price per sq ft against the locality median, and a comparison of the same property across MMR zones.

> Semester 7 · Machine Learning project — regression with feature engineering, model comparison and ensembling, deployed as a web app.

---

## Highlights

| | |
|---|---|
| **Data** | 76,038 listings → 55,090 after de-duplication and outlier removal |
| **Coverage** | 228 localities, 9 city / zone groups |
| **Model** | Weighted ensemble of LightGBM, XGBoost and CatBoost |
| **Hold-out R² (log price)** | 0.955 |
| **Median absolute error** | 8.6 % |
| **Within ±10 % / ±20 % of listed price** | 56 % / 84 % |
| **App** | Streamlit, light and dark themes |

---

## Dataset

[Mumbai House Prices](https://www.kaggle.com/datasets/dravidvaishnav/mumbai-house-prices) (Kaggle, originally scraped from Makaan.com). Columns: `bhk`, `type`, `locality` (project / society), `area`, `price`, `price_unit`, `region`, `status`, `age`.

Prices are **asking prices from online listings**, not registered sale values. This is the main limit on achievable accuracy (see *Limitations*).

### Cleaning

- Converted `price` + `price_unit` (Lakh / Crore) to rupees.
- **Removed ~27 % exact duplicate listings.** Duplicates would appear in both train and test sets and inflate the scores.
- Trimmed extreme price-per-sq-ft values (outside the 0.5th–99.5th percentile) and implausible areas.
- Mapped the 228 `region` values to nine zones with whole-word matching, so e.g. *Kandivali* is not mistaken for *Diva*.
- Projects with fewer than three listings are grouped as `Other`, so the app can handle unseen projects.

---

## Approach

1. **Target.** Predict `log(price)`. Prices are heavily right-skewed and errors are multiplicative.
2. **Features** (`features.py`)
   - Categorical: region, zone, project, property type, possession status, listing age.
   - Numeric: BHK, area, log-area, area per bedroom, ready-to-move flag.
   - Interactions: `zone × status`, `region × BHK`.
3. **Models** (`train.py`) — Ridge, LightGBM, XGBoost and two CatBoost variants.
   - Gradient-boosted models use cross-fitted target encoding for high-cardinality columns; CatBoost uses its native ordered categorical handling.
4. **Ensembling.** Non-negative least-squares weights fitted on 5-fold **out-of-fold** predictions, then normalised.
5. **Evaluation.** Stratified 80/20 split by zone; the 20 % is never used for fitting or weight selection. The final models are refit on all data for deployment.

### Results (20 % hold-out, 11,018 listings)

| Model | Hold-out RMSE (log) |
|---|---|
| Ridge | 0.2102 |
| CatBoost (depth 6) | 0.1907 |
| CatBoost (depth 8) | 0.1862 |
| XGBoost | 0.1760 |
| LightGBM | 0.1752 |
| **Weighted blend** | **0.1728** |

Blend weights: LightGBM 0.46 · XGBoost 0.29 · CatBoost-d8 0.26 · others 0.

| Metric | Value |
|---|---|
| R² (log price) | 0.955 |
| R² (price in ₹) | 0.914 |
| Median absolute % error | 8.6 % |
| Mean absolute % error | 12.3 % |
| Mean absolute error | ≈ ₹21 lakh |

**By zone**

| Zone | Test listings | R² (log) | Mean error |
|---|---|---|---|
| Thane | 1,533 | 0.92 | 13.5 % |
| Mumbai Western Suburbs | 2,378 | 0.90 | 14.4 % |
| Mumbai Eastern Suburbs | 1,422 | 0.91 | 12.2 % |
| Mumbai South | 502 | 0.89 | 17.3 % |
| Kalyan-Dombivli | 1,390 | 0.89 | 11.0 % |
| Navi Mumbai | 709 | 0.89 | 13.7 % |
| Vasai-Virar | 532 | 0.88 | 9.8 % |
| Panvel | 875 | 0.87 | 11.3 % |
| Mira-Bhayandar | 1,677 | 0.86 | 8.4 % |

Full metrics are saved in `models/metrics.json`.

---

## Project structure

```
.
├── app.py              # Streamlit application (UI, theming, inference)
├── features.py         # Data loading, zone mapping, feature engineering
├── train.py            # Cleaning, CV, ensembling, evaluation, model export
├── data/
│   └── mumbai.csv      # Listings dataset
├── models/
│   ├── model.joblib    # Trained ensemble + metadata used by the app
│   └── metrics.json    # Hold-out metrics
├── .streamlit/
│   └── config.toml     # Base Streamlit settings
└── requirements.txt
```

---

## Run locally

```bash
git clone https://github.com/GauravGirkar/house-price-prediction-ml.git
cd house-price-prediction-ml
pip install -r requirements.txt
streamlit run app.py
```

To retrain (takes several minutes on CPU):

```bash
python train.py
```

## Deploy (Streamlit Community Cloud)

1. Push the repository to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io), choose **New app**, select the repository and set the main file to `app.py`.
3. Deploy. `models/model.joblib` (~32 MB) is committed with the repository, so no training step is needed.

---

## Limitations

- **Asking prices, not transaction prices.** Listings are negotiated down; registered values are typically lower.
- **Missing drivers of price.** No floor number, facing/view, amenities, builder reputation, distance to station or metro, or listing date. These explain much of the remaining ±9 % error.
- **Time.** The dataset is a single snapshot; it does not model price movement.
- **Uneven coverage.** Thane and the Mumbai suburbs have thousands of listings; small nodes have few, so their estimates are less reliable. Mumbai South has the widest error because its prices vary the most.
- **Indicative only.** Not a formal valuation and not suitable for lending or legal purposes.

---

## Future scope

**Data**
- Add richer attributes: floor, facing, amenities, parking, carpet vs. built-up area, builder / RERA registration.
- Add geospatial features: latitude/longitude, distance to nearest railway, metro and highway, proximity to airport, schools and hospitals — especially relevant for upcoming nodes such as Navi Mumbai Airport and the Coastal Road corridor.
- Blend in registered-transaction data (IGR Maharashtra) to calibrate asking prices to actual sale prices.
- Refresh data on a schedule to keep estimates current.

**Modelling**
- Hyperparameter optimisation (Optuna) and stacking with a learned meta-model.
- Quantile regression or conformal prediction for statistically valid price intervals instead of the current error-based range.
- Per-zone or hierarchical models for sparse localities.
- Time-aware modelling and a price-trend forecast per locality.
- Explainability with SHAP, so each estimate shows which factors raised or lowered the price.

**Product**
- Rent estimation and rental-yield calculator.
- Buy-vs-rent and EMI affordability tools.
- Map view with price heat-map by locality; compare multiple properties side by side.
- REST API (FastAPI) and batch valuation from CSV upload.
- Model monitoring for data drift, plus automated retraining.

---

## Tech stack

Python · pandas · NumPy · scikit-learn · LightGBM · XGBoost · CatBoost · Altair · Streamlit

## Acknowledgements

Dataset by [Dravid Vaishnav](https://www.kaggle.com/datasets/dravidvaishnav/mumbai-house-prices) on Kaggle; listings originate from Makaan.com.
