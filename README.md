# Customer Segmentation Dashboard

This project groups retail customers according to their purchasing behaviour. It uses RFM analysis and additional behavioural features, then applies K-Means or Gaussian Mixture Model clustering. The results are shown in an interactive Streamlit dashboard and can be exported as CSV files.

## Main result

The bundled Online Retail dataset produces the following three-cluster K-Means result with the default project settings:

| Metric | Value |
|---|---:|
| Original rows | 541,909 |
| Valid transactions | 392,692 |
| Customers | 4,338 |
| Clusters | 3 |
| Silhouette Score | 0.308 |

| Segment | Customers | Revenue share |
|---|---:|---:|
| Champions | 1,038 | 75.1% |
| Developing Customers | 1,512 | 16.0% |
| At Risk | 1,788 | 8.9% |

The result changes when the model, features, country option, cluster count, or input dataset is changed.

## What the application does

- Loads the bundled dataset or an uploaded CSV/ZIP file.
- Removes duplicate rows, missing customer IDs, invalid dates, cancellations, returns, invalid quantities, and invalid prices.
- Creates customer-level RFM and behavioural features.
- Clips extreme model values, applies `log1p`, and scales the inputs with `RobustScaler`.
- Supports K-Means and Gaussian Mixture Model clustering.
- Calculates Silhouette, Calinski-Harabasz, and Davies-Bouldin scores.
- Provides optional model comparison and stability testing.
- Displays PCA cluster plots, feature distributions, cluster profiles, and business recommendations.
- Exports segmented customers and cluster profiles as CSV files.

## Customer features

| Feature | Description |
|---|---|
| Recency | Days since the most recent purchase |
| Frequency | Number of unique invoices |
| Monetary | Total customer spending |
| Tenure | Days between the first and latest purchase |
| Average Order Value | Average spending per order |
| Orders per Month | Average monthly order activity |
| Distinct Products | Number of different products purchased |
| Average Items per Order | Average quantity purchased per invoice |

## Dashboard pages

### Overview

Shows the main metrics, PCA cluster plot, revenue contribution, cluster profiles, and recommended actions.

### Cluster Explorer

Shows the selected cluster's customer count, median behaviour, relative feature profile, main countries, and highest-value customers.

### Feature Distributions

Shows feature distributions, cluster comparisons, and a Spearman correlation heatmap.

### Model Diagnostics

Shows the clustering metrics, assignment-strength distribution, optional K-Means/GMM comparison, and optional ARI stability test.

### Data Quality & Download

Shows the cleaning report and provides customer-segment and cluster-profile downloads.

## Project files

```text
Customer_Segmentation_Final/
├── app.py
├── requirements.txt
├── run_app.command
├── README.md
├── data/
│   └── online_retail.csv.zip
├── src/
│   ├── __init__.py
│   └── segmentation.py
└── .streamlit/
    └── config.toml
```

## Run on macOS

1. Extract the project folder.
2. Control-click `run_app.command` and select **Open**.
3. The first run creates a virtual environment and installs the required packages.
4. The dashboard opens in the browser.

If it does not open automatically, visit:

```text
http://localhost:8501
```

The same file can also be started from Terminal:

```bash
chmod +x run_app.command
./run_app.command
```

## Manual setup

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Input file format

Uploaded data must contain these columns:

```text
InvoiceNo, StockCode, Quantity, InvoiceDate, UnitPrice, CustomerID, Country
```

`Description` is optional.

Example:

```csv
InvoiceNo,StockCode,Description,Quantity,InvoiceDate,UnitPrice,CustomerID,Country
536365,85123A,WHITE HANGING HEART T-LIGHT HOLDER,6,2010-12-01 08:26:00,2.55,17850,United Kingdom
```

The application also recognizes common alternatives such as `Invoice`, `InvoiceID`, `Qty`, `Price`, and `Customer`.

## How the data is prepared

1. Exact duplicate rows are removed.
2. Records without a usable customer ID are removed.
3. Dates, quantities, and prices are converted to valid data types.
4. Cancelled invoices beginning with `C` are removed.
5. Returns, non-positive quantities, and non-positive prices are removed.
6. Transaction value is calculated as `Quantity × UnitPrice`.
7. Valid transactions are grouped into one record per customer.
8. Selected model features are clipped at the 1st and 99th percentiles.
9. The clipped values are transformed with `log1p` and scaled with `RobustScaler`.

The dashboard keeps the original business values for charts, tables, and exported files.

## Model notes

K-Means creates distance-based cluster assignments. Its assignment-strength value compares the nearest centroid with the next-nearest centroid; it is not a probability.

GMM produces probabilistic cluster memberships. The displayed value is the highest membership probability for each customer.

PCA is used only to create a two-dimensional chart. The clustering model uses the complete selected feature set.

## Limitations

- The dataset does not contain predefined correct customer segments.
- Segment names are interpretations based on cluster behaviour.
- Historical inactivity does not prove that a customer has permanently churned.
- Results depend on the selected features, preprocessing, algorithm, and cluster count.
- Business recommendations should be checked against actual company knowledge and campaign results.

## Dataset

The bundled data is the **Online Retail** dataset by Daqing Chen, published through the UCI Machine Learning Repository.

- DOI: `10.24432/C5BW33`
- Dataset licence: Creative Commons Attribution 4.0 International

## Author

Jaindu Gamage
