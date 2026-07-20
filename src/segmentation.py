from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from zipfile import BadZipFile, ZipFile
from typing import BinaryIO, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler

RANDOM_STATE = 42

REQUIRED_COLUMNS = {
    "InvoiceNo",
    "StockCode",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
}

NUMERIC_FEATURES = [
    "Recency",
    "Frequency",
    "Monetary",
    "Tenure_days",
    "AvgOrderValue",
    "Orders_per_month",
    "DistinctProducts",
    "AvgItemsPerOrder",
]

DEFAULT_MODEL_FEATURES = [
    "Recency",
    "Frequency",
    "Monetary",
    "Tenure_days",
    "AvgOrderValue",
    "Orders_per_month",
    "DistinctProducts",
]

COLUMN_ALIASES = {
    "invoiceno": "InvoiceNo",
    "invoice": "InvoiceNo",
    "invoiceid": "InvoiceNo",
    "stockcode": "StockCode",
    "productcode": "StockCode",
    "sku": "StockCode",
    "quantity": "Quantity",
    "qty": "Quantity",
    "invoicedate": "InvoiceDate",
    "date": "InvoiceDate",
    "transactiondate": "InvoiceDate",
    "unitprice": "UnitPrice",
    "price": "UnitPrice",
    "customerid": "CustomerID",
    "customer": "CustomerID",
    "country": "Country",
}


class DataValidationError(ValueError):
    pass


@dataclass(frozen=True)
class PreprocessingResult:

    matrix: np.ndarray
    processed: pd.DataFrame
    selected_features: list[str]
    country_columns: list[str]
    clipping_bounds: dict[str, tuple[float, float]]


def _canonical_column_key(name: object) -> str:
    return "".join(ch for ch in str(name).strip().lower() if ch.isalnum())


def normalise_column_names(df: pd.DataFrame) -> pd.DataFrame:
    rename_map: dict[object, str] = {}
    for column in df.columns:
        key = _canonical_column_key(column)
        if key in COLUMN_ALIASES:
            rename_map[column] = COLUMN_ALIASES[key]
        else:
            rename_map[column] = str(column).strip()
    return df.rename(columns=rename_map)


def read_transaction_csv(source: str | Path | bytes | BinaryIO) -> pd.DataFrame:
    if isinstance(source, (str, Path)):
        raw = Path(source).read_bytes()
    elif isinstance(source, bytes):
        raw = source
    else:
        raw = source.read()

    if not raw:
        raise DataValidationError("The selected data file is empty.")

    if raw[:2] == b"PK":
        try:
            with ZipFile(BytesIO(raw)) as archive:
                csv_files = [
                    name
                    for name in archive.namelist()
                    if not name.endswith("/") and name.lower().endswith(".csv")
                ]
                if not csv_files:
                    raise DataValidationError("The ZIP file does not contain a CSV file.")
                raw = archive.read(csv_files[0])
        except BadZipFile as exc:
            raise DataValidationError("The ZIP file is invalid or corrupted.") from exc

    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
        try:
            df = pd.read_csv(BytesIO(raw), encoding=encoding, low_memory=False)
            return normalise_column_names(df)
        except UnicodeDecodeError as exc:
            last_error = exc
        except pd.errors.ParserError as exc:
            raise DataValidationError(f"The CSV structure could not be parsed: {exc}") from exc

    raise DataValidationError(f"The CSV encoding could not be recognised: {last_error}")


def validate_schema(df: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_COLUMNS.difference(df.columns))
    if missing:
        raise DataValidationError(
            "Missing required columns: " + ", ".join(missing) + ". "
            "Expected columns are: " + ", ".join(sorted(REQUIRED_COLUMNS)) + "."
        )


def clean_transactions(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_schema(df)
    data = df.copy()
    initial_rows = len(data)

    report_rows: list[dict[str, int | str]] = []

    def record(step: str, before: int, after: int) -> None:
        report_rows.append(
            {"Cleaning step": step, "Rows removed": int(before - after), "Rows remaining": int(after)}
        )

    before = len(data)
    data = data.drop_duplicates().copy()
    record("Exact duplicate rows", before, len(data))

    data["CustomerID"] = data["CustomerID"].astype("string").str.strip()
    missing_customer = data["CustomerID"].isna() | data["CustomerID"].isin(["", "nan", "None", "<NA>"])
    before = len(data)
    data = data.loc[~missing_customer].copy()
    record("Missing CustomerID", before, len(data))

    data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"], errors="coerce")
    before = len(data)
    data = data.dropna(subset=["InvoiceDate"]).copy()
    record("Invalid InvoiceDate", before, len(data))

    data["Quantity"] = pd.to_numeric(data["Quantity"], errors="coerce")
    data["UnitPrice"] = pd.to_numeric(data["UnitPrice"], errors="coerce")
    before = len(data)
    data = data.dropna(subset=["Quantity", "UnitPrice"]).copy()
    record("Invalid Quantity or UnitPrice", before, len(data))

    data["InvoiceNo"] = data["InvoiceNo"].astype("string").str.strip()
    cancelled = data["InvoiceNo"].str.upper().str.startswith("C", na=False)
    before = len(data)
    data = data.loc[~cancelled].copy()
    record("Cancelled invoices", before, len(data))

    before = len(data)
    data = data.loc[data["Quantity"] > 0].copy()
    record("Returns or non-positive quantities", before, len(data))

    before = len(data)
    data = data.loc[data["UnitPrice"] > 0].copy()
    record("Non-positive prices", before, len(data))

    data["CustomerID"] = data["CustomerID"].str.replace(r"\.0$", "", regex=True)
    data["StockCode"] = data["StockCode"].astype("string").str.strip()
    data["Country"] = data["Country"].astype("string").fillna("Unknown").str.strip()
    data.loc[data["Country"].isin(["", "<NA>"]), "Country"] = "Unknown"
    data["TotalPrice"] = data["Quantity"] * data["UnitPrice"]

    before = len(data)
    finite_mask = np.isfinite(data["TotalPrice"]) & (data["TotalPrice"] > 0)
    data = data.loc[finite_mask].copy()
    record("Invalid transaction totals", before, len(data))

    if data.empty:
        raise DataValidationError("No valid purchase transactions remain after cleaning.")
    if data["CustomerID"].nunique() < 3:
        raise DataValidationError("At least three valid customers are required for clustering.")

    summary = pd.DataFrame(report_rows)
    total_row = pd.DataFrame(
        [{"Cleaning step": "Total", "Rows removed": initial_rows - len(data), "Rows remaining": len(data)}]
    )
    summary = pd.concat([summary, total_row], ignore_index=True)
    return data.sort_values("InvoiceDate").reset_index(drop=True), summary


def _mode_or_unknown(series: pd.Series) -> str:
    mode = series.dropna().mode()
    return str(mode.iloc[0]) if not mode.empty else "Unknown"


def build_customer_features(clean_df: pd.DataFrame) -> pd.DataFrame:
    reference_date = clean_df["InvoiceDate"].max().normalize() + pd.Timedelta(1, unit="D")

    customer = (
        clean_df.groupby("CustomerID", observed=True)
        .agg(
            LastPurchase=("InvoiceDate", "max"),
            FirstPurchase=("InvoiceDate", "min"),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("TotalPrice", "sum"),
            DistinctProducts=("StockCode", "nunique"),
            TotalItems=("Quantity", "sum"),
            Country=("Country", _mode_or_unknown),
        )
        .reset_index()
    )

    customer["Recency"] = (reference_date - customer["LastPurchase"].dt.normalize()).dt.days
    customer["Tenure_days"] = (
        customer["LastPurchase"].dt.normalize() - customer["FirstPurchase"].dt.normalize()
    ).dt.days + 1
    customer["Tenure_days"] = customer["Tenure_days"].clip(lower=1)
    customer["Frequency"] = customer["Frequency"].clip(lower=1)
    customer["AvgOrderValue"] = customer["Monetary"] / customer["Frequency"]

    # Use at least one active month to avoid inflated monthly order rates.
    active_months = np.maximum(customer["Tenure_days"] / 30.4375, 1.0)
    customer["Orders_per_month"] = customer["Frequency"] / active_months
    customer["AvgItemsPerOrder"] = customer["TotalItems"] / customer["Frequency"]

    customer = customer.loc[customer["Monetary"] > 0].copy()
    for column in NUMERIC_FEATURES + ["TotalItems"]:
        customer[column] = pd.to_numeric(customer[column], errors="coerce")
        customer[column] = customer[column].replace([np.inf, -np.inf], np.nan)
        customer[column] = customer[column].fillna(0.0).clip(lower=0)

    output_columns = [
        "CustomerID",
        *NUMERIC_FEATURES,
        "TotalItems",
        "Country",
        "FirstPurchase",
        "LastPurchase",
    ]
    return customer[output_columns].reset_index(drop=True)


def prepare_model_matrix(
    features: pd.DataFrame,
    selected_features: Sequence[str] = DEFAULT_MODEL_FEATURES,
    include_country: bool = False,
    top_n_countries: int = 8,
    country_weight: float = 0.35,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> PreprocessingResult:
    selected = list(dict.fromkeys(selected_features))
    invalid = [column for column in selected if column not in NUMERIC_FEATURES]
    if invalid:
        raise DataValidationError("Unsupported model features: " + ", ".join(invalid))
    if len(selected) < 2:
        raise DataValidationError("Select at least two numeric features for clustering.")

    numeric = features[selected].astype(float).copy()
    bounds: dict[str, tuple[float, float]] = {}
    for column in selected:
        low = float(numeric[column].quantile(lower_quantile))
        high = float(numeric[column].quantile(upper_quantile))
        if not np.isfinite(low):
            low = float(numeric[column].min())
        if not np.isfinite(high):
            high = float(numeric[column].max())
        if high < low:
            low, high = high, low
        numeric[column] = numeric[column].clip(lower=low, upper=high)
        bounds[column] = (low, high)

    transformed = np.log1p(numeric.clip(lower=0))
    variable_features = [
        column for column in selected if transformed[column].nunique(dropna=False) > 1
    ]
    transformed = transformed[variable_features]
    if len(variable_features) < 2 and not include_country:
        raise DataValidationError(
            "At least two selected features must vary across customers. "
            "Choose different features or provide a richer dataset."
        )

    scaler = RobustScaler(quantile_range=(10.0, 90.0))
    numeric_scaled = pd.DataFrame(
        scaler.fit_transform(transformed), columns=variable_features, index=features.index
    )

    country_columns: list[str] = []
    processed = numeric_scaled.copy()
    if include_country:
        top_countries = features["Country"].value_counts().head(top_n_countries).index
        grouped = features["Country"].where(features["Country"].isin(top_countries), "Other")
        dummies = pd.get_dummies(grouped, prefix="Country", dtype=float)
        dummies = dummies * float(country_weight)
        country_columns = dummies.columns.tolist()
        processed = pd.concat([processed, dummies], axis=1)

    processed = processed.loc[:, processed.nunique(dropna=False) > 1]
    country_columns = [column for column in country_columns if column in processed.columns]
    if processed.shape[1] < 2:
        raise DataValidationError(
            "The selected inputs do not produce at least two varying model columns."
        )

    matrix = processed.to_numpy(dtype=float)
    if not np.isfinite(matrix).all():
        raise DataValidationError("The model matrix contains invalid numeric values after preprocessing.")

    return PreprocessingResult(
        matrix=matrix,
        processed=processed,
        selected_features=variable_features,
        country_columns=country_columns,
        clipping_bounds=bounds,
    )


def _build_model(
    model_name: str,
    k: int,
    random_state: int = RANDOM_STATE,
    fast_evaluation: bool = False,
):
    if model_name == "KMeans":
        return KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=2 if fast_evaluation else 5,
            max_iter=300 if fast_evaluation else 500,
        )
    if model_name == "GMM":
        return GaussianMixture(
            n_components=k,
            covariance_type="diag",
            random_state=random_state,
            n_init=1 if fast_evaluation else 2,
            reg_covar=1e-6,
            max_iter=250 if fast_evaluation else 500,
        )
    raise ValueError("Model must be 'KMeans' or 'GMM'.")


def valid_k_values(n_customers: int, maximum: int = 10) -> list[int]:
    upper = min(maximum, n_customers - 1)
    if upper < 2:
        return []
    return list(range(2, upper + 1))


def evaluate_models(
    matrix: np.ndarray,
    k_values: Iterable[int],
    models: Sequence[str] = ("KMeans", "GMM"),
    silhouette_sample_size: int = 400,
    evaluation_sample_size: int = 800,
) -> pd.DataFrame:
    results: list[dict[str, float | int | str]] = []
    n_full = matrix.shape[0]
    if n_full > evaluation_sample_size:
        rng = np.random.default_rng(RANDOM_STATE)
        eval_indices = rng.choice(n_full, size=evaluation_sample_size, replace=False)
        evaluation_matrix = matrix[eval_indices]
    else:
        evaluation_matrix = matrix
    n = evaluation_matrix.shape[0]

    for model_name in models:
        for k in k_values:
            try:
                model = _build_model(model_name, int(k), fast_evaluation=True)
                labels = model.fit_predict(evaluation_matrix)
                unique = np.unique(labels)
                if len(unique) < 2 or len(unique) >= n:
                    continue
                sample_size = min(silhouette_sample_size, n)
                silhouette = silhouette_score(
                    evaluation_matrix,
                    labels,
                    sample_size=sample_size if sample_size < n else None,
                    random_state=RANDOM_STATE,
                )
                results.append(
                    {
                        "Model": model_name,
                        "k": int(k),
                        "Silhouette": float(silhouette),
                        "Calinski-Harabasz": float(calinski_harabasz_score(evaluation_matrix, labels)),
                        "Davies-Bouldin": float(davies_bouldin_score(evaluation_matrix, labels)),
                    }
                )
            except (ValueError, np.linalg.LinAlgError):
                continue

    result = pd.DataFrame(results)
    if result.empty:
        raise DataValidationError("No valid clustering solution could be fitted to this dataset.")
    return result.sort_values(["Model", "k"]).reset_index(drop=True)



def score_clustering(
    matrix: np.ndarray, labels: np.ndarray, sample_size: int = 500
) -> dict[str, float]:
    unique = np.unique(labels)
    if len(unique) < 2 or len(unique) >= len(labels):
        raise DataValidationError("The fitted model did not produce a valid multi-cluster solution.")
    n = len(labels)
    size = min(sample_size, n)
    silhouette = silhouette_score(
        matrix,
        labels,
        sample_size=size if size < n else None,
        random_state=RANDOM_STATE,
    )
    return {
        "Silhouette": float(silhouette),
        "Calinski-Harabasz": float(calinski_harabasz_score(matrix, labels)),
        "Davies-Bouldin": float(davies_bouldin_score(matrix, labels)),
    }

def suggest_k(results: pd.DataFrame, model_name: str) -> int:
    subset = results.loc[results["Model"] == model_name].copy()
    if subset.empty:
        raise DataValidationError(f"No valid evaluation results are available for {model_name}.")
    ranked = subset.sort_values(
        ["Silhouette", "Davies-Bouldin", "k"], ascending=[False, True, True]
    )
    best = ranked.iloc[0]
    # Prefer a practical multi-segment result when it remains close to the best score.
    if int(best["k"]) == 2:
        practical = subset.loc[subset["k"] >= 3].sort_values(
            ["Silhouette", "Davies-Bouldin", "k"], ascending=[False, True, True]
        )
        if not practical.empty and float(practical.iloc[0]["Silhouette"]) >= 0.75 * float(best["Silhouette"]):
            return int(practical.iloc[0]["k"])
    return int(best["k"])


def fit_clustering(
    matrix: np.ndarray, k: int, model_name: str
) -> tuple[np.ndarray, object, np.ndarray, str]:
    model = _build_model(model_name, k)
    labels = model.fit_predict(matrix)
    produced_clusters = len(np.unique(labels))
    if produced_clusters < 2:
        raise DataValidationError(
            "The selected data produced only one cluster. Choose fewer or more varied inputs."
        )
    if produced_clusters < k:
        raise DataValidationError(
            f"The model could form only {produced_clusters} distinct clusters instead of {k}. "
            "Reduce k or choose more varied features."
        )

    if model_name == "GMM":
        probabilities = model.predict_proba(matrix)
        confidence = probabilities.max(axis=1)
        confidence_name = "Membership probability"
    else:
        # Compare the nearest centroid with the next-nearest centroid.
        distances = np.sort(model.transform(matrix), axis=1)
        if distances.shape[1] > 1:
            confidence = 1.0 - distances[:, 0] / np.maximum(distances[:, 1], 1e-12)
        else:
            confidence = np.ones(matrix.shape[0])
        confidence = np.clip(confidence, 0.0, 1.0)
        confidence_name = "Distance separation score"

    return labels.astype(int), model, confidence.astype(float), confidence_name


def compute_pca(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coordinates = pca.fit_transform(matrix)
    return coordinates, pca.explained_variance_ratio_


def cluster_stability(
    matrix: np.ndarray,
    base_labels: np.ndarray,
    k: int,
    model_name: str,
    n_runs: int = 3,
    sample_fraction: float = 0.60,
    max_sample_size: int = 800,
) -> tuple[float, float, list[float]]:
    n = matrix.shape[0]
    sample_size = min(n, max_sample_size, max(k * 5, int(round(n * sample_fraction))))
    scores: list[float] = []

    for run in range(n_runs):
        rng = np.random.default_rng(RANDOM_STATE + run)
        indices = rng.choice(n, size=sample_size, replace=False)
        model = _build_model(
            model_name,
            k,
            random_state=RANDOM_STATE + run + 1,
            fast_evaluation=True,
        )
        labels = model.fit_predict(matrix[indices])
        scores.append(float(adjusted_rand_score(base_labels[indices], labels)))

    return float(np.mean(scores)), float(np.std(scores)), scores


def assign_cluster_types(
    profiles: pd.DataFrame, customer_features: pd.DataFrame
) -> pd.DataFrame:
    output = profiles.copy()
    q33 = customer_features[["Recency", "Frequency", "Monetary", "Tenure_days"]].quantile(0.33)
    q50 = customer_features[["Recency", "Frequency", "Monetary", "Tenure_days"]].quantile(0.50)
    q67 = customer_features[["Recency", "Frequency", "Monetary", "Tenure_days"]].quantile(0.67)

    segment_types: list[str] = []
    for _, row in output.iterrows():
        recency = row["MedianRecency"]
        frequency = row["MedianFrequency"]
        monetary = row["MedianMonetary"]
        tenure = row["MedianTenure"]

        if recency >= q67["Recency"] and (
            monetary >= q67["Monetary"] or frequency >= q67["Frequency"]
        ):
            label = "At Risk — High Value"
        elif recency >= q67["Recency"]:
            label = "At Risk"
        elif monetary >= q67["Monetary"] and frequency >= q67["Frequency"]:
            label = "Champions"
        elif frequency >= q67["Frequency"] and recency <= q50["Recency"]:
            label = "Loyal Regulars"
        elif monetary >= q67["Monetary"]:
            label = "High-Value Customers"
        elif recency <= q33["Recency"] and tenure <= q33["Tenure_days"]:
            label = "New & Promising"
        elif frequency <= q33["Frequency"] and monetary <= q33["Monetary"]:
            label = "Occasional Customers"
        else:
            label = "Developing Customers"
        segment_types.append(label)

    output["Segment_Type"] = segment_types
    output["Cluster_Name"] = output.apply(
        lambda row: f"{row['Segment_Type']} · Cluster {int(row['Cluster'])}", axis=1
    )
    return output


def build_cluster_profiles(segmented: pd.DataFrame) -> pd.DataFrame:
    profiles = (
        segmented.groupby("Cluster", observed=True)
        .agg(
            Customers=("CustomerID", "nunique"),
            MedianRecency=("Recency", "median"),
            MedianFrequency=("Frequency", "median"),
            MedianMonetary=("Monetary", "median"),
            MedianTenure=("Tenure_days", "median"),
            AvgRecency=("Recency", "mean"),
            AvgFrequency=("Frequency", "mean"),
            AvgMonetary=("Monetary", "mean"),
            TotalRevenue=("Monetary", "sum"),
            AvgConfidence=("AssignmentConfidence", "mean"),
        )
        .reset_index()
    )
    total_revenue = profiles["TotalRevenue"].sum()
    profiles["RevenueShare_%"] = np.where(
        total_revenue > 0, 100.0 * profiles["TotalRevenue"] / total_revenue, 0.0
    )
    return assign_cluster_types(profiles, segmented)


def get_segment_strategy(segment_type: str) -> str:
    strategies = {
        "Champions": (
            "Protect loyalty with priority service, early access, recognition, and relevant premium offers."
        ),
        "Loyal Regulars": (
            "Use replenishment reminders, personalised recommendations, referrals, and thoughtful cross-sells."
        ),
        "High-Value Customers": (
            "Encourage repeat purchases through tailored follow-ups, account support, and high-value bundles."
        ),
        "New & Promising": (
            "Deliver a strong onboarding journey, second-purchase incentive, and product education."
        ),
        "Developing Customers": (
            "Increase engagement with relevant recommendations, progress-based rewards, and category discovery."
        ),
        "Occasional Customers": (
            "Use low-pressure reminders, entry bundles, seasonal campaigns, and clear value communication."
        ),
        "At Risk": (
            "Run a measured reactivation campaign and collect feedback before using discounts broadly."
        ),
        "At Risk — High Value": (
            "Prioritise personal outreach, service recovery, feedback, and a carefully targeted win-back offer."
        ),
    }
    return strategies.get(segment_type, "Review the cluster profile before choosing a campaign.")
