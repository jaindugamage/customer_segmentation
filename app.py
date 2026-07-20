from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.segmentation import (
    DEFAULT_MODEL_FEATURES,
    NUMERIC_FEATURES,
    DataValidationError,
    build_cluster_profiles,
    build_customer_features,
    clean_transactions,
    cluster_stability,
    compute_pca,
    evaluate_models,
    fit_clustering,
    get_segment_strategy,
    prepare_model_matrix,
    read_transaction_csv,
    score_clustering,
    suggest_k,
    valid_k_values,
)

APP_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = APP_DIR / "data" / "online_retail.csv.zip"

st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
    [data-testid="stMetricValue"] {font-size: 1.55rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_and_clean(source_bytes: bytes) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = read_transaction_csv(source_bytes)
    cleaned, quality_report = clean_transactions(raw)
    customers = build_customer_features(cleaned)
    return cleaned, quality_report, customers


@st.cache_data(show_spinner=False)
def create_matrix(customers: pd.DataFrame, model_features: tuple[str, ...], include_country: bool):
    return prepare_model_matrix(
        customers,
        selected_features=model_features,
        include_country=include_country,
    )


@st.cache_data(show_spinner=False)
def evaluate_cached(matrix: np.ndarray, k_values: tuple[int, ...], models: tuple[str, ...]) -> pd.DataFrame:
    return evaluate_models(matrix, k_values, models=models)


@st.cache_data(show_spinner=False)
def fit_cached(matrix: np.ndarray, k: int, model_name: str):
    return fit_clustering(matrix, k, model_name)


@st.cache_data(show_spinner=False)
def pca_cached(matrix: np.ndarray):
    return compute_pca(matrix)


@st.cache_data(show_spinner=False)
def score_cached(matrix: np.ndarray, labels: np.ndarray):
    return score_clustering(matrix, labels)


@st.cache_data(show_spinner=False)
def stability_cached(matrix: np.ndarray, labels: np.ndarray, k: int, model_name: str):
    return cluster_stability(matrix, labels, k, model_name)


st.sidebar.title("Customer Segmentation")
st.sidebar.caption("RFM and behavioural clustering")

source_choice = st.sidebar.radio(
    "Data source",
    ["Bundled Online Retail dataset", "Upload a CSV"],
)

if source_choice == "Upload a CSV":
    uploaded_file = st.sidebar.file_uploader("Choose transaction CSV or ZIP", type=["csv", "zip"])
    if uploaded_file is None:
        st.info("Upload a transaction CSV from the sidebar to begin.")
        st.stop()
    source_bytes = uploaded_file.getvalue()
    source_name = uploaded_file.name
else:
    if not DEFAULT_DATA_PATH.exists():
        st.error("The bundled dataset file is missing from the data folder.")
        st.stop()
    source_bytes = DEFAULT_DATA_PATH.read_bytes()
    source_name = DEFAULT_DATA_PATH.name

try:
    with st.spinner("Cleaning transactions and building customer features..."):
        cleaned_df, quality_report, customer_features = load_and_clean(source_bytes)
except (DataValidationError, OSError, ValueError) as exc:
    st.error(str(exc))
    st.stop()

st.sidebar.divider()
st.sidebar.subheader("Model inputs")
selected_features = st.sidebar.multiselect(
    "Features used for clustering",
    options=NUMERIC_FEATURES,
    default=DEFAULT_MODEL_FEATURES,
    help="Reports keep original values. The model uses clipped, log-transformed, robust-scaled versions.",
)
include_country = st.sidebar.checkbox(
    "Include country with low weight",
    value=False,
    help="Off by default so geography does not dominate customer behaviour.",
)

if len(selected_features) < 2:
    st.warning("Select at least two clustering features.")
    st.stop()

try:
    preprocessing = create_matrix(customer_features, tuple(selected_features), include_country)
except DataValidationError as exc:
    st.error(str(exc))
    st.stop()

matrix = preprocessing.matrix
k_values = valid_k_values(len(customer_features), maximum=8)
if not k_values:
    st.error("The dataset does not contain enough customers for clustering.")
    st.stop()

st.sidebar.divider()
st.sidebar.subheader("Clustering settings")
model_choice = st.sidebar.selectbox(
    "Model",
    ["KMeans", "GMM"],
    help="KMeans creates hard distance-based groups. GMM estimates probabilistic membership.",
)

auto_select = st.sidebar.checkbox(
    "Automatically suggest k",
    value=False,
    help="Runs a sampled model search. Leave off for the fastest startup.",
)

if auto_select:
    with st.spinner(f"Evaluating {model_choice} cluster counts..."):
        model_search = evaluate_cached(matrix, tuple(k_values), (model_choice,))
    suggested_k = suggest_k(model_search, model_choice)
    st.sidebar.success(f"Suggested k: {suggested_k}")
    default_k = suggested_k
else:
    model_search = None
    default_k = 3 if 3 in k_values else min(k_values)

k_selected = st.sidebar.slider(
    "Number of clusters",
    min_value=min(k_values),
    max_value=max(k_values),
    value=default_k,
)

try:
    with st.spinner("Fitting the selected segmentation..."):
        labels, _, confidence, confidence_name = fit_cached(matrix, k_selected, model_choice)
        coordinates, explained_variance = pca_cached(matrix)
        current_scores = score_cached(matrix, labels)
except DataValidationError as exc:
    st.error(str(exc))
    st.stop()

segmented = customer_features.copy()
segmented["Cluster"] = labels
segmented["AssignmentConfidence"] = confidence
segmented["PCA1"] = coordinates[:, 0]
segmented["PCA2"] = coordinates[:, 1]

profiles = build_cluster_profiles(segmented)
profile_lookup = profiles.set_index("Cluster")
segmented["Segment_Type"] = segmented["Cluster"].map(profile_lookup["Segment_Type"])
segmented["Cluster_Name"] = segmented["Cluster"].map(profile_lookup["Cluster_Name"])
analysis_signature = (
    source_name,
    len(cleaned_df),
    len(customer_features),
    tuple(selected_features),
    include_country,
    model_choice,
    k_selected,
)

page = st.sidebar.radio(
    "Page",
    [
        "Overview",
        "Cluster Explorer",
        "Feature Distributions",
        "Model Diagnostics",
        "Data Quality & Download",
    ],
)

st.sidebar.divider()
st.sidebar.caption(f"Source: {source_name}")
st.sidebar.caption(f"Valid transactions: {len(cleaned_df):,}")
st.sidebar.caption(f"Customers: {len(customer_features):,}")

if page == "Overview":
    st.title("Customer Segmentation Dashboard")
    st.caption(
        "Customer groups are discovered with unsupervised learning and described using observed purchase behaviour."
    )

    metric_cols = st.columns(5)
    metric_cols[0].metric("Customers", f"{len(segmented):,}")
    metric_cols[1].metric("Clusters", k_selected)
    metric_cols[2].metric("Silhouette", f"{current_scores['Silhouette']:.3f}")
    confidence_label = (
        "Avg. membership probability"
        if model_choice == "GMM"
        else "Avg. assignment strength"
    )
    metric_cols[3].metric(
        confidence_label,
        f"{segmented['AssignmentConfidence'].mean():.2f}",
    )
    metric_cols[4].metric("Revenue", f"{segmented['Monetary'].sum():,.0f}")

    left, right = st.columns([1.7, 1])
    with left:
        fig = px.scatter(
            segmented,
            x="PCA1",
            y="PCA2",
            color="Cluster_Name",
            hover_data={
                "CustomerID": True,
                "Recency": ":.0f",
                "Frequency": ":.0f",
                "Monetary": ":.2f",
                "AssignmentConfidence": ":.2f",
                "PCA1": False,
                "PCA2": False,
            },
            title=f"{model_choice} clusters in PCA space",
        )
        fig.update_layout(legend_title_text="Cluster")
        st.plotly_chart(fig, width="stretch")
        st.caption(
            f"The two PCA axes explain {100 * explained_variance.sum():.1f}% of transformed feature variance. "
            "Visual overlap does not automatically mean full-dimensional clusters are identical."
        )

    with right:
        fig_revenue = px.bar(
            profiles.sort_values("TotalRevenue", ascending=True),
            x="TotalRevenue",
            y="Cluster_Name",
            orientation="h",
            text="RevenueShare_%",
            title="Revenue contribution by cluster",
        )
        fig_revenue.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        st.plotly_chart(fig_revenue, width="stretch")

    st.subheader("Cluster profiles")
    profile_display = profiles[
        [
            "Cluster_Name",
            "Customers",
            "MedianRecency",
            "MedianFrequency",
            "MedianMonetary",
            "TotalRevenue",
            "RevenueShare_%",
            "AvgConfidence",
        ]
    ].copy()
    st.dataframe(
        profile_display.style.format(
            {
                "MedianRecency": "{:.0f}",
                "MedianFrequency": "{:.1f}",
                "MedianMonetary": "{:,.2f}",
                "TotalRevenue": "{:,.2f}",
                "RevenueShare_%": "{:.1f}%",
                "AvgConfidence": "{:.2f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    st.subheader("Recommended actions")
    for _, row in profiles.sort_values("TotalRevenue", ascending=False).iterrows():
        with st.expander(row["Cluster_Name"]):
            st.write(get_segment_strategy(row["Segment_Type"]))
            st.caption(
                f"{int(row['Customers']):,} customers · median recency {row['MedianRecency']:.0f} days · "
                f"median frequency {row['MedianFrequency']:.1f} · median value {row['MedianMonetary']:,.2f}"
            )

elif page == "Cluster Explorer":
    st.title("Cluster Explorer")
    cluster_name = st.selectbox(
        "Choose a cluster",
        profiles.sort_values("TotalRevenue", ascending=False)["Cluster_Name"].tolist(),
    )
    selected = segmented.loc[segmented["Cluster_Name"] == cluster_name].copy()
    row = profiles.loc[profiles["Cluster_Name"] == cluster_name].iloc[0]

    cols = st.columns(4)
    cols[0].metric("Customers", f"{len(selected):,}")
    cols[1].metric("Median recency", f"{row['MedianRecency']:.0f} days")
    cols[2].metric("Median frequency", f"{row['MedianFrequency']:.1f}")
    cols[3].metric("Revenue share", f"{row['RevenueShare_%']:.1f}%")

    st.info(get_segment_strategy(row["Segment_Type"]))

    chart_left, chart_right = st.columns(2)
    with chart_left:
        overall_medians = segmented[NUMERIC_FEATURES].median().replace(0, np.nan)
        cluster_medians = selected[NUMERIC_FEATURES].median()
        relative_profile = (
            cluster_medians.div(overall_medians)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .rename("Relative value")
            .reset_index()
            .rename(columns={"index": "Feature"})
        )
        profile_chart = px.bar(
            relative_profile,
            x="Feature",
            y="Relative value",
            title="Relative feature profile",
        )
        profile_chart.add_hline(
            y=1.0,
            line_dash="dash",
            annotation_text="All-customer median",
        )
        st.plotly_chart(profile_chart, width="stretch")
        st.caption("A value of 1.0 represents the median across all customers.")
    with chart_right:
        country_counts = (
            selected["Country"].value_counts().head(10).rename_axis("Country").reset_index(name="Customers")
        )
        st.plotly_chart(
            px.bar(country_counts, x="Country", y="Customers", title="Top countries in this cluster"),
            width="stretch",
        )

    st.subheader("Top customers by monetary value")
    columns = [
        "CustomerID",
        "Recency",
        "Frequency",
        "Monetary",
        "AvgOrderValue",
        "Country",
        "AssignmentConfidence",
    ]
    st.dataframe(
        selected[columns]
        .sort_values("Monetary", ascending=False)
        .head(100)
        .style.format(
            {
                "Monetary": "{:,.2f}",
                "AvgOrderValue": "{:,.2f}",
                "AssignmentConfidence": "{:.2f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )

elif page == "Feature Distributions":
    st.title("Feature Distributions")
    st.caption("Charts use original business values rather than transformed model inputs.")

    feature = st.selectbox("Feature", NUMERIC_FEATURES, index=NUMERIC_FEATURES.index("Monetary"))
    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            px.histogram(
                segmented,
                x=feature,
                color="Segment_Type",
                marginal="box",
                nbins=50,
                title=f"Distribution of {feature}",
            ),
            width="stretch",
        )
    with right:
        fig_box = px.box(
            segmented,
            x="Cluster_Name",
            y=feature,
            points=False,
            title=f"{feature} by cluster",
        )
        fig_box.update_xaxes(tickangle=35)
        st.plotly_chart(fig_box, width="stretch")

    st.subheader("Feature correlation")
    correlation = segmented[NUMERIC_FEATURES].corr(method="spearman")
    fig_corr, ax = plt.subplots(figsize=(9, 6))
    image = ax.imshow(correlation, aspect="auto")
    ax.set_xticks(range(len(correlation.columns)), correlation.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(correlation.index)), correlation.index)
    fig_corr.colorbar(image, ax=ax, label="Spearman correlation")
    fig_corr.tight_layout()
    st.pyplot(fig_corr, width="content")
    plt.close(fig_corr)

elif page == "Model Diagnostics":
    st.title("Model Diagnostics")

    metric_cols = st.columns(3)
    metric_cols[0].metric("Silhouette", f"{current_scores['Silhouette']:.3f}")
    metric_cols[1].metric("Calinski-Harabasz", f"{current_scores['Calinski-Harabasz']:.1f}")
    metric_cols[2].metric("Davies-Bouldin", f"{current_scores['Davies-Bouldin']:.3f}")

    st.write(
        f"**Assignment confidence:** {confidence_name}. "
        "For GMM this is a probability. For KMeans it is a distance-based separation score, not a probability."
    )
    st.plotly_chart(
        px.histogram(
            segmented,
            x="AssignmentConfidence",
            color="Cluster_Name",
            nbins=40,
            title="Assignment confidence distribution",
        ),
        width="stretch",
    )

    st.subheader("Optional full model comparison")
    st.caption("This sampled search is intentionally optional so the dashboard opens quickly.")
    if st.button("Run KMeans and GMM comparison", type="primary"):
        with st.spinner("Evaluating candidate cluster counts..."):
            st.session_state["full_model_results"] = {
                "signature": analysis_signature,
                "data": evaluate_cached(matrix, tuple(k_values), ("KMeans", "GMM")),
            }

    stored_comparison = st.session_state.get("full_model_results")
    results_df = (
        stored_comparison["data"]
        if stored_comparison and stored_comparison.get("signature") == analysis_signature
        else None
    )
    if results_df is not None:
        chart_cols = st.columns(3)
        with chart_cols[0]:
            st.plotly_chart(
                px.line(results_df, x="k", y="Silhouette", color="Model", markers=True),
                width="stretch",
            )
        with chart_cols[1]:
            st.plotly_chart(
                px.line(results_df, x="k", y="Calinski-Harabasz", color="Model", markers=True),
                width="stretch",
            )
        with chart_cols[2]:
            st.plotly_chart(
                px.line(results_df, x="k", y="Davies-Bouldin", color="Model", markers=True),
                width="stretch",
            )
        st.dataframe(results_df, width="stretch", hide_index=True)

    st.subheader("Optional stability test")
    if st.button("Run repeated-subsample ARI stability test"):
        with st.spinner("Refitting on repeated customer samples..."):
            st.session_state["stability_result"] = {
                "signature": analysis_signature,
                "data": stability_cached(matrix, labels, k_selected, model_choice),
            }

    stored_stability = st.session_state.get("stability_result")
    stability_result = (
        stored_stability["data"]
        if stored_stability and stored_stability.get("signature") == analysis_signature
        else None
    )
    if stability_result is not None:
        mean_ari, std_ari, ari_scores = stability_result
        cols = st.columns(2)
        cols[0].metric("Mean ARI", f"{mean_ari:.3f}")
        cols[1].metric("ARI variation", f"{std_ari:.3f}")
        st.write("Run scores:", ", ".join(f"{score:.3f}" for score in ari_scores))

    with st.expander("Preprocessing details"):
        st.write(
            "Selected numeric variables are clipped at their 1st and 99th percentiles, transformed with log1p, "
            "and scaled with RobustScaler. Original values are preserved for reports and downloads."
        )
        bounds_df = pd.DataFrame(
            [
                {"Feature": key, "1st percentile": value[0], "99th percentile": value[1]}
                for key, value in preprocessing.clipping_bounds.items()
            ]
        )
        st.dataframe(bounds_df, width="stretch", hide_index=True)

elif page == "Data Quality & Download":
    st.title("Data Quality & Download")

    original_rows = int(quality_report.iloc[-1]["Rows removed"] + quality_report.iloc[-1]["Rows remaining"])
    metric_cols = st.columns(4)
    metric_cols[0].metric("Original rows", f"{original_rows:,}")
    metric_cols[1].metric("Valid rows", f"{len(cleaned_df):,}")
    metric_cols[2].metric("Rows removed", f"{quality_report.iloc[-1]['Rows removed']:,}")
    metric_cols[3].metric("Customers", f"{len(customer_features):,}")

    st.subheader("Cleaning report")
    st.dataframe(quality_report, width="stretch", hide_index=True)

    st.subheader("Required CSV columns")
    st.code("InvoiceNo, StockCode, Quantity, InvoiceDate, UnitPrice, CustomerID, Country")

    export_columns = [
        "CustomerID",
        *NUMERIC_FEATURES,
        "Country",
        "FirstPurchase",
        "LastPurchase",
        "Cluster",
        "Segment_Type",
        "Cluster_Name",
        "AssignmentConfidence",
        "PCA1",
        "PCA2",
    ]
    export_df = segmented[export_columns].sort_values(["Cluster", "Monetary"], ascending=[True, False])
    st.download_button(
        "Download segmented customers",
        export_df.to_csv(index=False).encode("utf-8"),
        file_name="customer_segments.csv",
        mime="text/csv",
        width="stretch",
    )
    st.download_button(
        "Download cluster profiles",
        profiles.to_csv(index=False).encode("utf-8"),
        file_name="cluster_profiles.csv",
        mime="text/csv",
        width="stretch",
    )

    st.subheader("Segmented customer preview")
    preview_columns = [
        "CustomerID",
        "Recency",
        "Frequency",
        "Monetary",
        "Country",
        "Segment_Type",
        "Cluster",
        "AssignmentConfidence",
    ]
    st.dataframe(
        export_df[preview_columns].head(100).style.format(
            {
                "Recency": "{:.0f}",
                "Frequency": "{:.0f}",
                "Monetary": "{:,.2f}",
                "AssignmentConfidence": "{:.2f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )
