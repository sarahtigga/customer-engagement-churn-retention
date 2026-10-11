
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# ==========================================
# CUSTOMER ENGAGEMENT ANALYSIS (EDA)
# BANKING CUSTOMER CHURN INTELLIGENCE
# ==========================================

st.set_page_config(
    page_title="Customer Engagement Analysis",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Customer Engagement Analysis")
st.subheader("Banking Customer Churn & Retention Intelligence")
st.write(
    "Explore customer activity, product usage, account balances, "
    "and factors associated with customer churn."
)

# ------------------------------------------
# LOAD DATA
# ------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("European_Bank.csv.csv")

try:
    original_df = load_data()
except FileNotFoundError:
    st.error(
        "European_Bank.csv.csv was not found. "
        "Place the CSV in the same folder as app.py."
    )
    st.stop()
except Exception as e:
    st.error(f"Unable to load the dataset: {e}")
    st.stop()

required_columns = [
    "IsActiveMember", "Exited", "NumOfProducts", "Balance"
]
missing_columns = [
    col for col in required_columns if col not in original_df.columns
]

if missing_columns:
    st.error(
        "The dataset is missing required columns: "
        + ", ".join(missing_columns)
    )
    st.stop()

df = original_df.drop(
    columns=["CustomerId", "Surname"],
    errors="ignore"
).copy()

st.success("European Bank dataset loaded successfully!")

# ------------------------------------------
# DATASET OVERVIEW
# ------------------------------------------
st.header("1. Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Customers", f"{len(df):,}")
col2.metric("Dataset Features", len(df.columns))
col3.metric(
    "Exited Customers",
    f"{int((df['Exited'] == 1).sum()):,}"
)
col4.metric(
    "Overall Churn Rate",
    f"{(df['Exited'].eq(1).mean() * 100):.2f}%"
)

with st.expander("Preview Dataset"):
    st.dataframe(original_df.head(20), use_container_width=True)

with st.expander("Data Quality Check"):
    quality = pd.DataFrame({
        "Missing Values": original_df.isnull().sum(),
        "Data Type": original_df.dtypes.astype(str)
    })
    st.dataframe(quality, use_container_width=True)

# ------------------------------------------
# ACTIVE MEMBER VS CHURN
# ------------------------------------------
st.header("2. Customer Activity vs Churn")

active_churn = pd.crosstab(
    df["IsActiveMember"],
    df["Exited"]
).reindex(index=[0, 1], columns=[0, 1], fill_value=0)

active_churn.index = ["Inactive", "Active"]
active_churn.columns = ["Retained", "Exited"]

fig, ax = plt.subplots(figsize=(7, 4))
active_churn.plot(
    kind="bar",
    ax=ax,
    color=["#4C9F70", "#E76F51"]
)
ax.set_title("Active Member vs Churn")
ax.set_xlabel("Customer Activity")
ax.set_ylabel("Number of Customers")
ax.tick_params(axis="x", rotation=0)
ax.legend(title="Customer Status")
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

activity_rates = (
    df.groupby("IsActiveMember")["Exited"]
    .mean()
    .mul(100)
)

col1, col2 = st.columns(2)
col1.metric(
    "Inactive Customer Churn Rate",
    f"{activity_rates.get(0, 0):.2f}%"
)
col2.metric(
    "Active Customer Churn Rate",
    f"{activity_rates.get(1, 0):.2f}%"
)

# ------------------------------------------
# NUMBER OF PRODUCTS VS CHURN
# ------------------------------------------
st.header("3. Number of Products vs Churn")

product_churn = pd.crosstab(
    df["NumOfProducts"],
    df["Exited"]
).reindex(columns=[0, 1], fill_value=0)

product_churn.columns = ["Retained", "Exited"]

fig, ax = plt.subplots(figsize=(8, 4))
product_churn.plot(
    kind="bar",
    ax=ax,
    color=["#4C9F70", "#E76F51"]
)
ax.set_title("Number of Products vs Churn")
ax.set_xlabel("Number of Products")
ax.set_ylabel("Number of Customers")
ax.tick_params(axis="x", rotation=0)
ax.legend(title="Customer Status")
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

product_rates = (
    df.groupby("NumOfProducts")["Exited"]
    .mean()
    .mul(100)
    .rename("Churn Rate (%)")
    .to_frame()
)

st.subheader("Churn Rate by Product Count")
st.dataframe(product_rates.round(2), use_container_width=True)

# ------------------------------------------
# BALANCE VS CHURN
# ------------------------------------------
st.header("4. Account Balance vs Churn")

fig, ax = plt.subplots(figsize=(7, 4))
sns.boxplot(
    data=df,
    x="Exited",
    y="Balance",
    ax=ax,
    hue="Exited",
    palette={0: "#4C9F70", 1: "#E76F51"},
    legend=False
)
ax.set_title("Account Balance by Customer Status")
ax.set_xlabel("Exited (0 = Retained, 1 = Exited)")
ax.set_ylabel("Account Balance")
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

balance_summary = (
    df.groupby("Exited")["Balance"]
    .agg(["count", "mean", "median"])
    .rename(index={0: "Retained", 1: "Exited"})
)

st.subheader("Balance Summary")
st.dataframe(balance_summary.round(2), use_container_width=True)

# ------------------------------------------
# CUSTOMER ACTIVITY DISTRIBUTION
# ------------------------------------------
st.header("5. Customer Activity Levels")

activity_counts = (
    df["IsActiveMember"]
    .map({0: "Inactive", 1: "Active"})
    .value_counts()
    .reindex(["Inactive", "Active"], fill_value=0)
)

fig, ax = plt.subplots(figsize=(7, 4))
activity_counts.plot(
    kind="bar",
    ax=ax,
    color=["#E9A23B", "#4C9F70"]
)
ax.set_title("Customer Activity Level")
ax.set_xlabel("Activity Status")
ax.set_ylabel("Number of Customers")
ax.tick_params(axis="x", rotation=0)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# ------------------------------------------
# CORRELATION ANALYSIS
# ------------------------------------------
st.header("6. Correlation Analysis")

df_encoded = pd.get_dummies(
    df,
    drop_first=True,
    dtype=int
)

numeric_df = df_encoded.select_dtypes(include=np.number)
corr = numeric_df.corr()

st.subheader("Correlation Heatmap")

fig, ax = plt.subplots(figsize=(11, 7))
sns.heatmap(
    corr,
    cmap="coolwarm",
    center=0,
    linewidths=0.2,
    ax=ax
)
ax.set_title("Correlation Heatmap - Engagement & Churn")
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.subheader("Correlation with Customer Churn")

if "Exited" in corr.columns:
    corr_target = corr["Exited"].drop("Exited").sort_values()

    fig, ax = plt.subplots(figsize=(8, max(4, len(corr_target) * 0.3)))
    sns.heatmap(
        corr_target.to_frame(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        ax=ax
    )
    ax.set_title("Feature Correlation with Churn")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Correlation shows association, not causation. "
        "A weak correlation does not necessarily mean a feature "
        "has no predictive value."
    )

# ------------------------------------------
# KEY INSIGHTS
# ------------------------------------------
st.header("7. Key Findings")

inactive_rate = activity_rates.get(0, 0)
active_rate = activity_rates.get(1, 0)

if inactive_rate > active_rate:
    st.write(
        f"- Inactive customers have a higher observed churn rate "
        f"({inactive_rate:.2f}%) than active customers "
        f"({active_rate:.2f}%)."
    )
else:
    st.write(
        f"- The observed churn rate is {active_rate:.2f}% among active "
        f"customers and {inactive_rate:.2f}% among inactive customers."
    )

highest_churn_product = product_rates["Churn Rate (%)"].idxmax()
highest_product_rate = product_rates.loc[
    highest_churn_product, "Churn Rate (%)"
]

st.write(
    f"- The highest observed churn rate by product count is "
    f"{highest_product_rate:.2f}% for customers with "
    f"{highest_churn_product} product(s)."
)

st.write(
    "- Compare the balance distributions and correlation results "
    "to identify additional customer retention patterns."
)

# ------------------------------------------
# DOWNLOAD ANALYSIS
# ------------------------------------------
st.header("8. Download Analysis Results")

st.download_button(
    label="Download Customer Dataset (CSV)",
    data=original_df.to_csv(index=False).encode("utf-8"),
    file_name="customer_engagement_data.csv",
    mime="text/csv"
)

st.download_button(
    label="Download Churn Correlations (CSV)",
    data=corr["Exited"].sort_values(
        ascending=False
    ).to_csv().encode("utf-8"),
    file_name="customer_churn_correlations.csv",
    mime="text/csv"
)

st.caption(
    "Customer churn analysis supports exploratory insights and "
    "does not establish the causes of customer exits."
)

