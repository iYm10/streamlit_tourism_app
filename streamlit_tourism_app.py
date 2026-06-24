import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title="Saudi Arabia Tourism EDA",
    page_icon="🇸🇦",
    layout="wide"
)

# -----------------------------
# Helper functions
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("tourism_data.csv")
    return df


def clean_data(df):
    df = df.copy()

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Make sure year is integer if the column exists
    if "YEARS" in df.columns:
        df["YEARS"] = df["YEARS"].astype(int)

    # Create new feature: spending per tourist
    if "Tourists_Spending" in df.columns and "Tourists_Number" in df.columns:
        df["Spending_Per_Tourist"] = df["Tourists_Spending"] / df["Tourists_Number"]
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(subset=["Spending_Per_Tourist"], inplace=True)

    return df


def format_number(value):
    if pd.isna(value):
        return "0"
    return f"{value:,.0f}"


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("Saudi Tourism EDA")
st.caption("Prepared by Yahya Majrashi")
raw_df = load_data()

original_shape = raw_df.shape
duplicate_count = raw_df.duplicated().sum()
missing_before = raw_df.isnull().sum().sum()

df = clean_data(raw_df)

# Sidebar filters
years = sorted(df["YEARS"].dropna().unique()) if "YEARS" in df.columns else []
provinces = sorted(df["Province"].dropna().unique()) if "Province" in df.columns else []
tourism_types = sorted(df["Tourism_Type"].dropna().unique()) if "Tourism_Type" in df.columns else []

selected_years = st.sidebar.multiselect("Select years", years, default=years)
selected_provinces = st.sidebar.multiselect("Select provinces", provinces, default=provinces)
selected_types = st.sidebar.multiselect("Select tourism types", tourism_types, default=tourism_types)

filtered_df = df.copy()
if selected_years:
    filtered_df = filtered_df[filtered_df["YEARS"].isin(selected_years)]
if selected_provinces:
    filtered_df = filtered_df[filtered_df["Province"].isin(selected_provinces)]
if selected_types:
    filtered_df = filtered_df[filtered_df["Tourism_Type"].isin(selected_types)]

# -----------------------------
# Header
# -----------------------------
st.title("Exploratory Data Analysis of Saudi Arabia Tourism Trends (2015–2024)")
st.write(
    "This Streamlit app presents an interactive EDA dashboard for Saudi Arabia tourism data. "
    "It analyzes tourist numbers, tourism spending, overnight stays, provinces, tourism types, correlations, and outliers."
)

# -----------------------------
# Overview metrics
# -----------------------------
st.header("1. Data Overview")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Original rows", format_number(original_shape[0]))
c2.metric("Rows after cleaning", format_number(df.shape[0]))
c3.metric("Duplicate rows removed", format_number(duplicate_count))
c4.metric("Missing values before cleaning", format_number(missing_before))

c5, c6, c7 = st.columns(3)
c5.metric("Total tourists", format_number(filtered_df["Tourists_Number"].sum()))
c6.metric("Total spending", format_number(filtered_df["Tourists_Spending"].sum()))
c7.metric("Total overnight stays", format_number(filtered_df["Overnight_Stay"].sum()))

with st.expander("Show data sample"):
    st.dataframe(filtered_df.head(20), use_container_width=True)

with st.expander("Show descriptive statistics"):
    st.dataframe(filtered_df.describe(), use_container_width=True)

# -----------------------------
# Distributions
# -----------------------------
st.header("2. Explore One Column at a Time")

col1, col2 = st.columns(2)
with col1:
    fig = px.histogram(
        filtered_df,
        x="Tourists_Number",
        nbins=30,
        title="Distribution of Tourists Number"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.info("Most tourism records have relatively small tourist numbers, while a few records have very high tourist counts. This indicates a right-skewed distribution.")

with col2:
    fig = px.histogram(
        filtered_df,
        x="Tourists_Spending",
        nbins=30,
        title="Distribution of Tourism Spending"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.info("Most spending values are low to moderate, but a few records show exceptionally high spending values.")

col3, col4 = st.columns(2)
with col3:
    fig = px.histogram(
        filtered_df,
        x="Avg_Stay",
        nbins=30,
        title="Distribution of Average Stay"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.info("Most average stays are short, while some records show longer stays, which may represent specific tourism categories or destinations.")

with col4:
    if "Spending_Per_Tourist" in filtered_df.columns:
        fig = px.histogram(
            filtered_df,
            x=np.log1p(filtered_df["Spending_Per_Tourist"]),
            nbins=30,
            title="Log Distribution of Spending Per Tourist",
            labels={"x": "Log Spending Per Tourist"}
        )
        st.plotly_chart(fig, use_container_width=True)
        st.info("The log transformation reduces the impact of outliers and makes the spending-per-tourist distribution easier to interpret.")

# -----------------------------
# Trends over time
# -----------------------------
st.header("3. Tourism Trends Over Time")

yearly = filtered_df.groupby("YEARS", as_index=False).agg(
    Tourists_Number=("Tourists_Number", "sum"),
    Tourists_Spending=("Tourists_Spending", "sum"),
    Overnight_Stay=("Overnight_Stay", "sum")
)

col1, col2 = st.columns(2)
with col1:
    fig = px.line(yearly, x="YEARS", y="Tourists_Number", markers=True, title="Total Tourists Number Over Years")
    st.plotly_chart(fig, use_container_width=True)
    st.success("Tourist numbers declined sharply in 2020 and then increased rapidly from 2021 to 2024, reaching the highest level in 2024.")

with col2:
    fig = px.line(yearly, x="YEARS", y="Tourists_Spending", markers=True, title="Total Tourism Spending Over Years")
    st.plotly_chart(fig, use_container_width=True)
    st.success("Tourism spending dropped in 2020 and recovered strongly afterward, reaching a record high in 2024.")

fig = px.line(yearly, x="YEARS", y="Overnight_Stay", markers=True, title="Total Overnight Stay Over Years")
st.plotly_chart(fig, use_container_width=True)
st.success("Overnight stays declined in 2020 and increased in the following years, suggesting strong recovery in tourism activity and visitor engagement.")

# -----------------------------
# Province and tourism type comparisons
# -----------------------------
st.header("4. Province and Tourism Type Comparisons")

province_summary = filtered_df.groupby("Province", as_index=False).agg(
    Tourists_Number=("Tourists_Number", "sum"),
    Tourists_Spending=("Tourists_Spending", "sum")
)

col1, col2 = st.columns(2)
with col1:
    top_provinces = province_summary.sort_values("Tourists_Number", ascending=False).head(10)
    fig = px.bar(
        top_provinces,
        x="Tourists_Number",
        y="Province",
        orientation="h",
        title="Top 10 Provinces by Tourists Number"
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)
    st.info("Makkah attracts the highest number of tourists by a large margin, followed by Riyadh and the Eastern Region.")

with col2:
    top_spending = province_summary.sort_values("Tourists_Spending", ascending=False).head(10)
    fig = px.bar(
        top_spending,
        x="Tourists_Spending",
        y="Province",
        orientation="h",
        title="Top 10 Provinces by Tourism Spending"
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)
    st.info("Makkah generates the highest tourism spending, followed by Riyadh and the Eastern Region.")

type_summary = filtered_df.groupby("Tourism_Type", as_index=False).agg(
    Tourists_Number=("Tourists_Number", "sum"),
    Tourists_Spending=("Tourists_Spending", "sum")
)

col3, col4 = st.columns(2)
with col3:
    fig = px.bar(type_summary, x="Tourism_Type", y="Tourists_Number", title="Tourists Number by Tourism Type")
    st.plotly_chart(fig, use_container_width=True)
    st.info("Domestic tourism attracts substantially more tourists than inbound tourism.")

with col4:
    fig = px.bar(type_summary, x="Tourism_Type", y="Tourists_Spending", title="Tourism Spending by Tourism Type")
    st.plotly_chart(fig, use_container_width=True)
    st.info("Inbound tourism generates higher total spending despite having fewer tourists, suggesting higher spending per trip.")

# -----------------------------
# Relationships and correlations
# -----------------------------
st.header("5. Relationships Between Variables")

col1, col2 = st.columns(2)
with col1:
    fig = px.scatter(
        filtered_df,
        x="Tourists_Number",
        y="Tourists_Spending",
        color="Tourism_Type",
        hover_data=["Province", "YEARS"],
        title="Tourists Number vs Tourism Spending"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.info("Tourism spending generally increases as the number of tourists increases, although other factors also influence spending.")

with col2:
    fig = px.scatter(
        filtered_df,
        x="Overnight_Stay",
        y="Tourists_Spending",
        color="Tourism_Type",
        hover_data=["Province", "YEARS"],
        title="Overnight Stay vs Tourism Spending"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.info("Overnight stays show a strong positive relationship with tourism spending, making length of stay an important revenue factor.")

numeric_df = filtered_df.select_dtypes(include=np.number)
corr = numeric_df.corr()
fig = px.imshow(
    corr,
    text_auto=True,
    title="Correlation Heatmap",
    color_continuous_scale="RdBu_r",
    zmin=-1,
    zmax=1
)
st.plotly_chart(fig, use_container_width=True)
st.success("The strongest relationship is between Overnight Stay and Tourists Spending, showing that longer stays are strongly associated with higher tourism revenue.")

# -----------------------------
# Outlier Analysis
# -----------------------------
st.header("6. Outlier Analysis")

col1, col2 = st.columns(2)
with col1:
    fig = px.box(filtered_df, x="Tourists_Spending", title="Boxplot of Tourism Spending")
    st.plotly_chart(fig, use_container_width=True)
    st.info("Tourism spending contains several outliers. These may represent provinces or years with exceptionally high tourism activity and revenue.")

with col2:
    fig = px.box(filtered_df, x="Tourists_Number", title="Boxplot of Tourists Number")
    st.plotly_chart(fig, use_container_width=True)
    st.info("Tourist numbers contain high-value outliers, confirming that tourism activity is concentrated in a limited number of observations.")

# -----------------------------
# Final insights and conclusion
# -----------------------------
st.header("7. Final Insights")

st.markdown(
    """
1. Makkah attracted the highest number of tourists and generated the highest tourism spending.
2. Domestic tourism attracted more tourists than inbound tourism.
3. Inbound tourism generated higher total spending despite having fewer tourists.
4. Tourist numbers and tourism spending both declined in 2020 and recovered strongly afterward.
5. Overnight stays had the strongest relationship with tourism spending.
6. Tourism activity and spending are concentrated in a few key provinces.
7. Outliers are meaningful because they represent unusually high tourism activity rather than simple data errors.
    """
)

st.header("8. Conclusion")
st.write(
    "This EDA dashboard shows that Saudi Arabia's tourism sector experienced strong post-pandemic recovery, "
    "with tourist numbers, spending, and overnight stays increasing significantly after 2020. "
    "Makkah, Riyadh, and the Eastern Region are the main tourism centers. "
    "The analysis also shows that inbound tourists spend more overall, while domestic tourists represent the largest visitor volume."
)
