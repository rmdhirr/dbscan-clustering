import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
import plotly.express as px
import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as ctx
from shapely.geometry import Point

# Function to ensure unique column names
def make_unique(column_names):
    seen = {}
    result = []
    for col in column_names:
        if col not in seen:
            seen[col] = 1
            result.append(col)
        else:
            seen[col] += 1
            result.append(f"{col}_{seen[col]}")
    return result

# Function to clean latitude and longitude columns
def clean_lat_lon(df):
    if 'longitude' in df.columns and 'latitude' in df.columns:
        df['longitude'] = df['longitude'].astype(str).str.replace('.0', '').str.replace(',', '.').astype(float)
        df['latitude'] = df['latitude'].astype(str).str.replace('.0', '').str.replace(',', '.').astype(float)
    return df

# Function to preprocess the data
def preprocess_data(df):
    st.write("Initial DataFrame:")
    st.write(df)

    # Ensure unique column names
    df.columns = make_unique(df.columns.tolist())
    
    # Clean latitude and longitude columns
    df = clean_lat_lon(df)
    
    # Separate numeric and non-numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    non_numeric_df = df.select_dtypes(exclude=[np.number])

    st.write("Numeric DataFrame:")
    st.write(numeric_df)
    
    st.write("Non-numeric DataFrame:")
    st.write(non_numeric_df)
    
    if numeric_df.empty:
        raise ValueError("No numeric columns found in the dataset.")
    
    # Check for missing values and fill them with median
    numeric_df = numeric_df.apply(lambda x: x.fillna(x.median()), axis=0)
    
    st.write("Numeric DataFrame after filling missing values:")
    st.write(numeric_df)
    
    # Handle outliers by capping them to the 99th percentile
    for column in numeric_df.columns:
        upper_limit = numeric_df[column].quantile(0.99)
        numeric_df[column] = np.where(numeric_df[column] > upper_limit, upper_limit, numeric_df[column])
    
    st.write("Numeric DataFrame after handling outliers:")
    st.write(numeric_df)
    
    # Apply quantile transformation
    transformer = QuantileTransformer(output_distribution='normal', random_state=42)
    df_transformed = transformer.fit_transform(numeric_df)
    df_transformed = pd.DataFrame(df_transformed, columns=numeric_df.columns)
    
    # Merge the transformed numeric data with non-numeric data
    final_df = pd.concat([df_transformed, non_numeric_df.reset_index(drop=True)], axis=1)
    
    return final_df, numeric_df.columns

# Function to perform DBSCAN clustering
def dbscan_clustering(data, eps, min_samples):
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
    labels = db.labels_
    
    return data, labels

# Streamlit app
st.title("DBSCAN Clustering App")

# Sidebar menu
st.sidebar.title("Menu")
option = st.sidebar.selectbox("Choose an action:", ["Upload CSV", "Preprocess Data", "DBSCAN Clustering"])

# Upload CSV
if option == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    if uploaded_file is not None:
        # Forcing delimiter to semicolon based on provided dataset example
        df = pd.read_csv(uploaded_file, delimiter=';')
        df.columns = make_unique(df.columns.tolist())  # Ensure unique column names after reading
        st.session_state['df'] = df
        st.write("Uploaded Data:")
        st.write(df)

# Preprocess Data
if option == "Preprocess Data":
    if 'df' not in st.session_state:
        st.sidebar.warning("Please upload a CSV file first.")
    else:
        df = st.session_state['df']
        try:
            final_df, numeric_columns = preprocess_data(df)
            st.session_state['final_df'] = final_df
            st.session_state['numeric_columns'] = numeric_columns
            st.write("Preprocessed Data:")
            st.write(final_df)
        except Exception as e:
            st.error(f"Error during preprocessing: {e}")

# DBSCAN Clustering
if option == "DBSCAN Clustering":
    if 'final_df' not in st.session_state:
        st.sidebar.warning("Please preprocess the data first.")
    else:
        final_df = st.session_state['final_df']
        numeric_columns = st.session_state['numeric_columns']
        eps = st.sidebar.slider("Select epsilon (eps):", 0.1, 5.0, 0.5)
        min_samples = st.sidebar.slider("Select minimum samples:", 1, 10, 5)
        try:
            # Apply PCA to reduce dimensions to 2D for clustering
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(final_df[numeric_columns])
            final_df['pca1'] = pca_result[:, 0]
            final_df['pca2'] = pca_result[:, 1]
            
            # Perform DBSCAN clustering
            data_with_labels, labels = dbscan_clustering(final_df[['pca1', 'pca2']], eps, min_samples)
            
            result_df = final_df.copy()
            result_df['Cluster'] = labels
            
            st.write("DBSCAN Clustering Results:")
            st.write(result_df)
            
            # Plot PCA results
            fig = px.scatter(result_df, x='pca1', y='pca2', color='Cluster', hover_data=result_df.columns, title="DBSCAN Clustering Results (PCA Reduced)")
            st.plotly_chart(fig)

            # Plot geographical distribution
            if 'longitude' in result_df.columns and 'latitude' in result_df.columns:
                gdf = gpd.GeoDataFrame(result_df, geometry=gpd.points_from_xy(result_df.longitude, result_df.latitude))
                gdf.crs = "EPSG:4326"  # Set the coordinate reference system
                fig, ax = plt.subplots(figsize=(10, 10))
                gdf.plot(column='Cluster', cmap='viridis', legend=True, ax=ax, alpha=0.6)
                ctx.add_basemap(ax, crs=gdf.crs.to_string(), source=ctx.providers.Stamen.TonerLite)
                plt.title("Geographical Distribution of Clusters")
                st.pyplot(fig)
            else:
                st.error("Longitude and Latitude columns are required for the map visualization.")
        except Exception as e:
            st.error(f"Error during DBSCAN clustering: {e}")
