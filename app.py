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
from sklearn.metrics.pairwise import euclidean_distances

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
    lat_lon_columns = ['latitude', 'longitude']
    numeric_df = df.drop(columns=lat_lon_columns, errors='ignore').select_dtypes(include=[np.number])
    non_numeric_df = df.select_dtypes(exclude=[np.number])
    lat_lon_df = df[lat_lon_columns] if set(lat_lon_columns).issubset(df.columns) else pd.DataFrame()

    st.write("Numeric DataFrame:")
    st.write(numeric_df)
    
    st.write("Non-numeric DataFrame:")
    st.write(non_numeric_df)

    st.write("Latitude and Longitude DataFrame:")
    st.write(lat_lon_df)
    
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
    
    # Merge the transformed numeric data with non-numeric data and lat/lon data
    final_df = pd.concat([df_transformed, non_numeric_df.reset_index(drop=True), lat_lon_df.reset_index(drop=True)], axis=1)
    
    return final_df, numeric_df.columns

# Function to perform DBSCAN clustering
def dbscan_clustering(data, eps, min_samples):
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
    labels = db.labels_
    
    return data, labels

# Initialize session state
if 'page' not in st.session_state:
    st.session_state['page'] = 'Upload CSV'

# Title for the app
st.title("DBSCAN Clustering App")

# Sidebar navigation buttons with equal size and centered text
def sidebar_button(label):
    st.sidebar.markdown(
        f"""
        <style>
        div.stButton > button:first-child {{
            width: 100%;
            height: 50px;
            text-align: center;
        }}
        </style>
        """, unsafe_allow_html=True)
    return st.sidebar.button(label)

if sidebar_button('Upload CSV'):
    st.session_state['page'] = 'Upload CSV'
if sidebar_button('Preprocess Data'):
    st.session_state['page'] = 'Preprocess Data'
if sidebar_button('DBSCAN Clustering'):
    st.session_state['page'] = 'DBSCAN Clustering'

# Upload CSV
if st.session_state['page'] == 'Upload CSV':
    st.header("Upload CSV")
    uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])
    if uploaded_file is not None:
        # Forcing delimiter to semicolon based on provided dataset example
        df = pd.read_csv(uploaded_file, delimiter=';')
        df.columns = make_unique(df.columns.tolist())  # Ensure unique column names after reading
        st.session_state['df'] = df
        st.write("Uploaded Data:")
        st.write(df)
    if st.button("Proceed to Preprocessing"):
        st.session_state['page'] = 'Preprocess Data'
        st.experimental_rerun()

# Preprocess Data
if st.session_state['page'] == 'Preprocess Data':
    st.header("Preprocess Data")
    if 'df' not in st.session_state:
        st.warning("Please upload a CSV file first.")
    else:
        df = st.session_state['df']
        try:
            final_df, numeric_columns = preprocess_data(df)
            st.session_state['final_df'] = final_df
            st.session_state['numeric_columns'] = numeric_columns
            st.write("Preprocessed Data:")
            st.write(final_df)
            if st.button("Proceed to DBSCAN Clustering"):
                st.session_state['page'] = 'DBSCAN Clustering'
                st.experimental_rerun()
        except Exception as e:
            st.error(f"Error during preprocessing: {e}")

# DBSCAN Clustering
if st.session_state['page'] == 'DBSCAN Clustering':
    st.header("DBSCAN Clustering")
    if 'final_df' not in st.session_state:
        st.warning("Please preprocess the data first.")
    else:
        final_df = st.session_state['final_df']
        numeric_columns = st.session_state['numeric_columns']
        eps = st.slider("Select epsilon (eps):", 0.1, 5.0, 0.5)
        min_samples = st.slider("Select minimum samples:", 1, 10, 5)
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
                # Create a GeoDataFrame for plotting
                gdf = gpd.GeoDataFrame(result_df, geometry=gpd.points_from_xy(result_df.longitude, result_df.latitude))
                gdf['cluster_labels'] = labels  # labels from DBSCAN clustering
                
                # Set CRS
                gdf.set_crs('epsg:4326', inplace=True)
                gdf = gdf.to_crs(epsg=3857)  # Project to Web Mercator for visualization purposes
                
                # Plot using geopandas and add a basemap with contextily
                fig, ax = plt.subplots(1, 1, figsize=(20, 15))  # Increased figure size for better layout
                gdf.plot(ax=ax, markersize=5, column='cluster_labels', cmap='tab20', legend=True)
                
                # Distance calculation for label placement
                coords = np.column_stack((gdf.geometry.x, gdf.geometry.y))
                distances = euclidean_distances(coords, coords)
                min_dist = np.percentile(distances[distances > 0], 1)  # Lower percentile if too restrictive
                
                # Add regency names, relaxing the distance condition
                for idx, row in gdf.iterrows():
                    if distances[idx][distances[idx] > 0].min() > min_dist / 2:  # Reduce min_dist to be less restrictive
                        ax.text(row.geometry.x, row.geometry.y, row['regency'], fontsize=8, ha='right', va='top', rotation=15)
                    else:
                        # Optionally add a marker or different text for skipped points
                        ax.text(row.geometry.x, row.geometry.y, '*', fontsize=12, color='red')  # Mark points too close to others
                
                # Add the basemap
                ctx.add_basemap(ax, source=ctx.providers.CartoDB.V
                # Distance calculation for label placement
                coords = np.column_stack((gdf.geometry.x, gdf.geometry.y))
                distances = euclidean_distances(coords, coords)
                min_dist = np.percentile(distances[distances > 0], 1)  # Lower percentile if too restrictive
                
                # Add regency names, relaxing the distance condition
                for idx, row in gdf.iterrows():
                    if distances[idx][distances[idx] > 0].min() > min_dist / 2:  # Reduce min_dist to be less restrictive
                        ax.text(row.geometry.x, row.geometry.y, row['regency'], fontsize=8, ha='right', va='top', rotation=15)
                    else:
                        # Optionally add a marker or different text for skipped points
                        ax.text(row.geometry.x, row.geometry.y, '*', fontsize=12, color='red')  # Mark points too close to others
                
                # Add the basemap
                ctx.add_basemap(ax, source=ctx.providers.CartoDB.Voyager)
                ax.set_title('DBSCAN Clustering of PCA-Reduced Data with Regency Labels')
                ax.set_axis_off()
                st.pyplot(fig)
        except Exception as e:
            st.error(f"Error during DBSCAN clustering: {e}")
