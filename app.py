import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import QuantileTransformer
import plotly.express as px

# Function to preprocess the data
def preprocess_data(df):
    # Check for missing values and fill them with median
    df.fillna(df.median(), inplace=True)
    
    # Handle outliers by capping them to the 99th percentile
    for column in df.columns:
        upper_limit = df[column].quantile(0.99)
        df[column] = np.where(df[column] > upper_limit, upper_limit, df[column])
    
    # Apply quantile transformation
    transformer = QuantileTransformer(output_distribution='normal', random_state=42)
    df_transformed = transformer.fit_transform(df)
    
    return df_transformed

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
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded Data:")
        st.write(df)

# Preprocess Data
if option == "Preprocess Data":
    if 'df' not in locals():
        st.sidebar.warning("Please upload a CSV file first.")
    else:
        df_transformed = preprocess_data(df)
        st.write("Preprocessed Data:")
        st.write(pd.DataFrame(df_transformed, columns=df.columns))

# DBSCAN Clustering
if option == "DBSCAN Clustering":
    if 'df_transformed' not in locals():
        st.sidebar.warning("Please preprocess the data first.")
    else:
        eps = st.sidebar.slider("Select epsilon (eps):", 0.1, 5.0, 0.5)
        min_samples = st.sidebar.slider("Select minimum samples:", 1, 10, 5)
        data_with_labels, labels = dbscan_clustering(df_transformed, eps, min_samples)
        
        st.write("DBSCAN Clustering Results:")
        st.write(pd.DataFrame(data_with_labels, columns=list(df.columns) + ['Cluster']))
        
        # Plotting
        fig = px.scatter(pd.DataFrame(data_with_labels, columns=list(df.columns) + ['Cluster']), 
                         x=0, y=1, color='Cluster', title="DBSCAN Clustering Results")
        st.plotly_chart(fig)
