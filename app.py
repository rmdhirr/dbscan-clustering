import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import QuantileTransformer
import plotly.express as px

# Function to preprocess the data
def preprocess_data(df):
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Check for missing values and fill them with median
    numeric_df = numeric_df.apply(lambda x: x.fillna(x.median()), axis=0)
    
    # Handle outliers by capping them to the 99th percentile
    for column in numeric_df.columns:
        upper_limit = numeric_df[column].quantile(0.99)
        numeric_df[column] = np.where(numeric_df[column] > upper_limit, upper_limit, numeric_df[column])
    
    # Apply quantile transformation
    transformer = QuantileTransformer(output_distribution='normal', random_state=42)
    df_transformed = transformer.fit_transform(numeric_df)
    
    return df_transformed, numeric_df.columns

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
        st.session_state['df'] = df
        st.write("Uploaded Data:")
        st.write(df)

# Preprocess Data
if option == "Preprocess Data":
    if 'df' not in st.session_state:
        st.sidebar.warning("Please upload a CSV file first.")
    else:
        df = st.session_state['df']
        df_transformed, columns = preprocess_data(df)
        st.session_state['df_transformed'] = df_transformed
        st.session_state['columns'] = columns
        st.write("Preprocessed Data:")
        st.write(pd.DataFrame(df_transformed, columns=columns))

# DBSCAN Clustering
if option == "DBSCAN Clustering":
    if 'df_transformed' not in st.session_state:
        st.sidebar.warning("Please preprocess the data first.")
    else:
        df_transformed = st.session_state['df_transformed']
        columns = st.session_state['columns']
        eps = st.sidebar.slider("Select epsilon (eps):", 0.1, 5.0, 0.5)
        min_samples = st.sidebar.slider("Select minimum samples:", 1, 10, 5)
        data_with_labels, labels = dbscan_clustering(df_transformed, eps, min_samples)
        
        result_df = pd.DataFrame(data_with_labels, columns=columns)
        result_df['Cluster'] = labels
        
        st.write("DBSCAN Clustering Results:")
        st.write(result_df)
        
        # Plotting
        fig = px.scatter(result_df, x=result_df.columns[0], y=result_df.columns[1], color='Cluster', title="DBSCAN Clustering Results")
        st.plotly_chart(fig)
