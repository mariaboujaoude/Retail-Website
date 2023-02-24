import streamlit as st
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import random

# Set page configuration
st.set_page_config(page_title="Retail Store", page_icon=":tada", layout="wide")

# Page header
st.subheader("Retail Store")
st.title("Welcome to our retail store")

# Load product images
@st.cache_data
def get_data():
    df = pd.read_csv('./product_images.csv')
    data_array = df.to_numpy()  
    return data_array

# Generate the different labels (clusters)
@st.cache_data
def get_labels(df):
    # Dimensionality reduction with PCA
    pca = PCA(n_components=9)
    pca_9 = pca.fit_transform(df)

    # Clustering with K-means
    kmeans = KMeans(n_clusters=7)
    kmeans.fit(pca_9)
    yy_kmeans = kmeans.predict(pca_9)
    return yy_kmeans

# Get the cluster of an image
@st.cache_data
def get_clusters():
    cluster_dict = {}
    for label, (idx, image_path) in zip(yy_kmeans, enumerate(data_array)):
        if label not in cluster_dict:
            cluster_dict[label] = [idx]
        else:
            cluster_dict[label].append(idx)
    return cluster_dict

data_array=get_data()
yy_kmeans=get_labels(data_array)
cluster_dict=get_clusters()

n_display=50


# Generate a random button that generates a random product and lists 9 items from the same cluster

if(st.button("Generate random product")):
    generated_idx = random.randint(1, len(data_array))
    generated_label=yy_kmeans[generated_idx]
    
    st.write("random product")
    st.image(data_array[generated_idx].reshape(28,28), width=400)
    st.write("other products in the same cluster")

    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(6,6))
    for idx, ax in zip(cluster_dict[generated_label][:9], axs.ravel()):
        image_data = data_array[idx].reshape(28,28)
        ax.imshow(image_data, cmap='gray')
        ax.axis('off')
    st.pyplot(fig)



