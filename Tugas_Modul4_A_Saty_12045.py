import streamlit as st
import pandas as pd
import os
import plotly.express as px
import numpy as np
from sklearn.metrics import pairwise_distances
import plotly.graph_objects as go
import pickle

def scatter(model, model_name, data, new_point, features, color_scale, title):
    clusters = model.fit_predict(data[features])
    data[f'{model_name}_Cluster'] = clusters

    if model_name == "KMeans_model":
        new_cluster = model.predict(new_point[features])[0]
    else:
        distances = pairwise_distances(new_point[features], data[features])
        nearest_index = distances.argmin()
        new_cluster = clusters[nearest_index]

    fig = px.scatter_3d(data, x='Avg_Credit_Limit', y='Total_Credit_Cards', z='Total_visits_online',
                        color=f'{model_name}_Cluster', title=title, color_continuous_scale=color_scale)

    fig.add_trace(
        go.Scatter3d(
            x=new_point['Avg_Credit_Limit'],
            y=new_point['Total_Credit_Cards'],
            z=new_point['Total_visits_online'],
            mode='markers',
            marker=dict(size=10, color='red'),
            name='New point'
        )
    )
    return fig, new_cluster

st.set_page_config(
    page_title="12045 - Unsupervised Learning",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

if uploaded_file is not None:
    input_data = pd.read_csv(uploaded_file)
    st.markdown("<h1 style='text-align: center;'>Unsupervised Learning - Saty</h1>", unsafe_allow_html=True)
    st.dataframe(input_data)

model_paths = {
    "AGG_model": 'AGG_model.pkl',
    "KMeans_model":'KMeans_model.pkl',
    "DBSCAN_model":'DBSCAN_model.pkl',
}

models = {}

for model_name, path in model_paths.items():
    if os.path.exists(path):
        with open(path, 'rb') as f:
            models[model_name] = pickle.load(f)
    else:
        st.write(f"Model {model_name} tidak ditemukan di path: ", path)

# Sidebar untuk memasukkan (meng-inputkan) nilai untuk titik baru yang akan diprediksi clusternya
avg_CL = st.sidebar.number_input("Average Credit Limit", 0, 1000000)
sum_CC = st.sidebar.number_input("Total Credit Cards", 0, 10)
sum_VO = st.sidebar.number_input("Total Visits Online", 0, 16)

if st.sidebar.button("Prediksi !!!"):
    # Data yang dibutuhkan untuk memprediksi
    features = ['Avg_Credit_Limit', 'Total_Credit_Cards', 'Total_visits_online']
    new_point = pd.DataFrame({
        "Avg_Credit_Limit": [avg_CL],
        "Total_Credit_Cards": [sum_CC],
        "Total_visits_online": [sum_VO]
    })

    # Model clustering yang digunakan dan warna grafik scatternya
    cluster_method = [
        ("KMeans_model", models["KMeans_model"], "KMeans Clustering", px.colors.sequential.Cividis),
        ("AGG_model", models["AGG_model"], "Agglomerative Clustering", px.colors.sequential.Mint),
        ("DBSCAN_model", models["DBSCAN_model"], "DBSCAN Clustering", px.colors.sequential.Plasma),
    ]

    # Membuat tiga kolom untuk menampilkan grafik
    col1, col2, col3 = st.columns(3)
    cols = [col1, col2, col3]

    for (model_name, model, title, color_scale), col in zip(cluster_method, cols):
        fig, new_cluster = scatter(model, model_name, input_data, new_point, features, color_scale, title)
        with col:
            st.plotly_chart(fig)
            st.markdown(f"<p style='text-align: center;'>Titik Dari Data yang baru masuk ke dalam cluster : {new_cluster}</p>", unsafe_allow_html=True)
