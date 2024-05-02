from sklearn.cluster import KMeans
import numpy as np
import pandas as pd


def under_sample_kmeans_df(df, k=None, label_column='label', embedding_column='vector'):
    if k == None:
        k = min(df[label_column].value_counts())
    sample_df_list = []
    for label in df[label_column].unique():
        label_df = df[df[label_column] == label].copy()
        arr = np.array(label_df[embedding_column].to_list())
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(arr)
        centroids = kmeans.cluster_centers_
        labels = kmeans.labels_
        label_df['cluster'] = labels
        sample_df_list.append(label_df.groupby('cluster').sample(1))
    return pd.concat(sample_df_list)



# def under_sample_kmeans(examples, k=None, label_column='label', embedding_column='vector'):
#     if hasattr(examples, label_column)
#     if k == None:
#         k = min(df[label_column].value_counts())
#     sample_df_list = []
#     for label in df[label_column].unique():
#         label_df = df[df[label_column] == label].copy()
#         arr = np.array(label_df[embedding_column].to_list())
#         kmeans = KMeans(n_clusters=k)
#         kmeans.fit(arr)
#         centroids = kmeans.cluster_centers_
#         labels = kmeans.labels_
#         label_df['cluster'] = labels
#         sample_df_list.append(label_df.groupby('cluster').sample(1))
#     return pd.concat(sample_df_list)

