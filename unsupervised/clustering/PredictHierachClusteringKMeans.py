import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import MeanShift

df = pd.read_csv('data/haberman.data')

original_df = pd.DataFrame.copy(df)
df.drop(['operation_year'], 1, inplace=True)
print(df.head())

# Features. Remove the column we are going to predict
features = np.array(df.drop(['survived'], 1).astype(float))
# Standardise the data set
features = preprocessing.scale(features)

# We don't need to tell how many clusters to classify the data.
classifier = MeanShift()
classifier.fit(features)

labels = classifier.labels_
cluster_centers = classifier.cluster_centers_

original_df['cluster_group'] = np.nan

for i in range(len(features)):
    original_df['cluster_group'].iloc[i] = labels[i]

n_clusters_ = len(np.unique(labels))

survival_rates = {}
for i in range(n_clusters_):
    temp_df = original_df[(original_df['cluster_group'] == float(i))]
    survival_cluster = temp_df[(temp_df['survived']) == 1]
    survival_rate = len(survival_cluster) / len(temp_df)
    survival_rates[i] = survival_rate

print(survival_rates)
