import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import preprocessing

df = pd.read_csv('data/haberman.data')

# year of the operation may not be relevant
df.drop(['operation_year'], 1, inplace=True)
print(df.head())

# Features. Remove the column we are going to predict
features = np.array(df.drop(['survived'], 1).astype(float))
# Standardise the data set
features = preprocessing.scale(features)

# We have to tell how many clusters to classify the data.
# In this case we need two (Survived or Not more than 5 years)
classifier = KMeans(n_clusters=2)
classifier.fit(features)

# Label. we use it compare if we predicted right or wrong
label = np.array(df['survived'])

correctPredictions = 0
for i in range(len(features)):
    row = np.array(features[i].astype(float))
    row = row.reshape(-1, len(row))
    row = classifier.predict(row)
    if row[0] == label[i]:
        correctPredictions += 1

print(correctPredictions / len(features))
