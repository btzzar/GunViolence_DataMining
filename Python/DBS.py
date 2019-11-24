import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

dataset = pd.read_csv('output.csv')
dataset = dataset.drop(['state', 'city_or_county', 'participant_age_group'], axis=1)
features = dataset.columns
dataset = dataset.dropna()

scaler = MinMaxScaler().fit(dataset)
x = pd.DataFrame(scaler.transform(dataset))
x.columns = features
x.head()

est = DBSCAN(eps=0.2, min_samples = 2)
est.fit(x)
x['labels'] = est.labels_
print('Eps: ', 0.2, 'Min_samples: ', 2)
print('silhouette_score: ', silhouette_score(x, est.labels_))
br_klas = x['labels'].unique()
print('broj klastera: ', len(br_klas))
print()
x.head()
#x.to_csv(r'Desktop\DBS.csv')