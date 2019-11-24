import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


df = pd.read_csv('output.csv')
df = df.drop(['state', 'city_or_county', 'participant_age_group'], axis=1)
df = df.dropna()
df = df.sample(n=10000)

print(df.head())

features = df.columns[1:]

scaler = MinMaxScaler().fit(df[features])
x = pd.DataFrame(scaler.transform(df[features]))
x.columns = features


#for link in ['complete', 'average', 'single']:
#    for aff in ['manhattan', 'euclidean']:
#        for n in range(2, 10):
#            est = AgglomerativeClustering(n_clusters=n, linkage=link, affinity=aff)
#            est.fit(x)
#            df['labels'] = est.labels_
#            print('link', link, 'affinity', aff, 'n of clusters', n, 'silhouette', silhouette_score(x, est.labels_))
            
est = AgglomerativeClustering(n_clusters=2, linkage='average', affinity='manhattan')     
est.fit(x)
df['labels'] = est.labels_
print('link', 'average', 'affinity', 'manhattan', 'n of clusters', 2, 'silhouette', silhouette_score(x, est.labels_))
df.head()
#df.to_csv(r'Desktop\Aggl.csv')