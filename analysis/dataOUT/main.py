
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import plots as g
import pca.pca as pca_module
from sklearn.preprocessing import StandardScaler
import missing_values as lipsa
import seaborn as sb
from sklearn.cluster import KMeans

# ANALIZA IN COMPONENTE PRINCIPALE

date = pd.read_csv('./dataIN/data_2021.csv');
pd.set_option('display.max_columns', None)
print(date);
print(date.columns);

# selectarea coloanelor utilizate pt ACP
varColoane = date.columns[2:];
print(varColoane);

#  lista etichete observatii
numeObs = date.index.values
print(numeObs, type(numeObs))

# creare matrice model ca numpy.ndarray
X = date[varColoane].values;
print(X);

# inlocuire valori lipsa
X = lipsa.inlocuireNAN(X);
print(X);

# standardizarea datelor
scaler = StandardScaler();
X_std = scaler.fit_transform(X);
print(X_std);


# Aplicare ACP
acp = pca_module.ACP(X_std);

valProp = acp.getValProp()
g.componentePrincipale(valoriProprii=valProp)
g.afisare();  #  => vom avea 2 componente principale


# extragerea celor 2 componente principale pastrate
compPrin = acp.getCompPrin()[:,:2];


# Creare nou data frame pentru cele 2 componente alese
acp_df = pd.DataFrame(data=compPrin, columns=['C1', 'C2']);
print(acp_df);

# Vizualizare ACP
plt.figure(figsize = (11,15))
plt.scatter(acp_df.iloc[:, 0], acp_df.iloc[:, 1])
plt.xlabel('Prima componenta principala')
plt.ylabel('A doua componenta principala')


for i, label in enumerate(acp_df.index):
    plt.text(acp_df.iloc[i, 0], acp_df.iloc[i, 1], label)
plt.show();



factorLoadingsACP = acp.getFactorLoadings()[:,:2];
factorLoadingsACP_df = pd.DataFrame(data=factorLoadingsACP, index=varColoane, columns=['C1', 'C2']);
g.corelograma(matrice=factorLoadingsACP_df, titlu='Corelograma Factor Loadings')
g.afisare();

# Salvare in fisier csv
acp_df.to_csv('./dataOUT/pca_results.csv')


# ANALIZA DE CLUSTER

# Selectare criterii pentru analiza cluster
criterii = date[['Exports of goods and services ', 'Imports of goods and services ']]

# Standardizare
criterii_std = scaler.fit_transform(criterii)

# Calcul distorsiuni - suma pătratelor distanțelor dintre fiecare punct de date și centrul clusterului său alocat
distortions = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init= 10 )
    kmeans.fit(criterii_std)
    distortions.append(kmeans.inertia_)

# Plot the elbow graph
plt.plot(range(1, 11), distortions, marker='o')
plt.title('Elbow Method pentru a afla numarul de clustere')
plt.xlabel('Numarul de clustere (k)')
plt.ylabel('Distorsiune')

plt.show()


#n_init -  numar de ori in care algoritmul se repeta cu diferiti centroizi
#random_state - numarul de generari al centroizilor
kmeans = KMeans(n_clusters= 3, n_init=10, random_state = 42)
clustere = kmeans.fit_predict(criterii_std)

date['cluster'] = clustere

# Vizualizare analiza de cluster
plt.figure(figsize=(11, 15))
plt.scatter(date['Exports of goods and services '], date['Imports of goods and services '], c=date['cluster'], cmap='viridis')
plt.title('Analiza de cluster')
plt.xlabel('Exports of goods and services')
plt.ylabel('Imports of goods and services')
for i, label in enumerate(date.index):
    plt.text(date['Exports of goods and services '][i], date['Imports of goods and services '][i], label)
plt.show()

date.to_csv('./dataOUT/cluster_results.csv')






