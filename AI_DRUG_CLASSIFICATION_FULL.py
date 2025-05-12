# -- Nepieciešamās bibliotēkas --
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# === 1. DAĻA – Datu pirmapstrāde un izpēte ===

df = pd.read_csv('drug_consumption.data', header=None)
columns = [
    'ID', 'Age', 'Gender', 'Education', 'Country', 'Ethnicity',
    'Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore', 'Impulsive', 'SS',
    'Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caff', 'Cannabis', 'Choc', 'Coke',
    'Crack', 'Ecstasy', 'Heroin', 'Ketamine', 'Legalh', 'LSD', 'Meth', 'Mushrooms', 'Nicotine', 'Semer', 'VSA'
]
df.columns = columns
df = df.drop('ID', axis=1)
df['Coke_binary'] = df['Coke'].apply(lambda x: 0 if x == 'CL0' else 1)

features = ['Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore', 'Impulsive', 'SS']

# 1. daļas grafiki
sns.scatterplot(x='Nscore', y='Cscore', hue='Coke_binary', data=df)
plt.title('1. daļa – Nscore vs Cscore by Cocaine Use')
plt.savefig('1a_scatter_Nscore_Cscore.png')
plt.show()

sns.histplot(data=df, x='Impulsive', hue='Coke_binary', kde=True)
plt.title('1. daļa – Impulsivitāte pēc klases')
plt.savefig('1b_hist_Impulsivity.png')
plt.show()

sns.kdeplot(df['Nscore'], shade=True)
plt.title('1. daļa – Nscore sadalījums')
plt.savefig('1c_kde_Nscore.png')
plt.show()

# 1. daļas statistika
for col in features:
    print(f"{col}: Mean = {df[col].mean():.2f}, Variance = {df[col].var():.2f}")

# === 2. DAĻA – Nepārraudzītā mašīnmācīšanās ===

X_unsupervised = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_unsupervised)

linked = linkage(X_scaled, method='ward')
plt.figure(figsize=(12, 6))
dendrogram(linked, truncate_mode='level', p=5)
plt.title('2. daļa – Dendrogramma (Ward metode)')
plt.savefig('2a_dendrogram_ward.png')
plt.show()

clusters_2 = fcluster(linked, 2, criterion='maxclust')
clusters_3 = fcluster(linked, 3, criterion='maxclust')
clusters_4 = fcluster(linked, 4, criterion='maxclust')

silhouette_scores = []
for k in range(2, 7):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    score = silhouette_score(X_scaled, kmeans.labels_)
    silhouette_scores.append((k, score))
    print(f'k={k}, Silhouette score={score:.4f}')

k_vals, scores = zip(*silhouette_scores)
plt.plot(k_vals, scores, marker='o')
plt.title('2. daļa – Silueta koeficients dažādiem k')
plt.xlabel('Klasteru skaits')
plt.ylabel('Vērtība')
plt.grid(True)
plt.savefig('2b_silhouette_kmeans.png')
plt.show()

# === 3. DAĻA – Pārraudzītā mašīnmācīšanās ===

X = df[features]
y = df['Coke_binary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

params_rf = [
    {'n_estimators': 50, 'max_depth': 5},
    {'n_estimators': 100, 'max_depth': 10},
    {'n_estimators': 200, 'max_depth': None}
]
for i, p in enumerate(params_rf, 1):
    model = RandomForestClassifier(**p, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_train)
    print(f"\n=== 3. daļa – Random Forest Eksperiments {i} ===")
    print(classification_report(y_train, preds))

best_rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
best_rf.fit(X_train, y_train)
test_preds_rf = best_rf.predict(X_test)
print("\n=== 3. daļa – Testa rezultāti – Random Forest ===")
print(classification_report(y_test, test_preds_rf))

params_lr = [0.01, 1.0, 100.0]
for i, c in enumerate(params_lr, 1):
    model = LogisticRegression(C=c, max_iter=1500, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_train)
    print(f"\n=== 3. daļa – Logistic Regression Eksperiments {i} ===")
    print(classification_report(y_train, preds))

best_lr = LogisticRegression(C=1.0, max_iter=1500, random_state=42)
best_lr.fit(X_train, y_train)
test_preds_lr = best_lr.predict(X_test)
print("\n=== 3. daļa – Testa rezultāti – Logistic Regression ===")
print(classification_report(y_test, test_preds_lr))

params_nn = [
    {'hidden_layer_sizes': (10,), 'activation': 'relu'},
    {'hidden_layer_sizes': (50, 20), 'activation': 'tanh'},
    {'hidden_layer_sizes': (100,), 'activation': 'logistic'}
]
for i, p in enumerate(params_nn, 1):
    model = MLPClassifier(**p, max_iter=3000, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_train)
    print(f"\n=== 3. daļa – Neironu tīkla Eksperiments {i} ===")
    print(classification_report(y_train, preds))

best_nn = MLPClassifier(hidden_layer_sizes=(50, 20), activation='tanh', max_iter=3000, random_state=42)
best_nn.fit(X_train, y_train)
test_preds_nn = best_nn.predict(X_test)
print("\n=== 3. daļa – Testa rezultāti – Neironu tīkls ===")
print(classification_report(y_test, test_preds_nn))
