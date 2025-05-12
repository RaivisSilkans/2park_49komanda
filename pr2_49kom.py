
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# === Palīgfunkcija PDF un CSV saglabāšanai ===
def save_report_csv_pdf(report, name):
    df = pd.DataFrame(report).transpose()
    csv_name = f"{name}_report.csv"
    pdf_name = f"{name}_report.pdf"
    df.to_csv(csv_name)
    print(f"CSV saglabāts: {csv_name}")

    c = canvas.Canvas(pdf_name, pagesize=A4)
    width, height = A4
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, height - 40, f"{name} – Classification Report")
    c.setFont("Helvetica", 10)

    y = height - 70
    for i, (index, row) in enumerate(df.round(3).iterrows()):
        line = f"{index:20} " + " ".join([f"{v:>8}" for v in row])
        c.drawString(40, y, line)
        y -= 15
        if y < 50:
            c.showPage()
            y = height - 40
    c.save()
    print(f"PDF saglabāts: {pdf_name}")

# === Datu ielāde un pirmapstrāde ===
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

# === Vizualizācijas 1. daļai ===
sns.scatterplot(x='Nscore', y='Cscore', hue='Coke_binary', data=df)
plt.title('1. daļa – Nscore vs Cscore by Cocaine Use')
plt.savefig('1a_scatter.png')
plt.show()

sns.histplot(data=df, x='Impulsive', hue='Coke_binary', kde=True)
plt.title('1. daļa – Impulsivitāte pēc klases')
plt.savefig('1b_hist.png')
plt.show()

sns.kdeplot(df['Nscore'], shade=True)
plt.title('1. daļa – Nscore sadalījums')
plt.savefig('1c_kde.png')
plt.show()

# === Statistika ===
for col in features:
    print(f"{col}: Mean = {df[col].mean():.2f}, Variance = {df[col].var():.2f}")

# === 2. daļa – Nepārraudzītā mācīšanās ===
X_unsupervised = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_unsupervised)

linked = linkage(X_scaled, method='ward')
plt.figure(figsize=(12, 6))
dendrogram(linked, truncate_mode='level', p=5)
plt.title('2. daļa – Dendrogramma (Ward metode)')
plt.savefig('2a_dendrogram.png')
plt.show()

silhouette_scores = []
for k in range(2, 7):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    score = silhouette_score(X_scaled, kmeans.labels_)
    silhouette_scores.append((k, score))
    print(f'k={k}, Silhouette score={score:.4f}')

k_vals, scores = zip(*silhouette_scores)
plt.plot(k_vals, scores, marker='o')
plt.title('2. daļa – Silueta koeficients')
plt.xlabel('Klasteru skaits')
plt.ylabel('Vērtība')
plt.grid(True)
plt.savefig('2b_silhouette.png')
plt.show()
# Atrodam labāko k pēc Silueta koeficienta
best_k = max(silhouette_scores, key=lambda x: x[1])[0]
print(f"\nLabākais k pēc Silueta koeficienta: {best_k}")

# Vēlreiz trenē KMeans ar labāko k un veido scatter plot
kmeans_best = KMeans(n_clusters=best_k, random_state=42)
labels_best = kmeans_best.fit_predict(X_scaled)

# Veido scatter plot (pēc pirmām divām pazīmēm)
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=X_scaled[:, 0], y=X_scaled[:, 1],
    hue=labels_best, palette='Set2', s=50, edgecolor='k'
)
plt.title(f'2. daļa – K-vidējo algoritma rezultāts (k={best_k})')
plt.xlabel('Pazīme 1 (Nscore scaled)')
plt.ylabel('Pazīme 2 (Escore scaled)')
plt.legend(title='Klasteri')
plt.savefig('2c_kmeans_scatter_best_k.png')
plt.show()


# Eksperiments 1 – 2 klasteri (augsts grieziens)
plt.figure(figsize=(12, 6))
dendrogram(linked, color_threshold=25, truncate_mode='level', p=5)
plt.title('2. daļa – Dendrogramma (Eksperiments 1 – 2 klasteri)')
plt.savefig('2a_dendrogram_exp1.png')
plt.show()

# Eksperiments 2 – 3 klasteri (vidējs grieziens)
plt.figure(figsize=(12, 6))
dendrogram(linked, color_threshold=15, truncate_mode='level', p=5)
plt.title('2. daļa – Dendrogramma (Eksperiments 2 – 3 klasteri)')
plt.savefig('2a_dendrogram_exp2.png')
plt.show()

# Eksperiments 3 – 4 klasteri (zemāks grieziens)
plt.figure(figsize=(12, 6))
dendrogram(linked, color_threshold=10, truncate_mode='level', p=5)
plt.title('2. daļa – Dendrogramma (Eksperiments 3 – 4 klasteri)')
plt.savefig('2a_dendrogram_exp3.png')
plt.show()

# === 3. daļa – Pārraudzītā mācīšanās ===
X = df[features]
y = df['Coke_binary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

models = {
    'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    'LogisticRegression': LogisticRegression(C=1.0, max_iter=1500, random_state=42),
    'NeuralNetwork': MLPClassifier(hidden_layer_sizes=(50, 20), activation='tanh', max_iter=3000, random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"=== 3. daļa – {name} klasifikācija ===")
    report = classification_report(y_test, y_pred, output_dict=True)
    print(classification_report(y_test, y_pred))
    save_report_csv_pdf(report, name)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'3. daļa – Confusion Matrix: {name}')
    plt.savefig(f"{name}_confusion_matrix.png")
    plt.close()
