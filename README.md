# ==============================
# BUSINESS ANALYTICS PROJECT
# ==============================

# 1️⃣ Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# 2️⃣ Load Dataset
# Download the dataset if not already present
!wget -nc https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx

df = pd.read_excel("Online Retail.xlsx")

print("Dataset Loaded Successfully")
print("Columns in dataset:")
print(df.columns)

# 3️⃣ Basic Cleaning
df = df.dropna(subset=['CustomerID'])
df = df[df['Quantity'] > 0]

df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

print("After Cleaning Shape:", df.shape)

# 4️⃣ Total Revenue
total_revenue = df['TotalPrice'].sum()
print("Total Revenue:", total_revenue)

# 5️⃣ Monthly Revenue Trend
df['Month'] = df['InvoiceDate'].dt.to_period('M')
monthly_revenue = df.groupby('Month')['TotalPrice'].sum()

plt.figure(figsize=(10,5))
monthly_revenue.plot()
plt.title("Monthly Revenue Trend")
plt.show()

# 6️⃣ RFM Analysis
snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalPrice': 'sum'
})

rfm.columns = ['Recency', 'Frequency', 'Monetary']

print("RFM Table Created")
print(rfm.head())

# 7️⃣ Customer Segmentation using KMeans
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

print("Clustering Completed")
print(rfm.head())

# 8️⃣ Visualize Customer Segments
plt.figure(figsize=(8,5))
sns.boxplot(x='Cluster', y='Monetary', data=rfm)
plt.title("Customer Segments by Spending")
plt.show()

# 9️⃣ Simple Predictive Model (Will customer purchase recently?)

df['RecentPurchase'] = np.where(
    df['InvoiceDate'] > df['InvoiceDate'].max() - pd.DateOffset(months=1),
    1,
    0
)

target = df.groupby('CustomerID')['RecentPurchase'].max()
rfm = rfm.merge(target, on='CustomerID')

X = rfm[['Recency', 'Frequency', 'Monetary']]
y = rfm['RecentPurchase']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("ROC AUC Score:")
print(roc_auc_score(y_test, model.predict_proba(X_test)[:,1]))

print("PROJECT COMPLETED SUCCESSFULLY")
