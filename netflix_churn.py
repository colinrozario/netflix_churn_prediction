#Netflix churn prediction 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

#Loading dataset
df = pd.read_csv(r"C:\Users\colin\OneDrive\Desktop\AI-ML Projects\netflix_churn\netflix_customer_churn.csv")

#Dataset info display
print("First few data samples",df.head())
print("Dataset information",df.info())

#Encoding categorical variables
le = LabelEncoder()
df['churned'] = le.fit_transform(df["churned"])

#Defining features and targets
X = df.drop(columns=['churned'])
y = df['churned']

#Encode all object types columnns in x
for col in X.select_dtypes(include='object').columns:
    X[col] = le.fit_transform(X[col])

#Scaling features
scaler = StandardScaler()
X = scaler.fit_transform(X)


#Splitting dataset
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=43)

#Training logistic regression model
log_model = LogisticRegression(max_iter=200)
log_model.fit(X_train,y_train)

#Training kNN model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train,y_train)

#Evaluate models
log_pred = log_model.predict(X_test)
knn_pred = knn_model.predict(X_test)

print("Logistic Regression classification report\n")
print(classification_report(y_test,log_pred))

print('k-NN Classification Report:')
print(classification_report(y_test,knn_pred))

#Confusion Matrix
print('Confusion matrix of Logistic Regression')
print(confusion_matrix(y_test,log_pred))

print('\nConfusion matrix of k-NN')
print(confusion_matrix(y_test,knn_pred))

#Visual Confusion Matrix
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

#Logistic Regression Confusion Matrix
cm_log = confusion_matrix(y_test, log_pred)
sns.heatmap(cm_log, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=True)
axes[0].set_title('Confusion Matrix - Logistic Regression', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Predicted Label')
axes[0].set_ylabel('True Label')

#k-NN Confusion Matrix
cm_knn = confusion_matrix(y_test, knn_pred)
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Greens', ax=axes[1], cbar=True)
axes[1].set_title('Confusion Matrix - k-NN', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Predicted Label')
axes[1].set_ylabel('True Label')

plt.tight_layout()
plt.show()

