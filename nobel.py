import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier  # للـ ANN
warnings.filterwarnings("ignore")
nobel = pd.read_csv('nobel.csv')
nobel.head()

print(nobel.info())
print(nobel.describe())
print(nobel.isnull().sum())

# Fill missing 'Birth Country' and 'Sex' with mode
nobel['Birth Country'].fillna(nobel['Birth Country'].mode()[0], inplace=True)
nobel['Sex'].fillna(nobel['Sex'].mode()[0], inplace=True)

# Drop columns with too many nulls or irrelevant
nobel.drop(['Death Date', 'Death City', 'Death Country'], axis=1, inplace=True)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
nobel['category_encoded'] = le.fit_transform(nobel['Category'])
nobel['sex_encoded'] = le.fit_transform(nobel['Sex'])
nobel['country_encoded'] = le.fit_transform(nobel['Birth Country'])


X = nobel[['year', 'category_encoded', 'country_encoded', 'sex_encoded']]
nobel['female_winner'] = (nobel['sex'] == 'Female').astype(int)

y = nobel['female_winner'].astype(int)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# Support Vector Machine
svm = SVC()
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)

# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_pred))
print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
print("SVM Accuracy:", accuracy_score(y_test, svm_pred))
print("Decision Tree Accuracy:", accuracy_score(y_test, dt_pred))

# Create result folder if not exist
import os
os.makedirs('Data/Result', exist_ok=True)

# Save predictions
pd.DataFrame({'Actual': y_test, 'LR_Pred': lr_pred}).to_csv('Data/Result/predictions_LR.csv', index=False)
pd.DataFrame({'Actual': y_test, 'RF_Pred': rf_pred}).to_csv('Data/Result/predictions_RF.csv', index=False)
pd.DataFrame({'Actual': y_test, 'SVM_Pred': svm_pred}).to_csv('Data/Result/predictions_SVM.csv', index=False)
pd.DataFrame({'Actual': y_test, 'DT_Pred': dt_pred}).to_csv('Data/Result/predictions_DT.csv', index=False)


# Gender distribution
sns.countplot(data=nobel, x='sex')
plt.title('Gender Distribution of Nobel Laureates')
plt.show()

# Top 10 countries
nobel['birth_country'].value_counts().head(10).plot(kind='barh')
plt.title('Top 10 Countries with Most Nobel Laureates')
plt.show()

nobel['decade'] = (nobel['year'] // 10) * 10
nobel['usa_born_winners'] = nobel['birth_country'] == 'United States of America'

# USA-born winners by decade
sns.lineplot(data=nobel.groupby('decade')['usa_born_winners'].mean().reset_index(), x='decade', y='usa_born_winners')
plt.title('Proportion of USA-born Winners per Decade')
plt.show()


# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# Support Vector Machine
svm = SVC()
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)

# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

# Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
nb_pred = nb.predict(X_test)

# K-Nearest Neighbors
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)

# Linear Regression (مع تحويل القيم إلى 0 أو 1)
linreg = LinearRegression()
linreg.fit(X_train, y_train)
linreg_pred = (linreg.predict(X_test) > 0.5).astype(int)

# Artificial Neural Network
ann = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
ann.fit(X_train, y_train)
ann_pred = ann.predict(X_test)

# إنشاء مجلد لحفظ النتائج
import os
os.makedirs('Data/Result', exist_ok=True)

# حفظ كل التوقعات
pd.DataFrame({'Actual': y_test, 'LR_Pred': lr_pred}).to_csv('Data/Result/predictions_LR.csv', index=False)
pd.DataFrame({'Actual': y_test, 'RF_Pred': rf_pred}).to_csv('Data/Result/predictions_RF.csv', index=False)
pd.DataFrame({'Actual': y_test, 'SVM_Pred': svm_pred}).to_csv('Data/Result/predictions_SVM.csv', index=False)
pd.DataFrame({'Actual': y_test, 'DT_Pred': dt_pred}).to_csv('Data/Result/predictions_DT.csv', index=False)
pd.DataFrame({'Actual': y_test, 'NB_Pred': nb_pred}).to_csv('Data/Result/predictions_NB.csv', index=False)
pd.DataFrame({'Actual': y_test, 'KNN_Pred': knn_pred}).to_csv('Data/Result/predictions_KNN.csv', index=False)
pd.DataFrame({'Actual': y_test, 'LinReg_Pred': linreg_pred}).to_csv('Data/Result/predictions_LinReg.csv', index=False)
pd.DataFrame({'Actual': y_test, 'ANN_Pred': ann_pred}).to_csv('Data/Result/predictions_ANN.csv', index=False)
