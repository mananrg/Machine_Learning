# Importing the Libraries and the dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('Data.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Taking Care of the missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X[:, 1:3] = imputer.fit_transform(X[:, 1:3])

# Encoding the independent variable (Country)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Encoding the dependent variable (Purchased)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

# Splitting the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 3:]=sc.fit_transform(X_train[:, 3:])
X_test[:, 3:]=sc.transform(X_test[:, 3:])
print("X_train")
print(X_train)
print("")
print("X_test")
print(X_test)