import pandas as pd
import numpy as np
df = pd.read_csv('Housing.csv')
df.head(5)
df.info()
df.isnull().sum()
for i in df.columns:
  dis = len(df[i].unique())
  print(f"{i} - {dis}")
df['furnishingstatus'].value_counts()
from sklearn.preprocessing import OrdinalEncoder
enc = OrdinalEncoder()
df['mainroad'] = enc.fit_transform(df[['mainroad']])
df['guestroom'] = enc.fit_transform(df[['guestroom']])
df['basement'] = enc.fit_transform(df[['basement']])
df['hotwaterheating'] = enc.fit_transform(df[['hotwaterheating']])
df['airconditioning'] = enc.fit_transform(df[['airconditioning']])
df['prefarea'] = enc.fit_transform(df[['prefarea']])
df.head()
rank=['unfurnished','semi-furnished','furnished']
oe = OrdinalEncoder(categories=[rank])
df['furnishingstatus']=oe.fit_transform(df[['furnishingstatus']])
df.head()
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
sns.distplot(df['price'], bins=50)
sns.distplot(df['area'], bins=100)
sns.scatterplot(x='price', y='area', hue='mainroad', data=df)

sns.scatterplot(x='price', y='area', hue='guestroom', data=df)
sns.scatterplot(x='price', y='area', hue='basement', data=df)
sns.scatterplot(x='price', y='area', hue='hotwaterheating', data=df)
sns.scatterplot(x='price', y='area', hue='airconditioning', data=df)
sns.pairplot(vars=['area','bedrooms','bathrooms','stories','price'],data=df)
correlation_matrix = df.corr()
plt.figure(figsize=(15, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='rocket_r', linewidths=0.5, fmt='.2f')
X=df.drop(['price'],axis=1)
y=df['price']
X.shape
y.shape
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
from sklearn.preprocessing import MinMaxScaler
scalar=MinMaxScaler()
X_train=scalar.fit_transform(X_train)
X_test=scalar.fit_transform(X_test)
#y_train=scalar.fit_transform(y_train.values.reshape(-1, 1))
#y_test=scalar.fit_transform(y_test.values.reshape(-1, 1))
X_train
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train,y_train)
y_pred = reg.predict(X_test)
y_pred
from sklearn.metrics import r2_score,mean_squared_error
mean_squared_error(y_pred,y_test)
k=r2_score(y_pred,y_test)
k
adj_r2= 1-(1-k)*(545-1)/(545-12-1)
