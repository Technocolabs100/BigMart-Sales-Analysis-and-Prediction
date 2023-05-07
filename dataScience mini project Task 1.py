#!/usr/bin/env python
# coding: utf-8

# In[69]:


import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;
import seaborn as sns;




# In[2]:


df= pd.read_csv('test.csv')
df1= pd.read_csv('train.csv')


# In[3]:


df.shape


# In[4]:


df1.shape


# In[5]:


print(df.shape)
print(df1.shape)

print(df.info())
print(df1.info())

print(df.isna().sum())
print(df1.isna().sum())


# In[6]:



print(df.describe())
print(df1.describe())


# In[7]:


df.dtypes


# In[8]:


df1.dtypes


# In[9]:


df[['Item_Weight', 'Item_Visibility', 'Item_MRP']] = df[['Item_Weight', 'Item_Visibility', 'Item_MRP']].astype(float)


# In[10]:


df1[['Item_Weight', 'Item_Visibility', 'Item_MRP']] = df[['Item_Weight', 'Item_Visibility', 'Item_MRP']].astype(float)


# In[11]:



print(df.isnull().sum())


# In[12]:



print(df1.isnull().sum())


# In[13]:


categorical_cols = list(df.select_dtypes(include=['object', 'category']).columns)
print('Categorical columns:', categorical_cols)


numeric_cols = list(df.select_dtypes(include=['int', 'float']).columns)
print('Numeric columns:', numeric_cols)


# In[14]:


categorical_cols = list(df1.select_dtypes(include=['object', 'category']).columns)
print('Categorical columns:', categorical_cols)

# Identify numeric columns
numeric_cols = list(df1.select_dtypes(include=['int', 'float']).columns)
print('Numeric columns:', numeric_cols)


# In[15]:


numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

numeric_cols = df1.select_dtypes(include=np.number).columns.tolist()
df1[numeric_cols] = df1[numeric_cols].fillna(df1[numeric_cols].median())

categorical_cols = df1.select_dtypes(exclude=np.number).columns.tolist()
df1[categorical_cols] = df1[categorical_cols].fillna(df1[categorical_cols].mode().iloc[0])


# In[16]:


print(df.isnull(),df1.isnull().sum())


# In[17]:


duplicates_df = df[df.duplicated()]
print(duplicates_df)


# In[18]:


duplicates_df1 = df1[df1.duplicated()]
print(duplicates_df1)


# In[19]:


import matplotlib.pyplot as plt

# Select only numeric columns
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
numeric_cols = df1.select_dtypes(include=np.number).columns.tolist()

# Create boxplots for each numeric column
for col in numeric_cols:
    plt.figure()
    df.boxplot(column=col)
    df1.boxplot(column=col)
    plt.title(col)
    plt.show()


# In[20]:


df.corr()


# In[21]:


df1.corr()


# In[23]:


from sklearn.preprocessing import StandardScaler


numerical_cols = df.select_dtypes(include=np.number).columns.tolist()

scaler = StandardScaler()


df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

numerical_cols = df1.select_dtypes(include=np.number).columns.tolist()
df1[numerical_cols] = scaler.fit_transform(df1[numerical_cols])


# In[24]:


df.head()


# In[25]:


df1.head()


# In[26]:


# select categorical columns
cat_cols = df.select_dtypes(include=['object']).columns.tolist()

# create dummy variables for each categorical column
for col in cat_cols:
    dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
    df = pd.concat([df, dummies], axis=1)
    
# drop the original categorical columns
df.drop(cat_cols, axis=1, inplace=True)


# In[27]:


df.head()


# In[28]:


df1 = pd.get_dummies(df1)


# In[29]:


df1.head()


# In[30]:


cat_cols = df.select_dtypes(include=['object']).columns
print(cat_cols)


# In[31]:


cat_cols = df1.select_dtypes(include=['object']).columns
print(cat_cols)


# In[42]:


df.describe()


# In[45]:


num_cols = ['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Establishment_Year']

# create a scatter matrix
pd.plotting.scatter_matrix(df[num_cols], figsize=(10,10))
plt.show()


# In[46]:


import matplotlib.pyplot as plt

plt.scatter(df['Item_Weight'], df['Item_MRP'])

# Add labels and title
plt.xlabel('Item Weight')
plt.ylabel('Item MRP')
plt.title('Scatter Plot of Item Weight vs. Item MRP')

plt.show()


# In[48]:


plt.scatter(df1['Item_Weight'], df1['Item_Outlet_Sales'])

# Add axis labels and a title
plt.xlabel('Item Weight')
plt.ylabel('Item Outlet Sales')
plt.title('Relationship between Item Weight and Outlet Sales')

# Display the plot
plt.show()


# In[50]:


df.isna().sum()


# In[51]:


df.isna().sum()


# In[57]:


print(df.columns)


# In[58]:


print(df1.columns)


# In[62]:


from sklearn import preprocessing
  
.
label_encoder = preprocessing.LabelEncoder()
  

df['Item_Weight']= label_encoder.fit_transform(df['Item_Weight'])
  
df['Item_Weight'].unique()


# In[72]:


# Select the features and target variable
X = df[['Item_Weight', 'Item_MRP']]
y = df['Item_Outlet_Sales']

# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the linear regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict the target variable on the test set
y_pred = regressor.predict(X_test)

# Evaluate the performance of the model
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R^2 Score:", r2)


# In[73]:


print(df.columns)


# In[77]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the data

# Select the relevant features
X = df[['Item_MRP', 'Outlet_Establishment_Year', 'Outlet_Size_Medium']]
y = df['Outlet_Establishment_Year']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model on the testing set
score = model.score(X_test, y_test)
print(f'R^2 score: {score:.2f}')


# In[78]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Select the relevant features
X = df.drop('Outlet_Identifier', axis=1)
y = df['Outlet_Identifier']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the random forest classifier with 100 trees
rfc = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model on the training data
rfc.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = rfc.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)


# In[ ]:




