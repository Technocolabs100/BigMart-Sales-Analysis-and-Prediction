# my script does the following
# 1. Imports relevant libraries.
# 2. Reads 'train.csv' and 'test.csv' files into pandas dataframes.
# 3. Displays the shape and columns of train and test datasets.
# 4. Performs Exploratory Data Analysis by displaying the first few rows of the train dataset.
# 5. Executes Univariate Analysis on the 'Item_Fat_Content' and 'Item_Visibility' columns.
# 6. Performs Bivariate Analysis by calculating the average 'Item_Outlet_Sales' grouped by 'Outlet_Type'. Also creates a box plot for visualization.
# 7. Identifies and fills missing values in the train and test datasets.
# 8. Encodes categorical variables using LabelEncoder and one-hot encoding techniques.
# 9. Prepares data for model training by dropping unnecessary columns and normalizing the data using StandardScaler.
# 10. Trains and tests different predictive models, including Linear Regression, Ridge, Lasso, Random Forest, and XGBoost regressors.


# 1. Import necessary packages
import numpy as np
import pandas as pd

# 2. Load and read the data
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
# 3. Data Structure and Content
print("Shape of train data: ", train_data.shape)
print("Shape of test data: ", test_data.shape)

print("Train data columns: ", train_data.columns)
print("Test data columns: ", test_data.columns)
train_data.info()
test_data.info()
# 4. Exploratory Data Analysis
train_data.head()

# 5. Univariate Analysis
train_data['Item_Fat_Content'].value_counts()
train_data['Item_Visibility'].describe()

# 6. Bivariate Analysis
train_data.groupby('Outlet_Type')['Item_Outlet_Sales'].mean()
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
sns.boxplot(x=train_data['Outlet_Type'], y=train_data['Item_Outlet_Sales'])
plt.show()
# 8. Missing Value Treatment
train_data.isnull().sum()
train_data['Item_Weight'].fillna((train_data['Item_Weight'].mean()), inplace=True)

train_data.isnull().sum()
train_data['Item_Weight'].fillna((train_data['Item_Weight'].mean()), inplace=True)
train_data['Outlet_Size'].fillna('Medium', inplace=True)

test_data.isnull().sum()
test_data['Item_Weight'].fillna((test_data['Item_Weight'].mean()), inplace=True)
test_data['Outlet_Size'].fillna('Medium', inplace=True)


# 10. Encoding Categorical Variables
from sklearn.preprocessing import LabelEncoder

# 11. Label Encoding
le = LabelEncoder()
train_data['Item_Identifier'] = le.fit_transform(train_data['Item_Identifier'])
test_data['Item_Identifier'] = le.fit_transform(test_data['Item_Identifier'])
#
# # 12. One Hot Encoding
train_data = pd.get_dummies(train_data, columns=['Item_Fat_Content', 'Item_Type','Outlet_Identifier','Outlet_Location_Type','Outlet_Size', 'Outlet_Type'])
test_data = pd.get_dummies(test_data, columns=['Item_Fat_Content','Item_Type','Outlet_Identifier', 'Outlet_Location_Type', 'Outlet_Size','Outlet_Type'])

# 13. PreProcessing Data
X_train = train_data.drop([ 'Item_Visibility', 'Item_MRP',
       'Outlet_Establishment_Year', 'Item_Outlet_Sales', 'Item_Fat_Content_LF',
       'Item_Fat_Content_Low Fat', 'Item_Fat_Content_Regular',
       'Item_Fat_Content_low fat', 'Item_Fat_Content_reg',
       'Item_Type_Baking Goods', 'Item_Type_Breads', 'Item_Type_Breakfast',
       'Item_Type_Canned', 'Item_Type_Dairy', 'Item_Type_Frozen Foods',
       'Item_Type_Fruits and Vegetables', 'Item_Type_Hard Drinks',
       'Item_Type_Health and Hygiene', 'Item_Type_Household', 'Item_Type_Meat',
       'Item_Type_Others', 'Item_Type_Seafood', 'Item_Type_Snack Foods',
       'Item_Type_Soft Drinks', 'Item_Type_Starchy Foods',
       'Outlet_Identifier_OUT010', 'Outlet_Identifier_OUT013',
       'Outlet_Identifier_OUT017', 'Outlet_Identifier_OUT018',
       'Outlet_Identifier_OUT019', 'Outlet_Identifier_OUT027',
       'Outlet_Identifier_OUT035', 'Outlet_Identifier_OUT045',
       'Outlet_Identifier_OUT046', 'Outlet_Identifier_OUT049',
       'Outlet_Location_Type_Tier 1', 'Outlet_Location_Type_Tier 2',
       'Outlet_Location_Type_Tier 3', 'Outlet_Size_High', 'Outlet_Size_Medium',
       'Outlet_Size_Small', 'Outlet_Type_Grocery Store',
       'Outlet_Type_Supermarket Type1', 'Outlet_Type_Supermarket Type2',
       'Outlet_Type_Supermarket Type3'], axis=1)
y_train = train_data[['Item_Identifier','Item_Weight']]
X_test = test_data.drop(['Item_Visibility', 'Item_MRP',
       'Outlet_Establishment_Year', 'Item_Fat_Content_LF',
       'Item_Fat_Content_Low Fat', 'Item_Fat_Content_Regular',
       'Item_Fat_Content_low fat', 'Item_Fat_Content_reg',
       'Item_Type_Baking Goods', 'Item_Type_Breads', 'Item_Type_Breakfast',
       'Item_Type_Canned', 'Item_Type_Dairy', 'Item_Type_Frozen Foods',
       'Item_Type_Fruits and Vegetables', 'Item_Type_Hard Drinks',
       'Item_Type_Health and Hygiene', 'Item_Type_Household', 'Item_Type_Meat',
       'Item_Type_Others', 'Item_Type_Seafood', 'Item_Type_Snack Foods',
       'Item_Type_Soft Drinks', 'Item_Type_Starchy Foods',
       'Outlet_Identifier_OUT010', 'Outlet_Identifier_OUT013',
       'Outlet_Identifier_OUT017', 'Outlet_Identifier_OUT018',
       'Outlet_Identifier_OUT019', 'Outlet_Identifier_OUT027',
       'Outlet_Identifier_OUT035', 'Outlet_Identifier_OUT045',
       'Outlet_Identifier_OUT046', 'Outlet_Identifier_OUT049',
       'Outlet_Location_Type_Tier 1', 'Outlet_Location_Type_Tier 2',
       'Outlet_Location_Type_Tier 3', 'Outlet_Size_High', 'Outlet_Size_Medium',
       'Outlet_Size_Small', 'Outlet_Type_Grocery Store',
       'Outlet_Type_Supermarket Type1', 'Outlet_Type_Supermarket Type2',
       'Outlet_Type_Supermarket Type3'], axis=1)

print("X_train columns:", X_train.columns)
print("X_test columns:", X_test.columns)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 14. Modeling
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# 15. Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

# 16. Regularized Linear Regression
ridge = Ridge()
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)

lasso = Lasso()
lasso.fit(X_train, y_train)
lasso_pred = lasso.predict(X_test)

# 17. RandomForest
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# 18. XGBoost
xg_reg = xgb.XGBRegressor(objective ='reg:squarederror')
xg_reg.fit(X_train,y_train)
xgb_pred = xg_reg.predict(X_test)
# 19. Summary
output = pd.DataFrame({'Item_Identifier': pd.Series(test_data.Item_Identifier.values.ravel()),
                       'Item_Weight1': pd.Series(test_data.Item_Weight.values.ravel()),
                       'Item_Weight2': pd.Series(lr_pred.ravel()),
                       'Item_Weight3': pd.Series(ridge_pred.ravel())})



sns.pairplot(output)
plt.show()
output.head()
