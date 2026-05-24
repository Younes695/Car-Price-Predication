import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv('data/car data.csv')

# First 5 Rows
print(df.head())
# Statistical
print(df.describe())
# Missing Vals
print(df.isnull().sum())
# Informations
print(df.info())

# Checking the Distribution of Categorical Data
print(df.Fuel_Type.value_counts())
print(df.Seller_Type.value_counts())
print(df.Transmission.value_counts())

# Encoding the Categorical Data
df.replace({'Fuel_Type': {'Petrol': 0, 'Diesel': 1, 'CNG': 2}}, inplace=True)
df.replace({'Seller_Type': {'Dealer': 0, 'Individual': 1}}, inplace=True)
df.replace({'Transmission': {'Manual': 0, 'Automatic': 1}}, inplace=True)

# Splitting Data & Target
X = df.drop(['Car_Name', 'Selling_Price'], axis=1)
y = df['Selling_Price']

# Splitting Data into Training & Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=2)

# Model Training
## Linear Regression
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
### Model Evaluation
y_pred_lin = lin_model.predict(X_test)
r2_lin = r2_score(y_test, y_pred_lin)
mse_lin = mean_squared_error(y_test, y_pred_lin)

print(f'Linear Regression R2 Score: {r2_lin:.2f}')
print(f'Linear Regression MSE: {mse_lin:.2f}')

#### Visualize the Actual Price vs the Predicted Price
plt.scatter(y_test, y_pred_lin)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color='red', linewidth=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual Price vs Predicted Price (Linear Regression)')
plt.show()

## Lasso Regression
las_model = Lasso()
las_model.fit(X_train, y_train)
### Model Evaluation
y_pred_las = las_model.predict(X_test)
r2_las = r2_score(y_test, y_pred_las)
mse_las = mean_squared_error(y_test, y_pred_las)

print(f'Lasso Regression R2 Score: {r2_las:.2f}')
print(f'Lasso Regression MSE: {mse_las:.2f}')

#### Visualize the Actual Price vs the Predicted Price
plt.scatter(y_test, y_pred_las)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color='red', linewidth=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual Price vs Predicted Price (Lasso Regression)')
plt.show()

## Polynomial Regression
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
### Model Evaluation
y_pred_poly = poly_model.predict(X_test_poly)
r2_poly = r2_score(y_test, y_pred_poly)
mse_poly = mean_squared_error(y_test, y_pred_poly)

print(f'Polynomial Regression R2 Score: {r2_poly:.2f}')
print(f'Polynomial Regression MSE: {mse_poly:.2f}')


#### Visualize the Actual Price vs the Predicted Price
plt.scatter(y_test, y_pred_poly)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color='red', linewidth=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual Price vs Predicted Price (Polynomial Regression)')
plt.show()

## Ridge Regression
ridge_model = Ridge()
ridge_model.fit(X_train, y_train)
### Model Evaluation
y_pred_ridge = ridge_model.predict(X_test)
r2_ridge = r2_score(y_test, y_pred_ridge)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)

print(f'Ridge Regression R2 Score: {r2_ridge:.2f}')
print(f'Ridge Regression MSE: {mse_ridge:.2f}')


#### Visualize the Actual Price vs the Predicted Price
plt.scatter(y_test, y_pred_ridge)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color='red', linewidth=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual Price vs Predicted Price (Ridge Regression)')
plt.show()