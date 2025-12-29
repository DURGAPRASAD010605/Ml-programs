from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

diabetes = load_diabetes()
x = diabetes.data
y = diabetes.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

knn_regressor = KNeighborsRegressor(n_neighbors=3)
knn_regressor.fit(x_train, y_train)

y_pred = knn_regressor.predict(x_test)

print("mean squared error (mse):", mean_squared_error(y_test, y_pred))
print("R-squared (r2):", r2_score(y_test, y_pred))
