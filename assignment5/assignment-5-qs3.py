import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

X = np.arange(1, 11).reshape(-1, 1)
y = 3 * X.squeeze()**2 + 2 * X.squeeze() + 1

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)
y_pred = model.predict(X_poly)

r2 = r2_score(y, y_pred)
print(f"Polynomial Regression R² Score: {r2:.2f}")
