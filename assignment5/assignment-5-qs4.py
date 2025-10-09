import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import r2_score

X = np.arange(1, 11).reshape(-1, 1)
y = np.sin(X).ravel()

model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
model.fit(X, y)
y_pred = model.predict(X)

r2 = r2_score(y, y_pred)
print(f"SVR R² Score: {r2:.2f}")