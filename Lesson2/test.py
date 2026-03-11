import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('house_price_regression_dataset.csv')
model = LinearRegression()

X = np.array(df[["Square_Footage"]])
y = np.array(df["House_Price"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=50)

# Обучаем
model.fit(X_train, y_train)
# Делаем предсказание
y_pred = model.predict(X_test)

import matplotlib.pyplot as plt

# Построение
plt.plot(X_test, y_pred, color='red')
# plt.plot(X_test, y_test)
# plt.scatter(X_test, y_test, marker='o')
# plt.scatter(X_test, y_pred,color='red', marker='o')


plt.grid(True)
plt.show()

for i in range(0, len(y_pred)):
    print(X_test[i], " -- ", y_test[i], " -- ", y_pred[i])

'''
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# MSE (Среднеквадратичная ошибка): чем ближе к 0, тем лучше
mse = mean_squared_error(y_test, y_pred)
# RMSE (Корень из MSE): ошибка в тех же единицах, что и целевая переменная
# (например, в рублях или метрах)
rmse = mean_squared_error(y_test, y_pred)
# MAE (Средняя абсолютная ошибка): среднее отклонение
mae = mean_absolute_error(y_test, y_pred)
# R^2 (Коэффициент детерминации): точность от 0 до 1
r2 = r2_score(y_test, y_pred)
print(r2, mae, rmse)
'''
