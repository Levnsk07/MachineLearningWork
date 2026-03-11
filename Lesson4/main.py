import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

"""
# Загружаем данные
data =  fetch_california_housing() # хранит в себе куча параметров (данные, цель, признаки)
df = pd.DataFrame(data.data,columns=data.feature_names) # новый словарь с данными с признакаи из данных
df['MedalHouseValue'] = data.target # добовление столбца -- цены (их целей)
"""

df = pd.read_csv('final.csv')

X = (df.drop('price', axis=1)).values
y = df['price'].values.reshape(-1,1)


"""
# Подготовка целей и данных
X = df.drop('MedalHouseValue', axis=1).values # drop удаляет столбец и возращает НОВЫЙ dataFrame
y = df['MedalHouseValue'].values.reshape(-1,1) # reshape делает из данных двумерный массив
"""







# подготовка данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) # Разбиение на обучающую и на тестовую часть

# Масштабирование
scaler = StandardScaler()   # Масштабирование не обходимо для избавления от доменантных признаков, например 0-10000 а другая колонка в тех-же данных 0-1
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32) # Тензоры это один из способов многомерного массива оптемезированного для векторов и скаляров
X_test  = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test  = torch.tensor(y_test, dtype=torch.float32)

# print(X_train[1].shape)
model = nn.Linear(X_train.shape[1], 1) # создаётся модель (слой) с кол. входных данных и кол. выходных (соответственно)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01) # Адаптивный оптимизатор, меняет "скорость" обучения в зависимости от данных
criterion = nn.MSELoss() # Функция потерь для подсчёта среднеквадратичной ошибки (MSE)


# ========================
epochs = 10**20
history = []
for epoch in range(epochs):

    # Прямой проход -- вычисление предсказаний
    y_pred = model(X_train) # делает предсказание для всего массива
    loss = criterion(y_pred, y_train) # вычесляет полученную ошибку

    # Обратных проход -- вычесление градиентов
    optimizer.zero_grad() # Во время обучения накапливаются параметры с предыдущих итераций, поэтому их обнуляют
    loss.backward() # вычесление градиентов показывающие как изменить веса для уменьшения веса
    optimizer.step() # Обновление весов для модели


    # Сохранение и вывод процесса
    history.append(loss.item())
    if (epoch+1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")





# Вывод анализа
plt.figure(figsize=(15, 5))

# График обучения
plt.subplot(1, 2, 1)
plt.plot(history)
plt.title("Процесс обучения (MSE)")
plt.xlabel("Эпоха")

# Анализ остатков
with torch.no_grad():
    test_preds = model(X_test).numpy()
    residuals = y_test - test_preds

plt.subplot(1, 2, 2)
plt.scatter(test_preds, residuals, alpha=0.3, color='teal')
plt.axhline(0, color='red', linestyle='--')
plt.title("Анализ остатков (Residual Analysis)")
plt.xlabel("Предсказанная цена")
plt.ylabel("Ошибка")
plt.show()

# Корреляционная матрица для EDA
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Корреляция признаков")
plt.show()


