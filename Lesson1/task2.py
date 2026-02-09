import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

df1 = pd.DataFrame({
    'Age': [25, 30, 35, 40, 45, 50],
    'ID System': [np.nan, 102, np.nan, 105, np.nan, 107],
    'Target': ['Yes', 'No', 'No', 'Yes', 'No', 'Yes']
})
# Так как ID должен быть единственным, замена его на "приближённые" значения может вызвать ошибки, поэтому не используются данные среди которых есть отсутствующие данные
df1.dropna(inplace=True)

# Target выполняет суть истенны или лжи, поэтому замена на аналоги 0 и 1
df1['Target'] = df1['Target'].map({'Yes': 1, 'No': 0})
print("first one")
print(df1)


# Таргет: Уровень подписки (Basic < Silver < Gold — с порядком)
df2 = pd.DataFrame({
    'City': ['Moscow', 'Moscow', 'London', 'Moscow', np.nan, 'Moscow', 'London'],
    'Age': [20, 25, 30, 35, 40, 45, 50],
    'Target': ['Basic', 'Basic', 'Silver', 'Silver', 'Gold', 'Gold', 'Gold']
})


# Замена на наиболее популярный город
imputer = SimpleImputer(strategy='most_frequent')
df2['City'] = imputer.fit_transform(df2['City'].values.reshape(-1, 1)).flatten()

"""
ВОПРОС: Дополнительно, как работает SimpleImputer
"""

# Уровни подписки и чем выше тем более значима, соответствующий аналог простое перечисление(как в очереде от меньшего к большему)
df2['Target'] = df2['Target'].map({'Basic': 0, 'Silver': 1, 'Gold': 2})
print("second one")
print(df2)


# Таргет: Группа здоровья (A < B < C — с порядком)
df3 = pd.DataFrame({
    'Pulse': [70, 72, 75, np.nan, 68, 71, 73, 74],
    'Temp': [36.6, 36.7, 36.8, 36.6, 36.9, 36.6, 36.7, 36.8],
    'Target': ['A', 'A', 'B', 'A', 'B', 'A', 'B', 'C']
})


# среднее число без учёта резко высоких значений
imputer = SimpleImputer(strategy='median')
df3['Pulse'] = imputer.fit_transform( df3['Pulse'].values.reshape(-1, 1)).flatten()

df3['Target'] = df3['Target'].map({'A': 0, 'B': 1, 'C': 2})

print("third one")
print(df3)



# Таргет: Прошел проверку безопасности (Да/Нет)
df4 = pd.DataFrame({
    'Days_Since_Last_Incident': [10, 5, 20, np.nan, 15, 30],
    'Risk_Score': [0.1, 0.2, 0.1, 0.4, 0.2, 0.1],
    'Target': ['Safe', 'Safe', 'Warning', 'Safe', 'Safe', 'Warning']
})

# среднее число дней
imputer = SimpleImputer(strategy='mean')
df4['Days_Since_Last_Incident'] = imputer.fit_transform( df4['Days_Since_Last_Incident'].values.reshape(-1, 1)).flatten()

# df4['Days_Since_Last_Incident'] = df4['Days_Since_Last_Incident'].interpolate(method='linear')

df4['Target'] = df4['Target'].map({'Warning': 0, 'Safe': 1})

print("fourth one")
print(df4)


# Таргет: Кредитный рейтинг (Low < High — с порядком)
df5 = pd.DataFrame({
    'Bonus_Points': [100, 500, np.nan, 200, np.nan, 800],
    'Salary_K': [50, 100, 40, 120, 30, 150],
    'Target': ['Low', 'High', 'Low', 'High', 'Low', 'High']
})

# среднее без резкого роста
imputer = SimpleImputer(strategy='median')
df5['Bonus_Points'] = imputer.fit_transform( df5['Bonus_Points'].values.reshape(-1, 1)).flatten()
df5['Target'] = df5['Target'].map({'Low': 0, 'High': 1})

print("fifth one")
print(df5)













