# MinMaxScaler (Нормализация)
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# таргет: уровень премии (низкий/средний/высокий)
df6 = pd.DataFrame({
    'Completion_Pct': [10, 25, 45, 50, 75, 85, 95, 100],
    'Experience_Years': [1, 2, 3, 4, 5, 6, 7, 8],
    'Target': ['Low', 'Low', 'Medium', 'Medium', 'Medium', 'High', 'High', 'High']
})
df6['Target'] = df6['Target'].map({'Low': 0, 'Medium': 1,'High':2})

df6[['Completion_Pct']]=MinMaxScaler().fit_transform(df6[['Completion_Pct']])

print(df6)


# таргет: одобрение кредита (да/нет)
df7 = pd.DataFrame({
    'Income_K': [30, 35, 40, 45, 50, 42, 38, 1000],
    'Credit_Score': [600, 620, 640, 610, 650, 630, 615, 800],
    'Target': ['No', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes']
})

df7['Target'] = df6['Target'].map({'No': 0, 'Yes': 1})
df7[['Income_K']]=StandardScaler().fit_transform(df7[['Income_K']])