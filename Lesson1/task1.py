import random

import numpy as np
from sklearn.model_selection import train_test_split


X = np.array( [ [random.randint(10,100),random.randint(10,100),] for i in range(40) ] )
y = np.array([1 if x[0] >x[1] else 0 for x in X])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

