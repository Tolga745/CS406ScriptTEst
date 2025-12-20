from pycontree import ConTree
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import time


df = pd.read_csv("datasets/occupancy.txt", sep=" ", header=None)
X = df[df.columns[1:]].values
y = df[0].values

# In this example, we purposefully set the test size to 80%.
# The small training set increases the likelihood of overfitting, so the gridsearch selects a smaller tree
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

##################################################################
##### 1. Tune using GridSearchCV from sklearn ####################
##################################################################

model = ConTree()

params = {'max_depth': list(range(2,5))}

gs_knn = GridSearchCV(model,
                      param_grid=params,
                      scoring='accuracy',
                      cv=5,
                      n_jobs=1,
                      verbose=3)
start = time.perf_counter()
gs_knn.fit(X_train, y_train)
gs_duration = time.perf_counter() - start

print(f"\nSklearn gridsearch finished in {gs_duration} seconds")
print("Best params from grid search: ", gs_knn.best_params_)

yhat = gs_knn.predict(X_test)

accuracy = accuracy_score(y_test, yhat)
print(f"Test Accuracy Score: {accuracy * 100}%")
