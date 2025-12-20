from pycontree import ConTree
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("datasets/bank.txt", sep=" ", header=None)

X = df[df.columns[1:]]
y = df[0]

max_depth = 2

contree = ConTree(max_depth=max_depth, verbose=True)
cart = DecisionTreeClassifier(max_depth=max_depth)

contree.fit(X, y)
cart.fit(X, y)

contree_ypred = contree.predict(X)
print("ConTree Accuracy: " , accuracy_score(y, contree_ypred))

cart_ypred = cart.predict(X)
print("CART Accuracy: " , accuracy_score(y, cart_ypred))