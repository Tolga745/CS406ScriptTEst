import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from pycontree import ConTree
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import time
try:
    from pystreed import STreeDClassifier
    pystreed_installed = True
except:
    pystreed_installed = False
    STreeDClassifier = None

datasets = ["bank", "wilt", "bidding"]
results = []

for dataset in datasets:

    for max_depth in range(2, 5):

        df = pd.read_csv(f"train-datasets/{dataset}.txt", sep=" ", header=None)
        
        X = df[df.columns[1:]]
        y = df[0]

        for i in range(1, 11):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42 + i)
        
            contree = ConTree(max_depth=max_depth)
            start = time.perf_counter()
            contree.fit(X_train, y_train)
            contree_duration = start = time.perf_counter() - start
            contree_train_ypred = contree.predict(X_train)
            contree_train_accuracy = accuracy_score(y_train, contree_train_ypred)
            contree_test_ypred = contree.predict(X_test)
            contree_test_accuracy = accuracy_score(y_test, contree_test_ypred)
            print(f"{dataset.capitalize():10s} |  Run {i:2d} | ConTree   d={max_depth}  |  run time: {contree_duration:6.3f}  | train accuracy: {contree_train_accuracy*100:4.2f} | test accuracy: {contree_test_accuracy*100:4.2f}")
            results.append({"Method": "ConTree", "Dataset": dataset.capitalize(), "Run": i, "Depth": max_depth, "Runtime": contree_duration, "Train Accuracy": contree_train_accuracy*100, "Test Accuracy": contree_test_accuracy*100})
            
            cart = DecisionTreeClassifier(max_depth=max_depth)
            start = time.perf_counter()
            cart.fit(X_train, y_train)
            cart_duration = start = time.perf_counter() - start
            cart_train_ypred = cart.predict(X_train)
            cart_train_accuracy = accuracy_score(y_train, cart_train_ypred)
            cart_test_ypred = cart.predict(X_test)
            cart_test_accuracy = accuracy_score(y_test, cart_test_ypred)
            print(f"{dataset.capitalize():10s} |  Run {i:2d} | CART      d={max_depth}  |  run time: {cart_duration:6.3f}  | train accuracy: {cart_train_accuracy*100:4.2f} | test accuracy: {cart_test_accuracy*100:4.2f}")
            results.append({"Method": "CART", "Dataset": dataset.capitalize(), "Run": i, "Depth": max_depth, "Runtime": cart_duration, "Train Accuracy": cart_train_accuracy*100, "Test Accuracy": cart_test_accuracy*100})

            if pystreed_installed:
                streed = STreeDClassifier(max_depth=max_depth, n_thresholds=10)
                start = time.perf_counter()
                streed.fit(X_train, y_train)
                streed_duration = start = time.perf_counter() - start
                streed_train_ypred = streed.predict(X_train)
                streed_train_accuracy = accuracy_score(y_train, streed_train_ypred)
                streed_test_ypred = streed.predict(X_test)
                streed_test_accuracy = accuracy_score(y_test, streed_test_ypred)
                print(f"{dataset.capitalize():10s} |  Run {i:2d} | STreeD    d={max_depth}  |  run time: {streed_duration:6.3f}  | train accuracy: {streed_train_accuracy*100:4.2f} | test accuracy: {streed_test_accuracy*100:4.2f}")
                results.append({"Method": "STreeD", "Dataset": dataset.capitalize(), "Run": i, "Depth": max_depth, "Runtime": streed_duration, "Train Accuracy": streed_train_accuracy*100, "Test Accuracy": streed_test_accuracy*100})

df = pd.DataFrame(results)

data = df.melt(id_vars=["Depth", "Method", "Dataset", "Run"], value_vars=["Runtime", "Train Accuracy", "Test Accuracy"])

# Set up plot settings
sns.set_context('paper')
plt.rc('font', size=10, family='serif')
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')
plt.rc('axes', labelsize='small', grid=True)
plt.rc('legend', fontsize='small')
plt.rc('pdf',fonttype = 42)
plt.rc('ps',fonttype = 42)
plt.rc('path', simplify = True)
sns.set_palette("colorblind")

g = sns.relplot(
    data=data,
    x="Depth",
    y="value",
    hue="Method",
    row="variable", 
    col="Dataset", 
    kind="line",
    row_order=["Train Accuracy", "Test Accuracy", "Runtime"],
    height=1.6,
    aspect=1.5,
    legend=True,
    facet_kws=dict(sharey=False, sharex=True))

g.axes.flat[0].xaxis.set_major_locator(MaxNLocator(integer=True))
g.set_titles('{col_name}')
for ax in g.axes[1, :]: ax.set_title("") 
for ax in g.axes[2, :]: ax.set_title("") 
g.axes[0,0].set_ylabel("Train Accuracy (%)")
g.axes[1,0].set_ylabel("Test Accuracy (%)")
g.axes[2,0].set_ylabel("Runtime (s)")
handles, labels = g.legend.legend_handles, g.legend.get_texts()
g.axes[0, -1].legend(handles=handles, labels=[label.get_text() for label in labels], loc='upper right')
g.legend.remove()

plt.tight_layout()
plt.show()
        