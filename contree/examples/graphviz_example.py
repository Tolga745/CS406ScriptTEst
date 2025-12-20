import graphviz
from sklearn.datasets import load_breast_cancer as load_data
from pycontree import ConTree
import pandas as pd

_data = load_data(as_frame=True)
data = pd.DataFrame(_data["data"], columns=_data["feature_names"])
target = _data["target"]
class_names = _data["target_names"]

model = ConTree(max_depth = 3)
model.fit(data, target)

print(str(model.get_tree()))

dot_graph = model.export_dot(label_names=class_names)
g = graphviz.Source(dot_graph)
g.render(outfile="tree.pdf", view=True, cleanup=True)