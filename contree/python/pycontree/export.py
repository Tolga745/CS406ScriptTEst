from .ccontree import Tree
import numpy as np
import math

class TreeExporter:
     
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.__comparator = "&#8804;" # &#8804; is the <= character
    
    def export(self, fh, tree, feature_names, label_names, train_data):
        fh.write("digraph Tree {\n")
        fh.write("node [shape=box, style=\"filled\", fontname=\"helvetica\", fontsize=\"8\"] ;\n")
        fh.write("edge [fontname=\"helvetica\", fontsize=\"6\"] ;\n")
        self.recursive_export_dot(fh, tree, 0, feature_names, label_names, train_data)
        fh.write("}")

    def get_label_str(self, label, label_names=None):
        return f"Class {label}" if not isinstance(label, int) or label_names is None else label_names[label]

    def get_predicate_str(self, feature, threshold, feature_names=None):
        if feature_names is None:
            return f"Feature {feature} {self.__comparator} {threshold}"
        feature_name = feature_names[feature]
        threshold = _column_threshold(float(threshold))
        return f"{feature_name} {self.__comparator} {threshold}"

    def get_class_counts_str(self, train_y):
        count_dict = {label: 0 for label in range(self.n_classes)}
        unique, counts = np.unique(train_y, return_counts=True)
        total = sum(counts)
        gini = 1 - sum([(c/total)**2 for c in counts])
        for label, count in zip(unique, counts):
            count_dict[label] = count
        counts = [(k, v) for k, v in count_dict.items()]
        counts = sorted(counts, key=lambda c: c[0])
        return "Counts = \["+ ",".join([str(c[1]) for c in counts]) + "\]\n" \
            + f"Gini = {gini:.3f}"

    def export_dot_leaf_node(self, fh, node: Tree, node_id, label_names, train_data):
        if not hasattr(self, "colors"):
            self.colors = _color_brew(self.n_classes)
        color = self.colors[node.get_label()]
        label =  self.get_label_str(node.get_label(), label_names)
        hex_color = "#{:02x}{:02x}{:02x}".format(*color)
        hex_line_color = "#{:02x}{:02x}{:02x}".format(*[int(0.4 * c) for c in color])
        class_count_str = self.get_class_counts_str(train_data[1])
        fh.write(f"{node_id}  [label=\"{label}\n{class_count_str}\", color=\"{hex_line_color}\" fillcolor=\"{hex_color}\"] ;\n")
    
    def get_branching_color(self, train_y):
        if not hasattr(self, "colors"):
            self.colors = _color_brew(self.n_classes)
        unique, counts = np.unique(train_y, return_counts=True)
        color = list(self.colors[unique[np.argmax(counts)]])
        sorted_counts = sorted(counts, reverse=True)
        if len(sorted_counts) == 1:
            alpha = 0.0
        else:
            total = sum(counts)
            alpha = (sorted_counts[0] - sorted_counts[1]) / (total - sorted_counts[1])
        color = [int(round(alpha * c + (1 - alpha) * 255, 0)) for c in color]
        # Return html color code in #RRGGBB format
        return "#%2x%2x%2x" % tuple(color)


    def recursive_export_dot(self, fh, node: Tree, node_id, feature_names, label_names, train_data, parent_threshold=None):
        if node.is_leaf_node():
            self.export_dot_leaf_node(fh, node, node_id, label_names, train_data)
        else:
            predicate = self.get_predicate_str(node.get_split_feature(), node.get_split_threshold(), feature_names)
            branching_color = self.get_branching_color(train_data[1])
            class_count_str = self.get_class_counts_str(train_data[1])
            fh.write(f"{node_id} [label=\"{predicate}\n{class_count_str}\", color=\"#222222\", fillcolor=\"{branching_color}\"] ;\n")
        if node_id > 0 and not parent_threshold is None:
            threshold = _column_threshold(parent_threshold)
            parent_id = (node_id - 1) // 2
            if node_id % 2 == 0:
                feature_label = f"> {threshold}"
                angle = 45
            else:
                feature_label = f"{self.__comparator} {threshold}"
                angle = -45
            fh.write(f"{parent_id} -> {node_id} [labeldistance=2.5, labelangle={angle}, label=\"{feature_label}\"] ;\n")

        if node.is_branching_node():
            left_data, right_data = self.split(train_data, node.get_split_feature(), node.get_split_threshold())
            self.recursive_export_dot(fh, node.get_left(), node_id * 2 + 1, feature_names, label_names, left_data, node.get_split_threshold())
            self.recursive_export_dot(fh, node.get_right(), node_id * 2 + 2, feature_names, label_names, right_data, node.get_split_threshold())

     
    def split(self, data, feature, threshold):
        x, y = data
        go_left = x[:, feature] <= threshold
        left_data =  (x[ go_left], y[ go_left])
        right_data = (x[~go_left], y[~go_left])
        return left_data, right_data

def _column_threshold(t):
    if int(t) == t:
        return str(t)
    if t % 1 == 0:
        return str(t)
    if math.log10(abs(t)) >= 6 or math.log10(abs(t)) <= -4:
        return f"{t:.2e}"
    if math.log10(abs(t)) >= 2:
        return f"{t:.2f}".rstrip('0').rstrip('.')
    return f"{t:f}".rstrip('0').rstrip('.')

def _color_brew(n):
    """Generate n colors with equally spaced hues.
    From: https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/tree/_export.py

    Parameters
    ----------
    n : int
        The number of colors required.

    Returns
    -------
    color_list : list, length n
        List of n tuples of form (R, G, B) being the components of each color.
    """
    color_list = []

    # Initialize saturation & value; calculate chroma & value shift
    s, v = 0.75, 0.9
    c = s * v
    m = v - c

    for h in np.arange(25, 385, 360.0 / n).astype(int):
        # Calculate some intermediate values
        h_bar = h / 60.0
        x = c * (1 - abs((h_bar % 2) - 1))
        # Initialize RGB with same hue & chroma as our color
        rgb = [
            (c, x, 0),
            (x, c, 0),
            (0, c, x),
            (0, x, c),
            (x, 0, c),
            (c, 0, x),
            (c, x, 0),
        ]
        r, g, b = rgb[int(h_bar)]
        # Shift the initial RGB values to match value and store
        rgb = [(int(255 * (r + m))), (int(255 * (g + m))), (int(255 * (b + m)))]
        color_list.append(rgb)

    return np.array(color_list)