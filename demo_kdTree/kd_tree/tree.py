import numpy as np
import time
from anytree import NodeMixin
from anytree.exporter.dotexporter import UniqueDotExporter
from tkinter import Canvas
from PIL import Image, ImageTk


class KdData:
    def __init__(self) -> None:
        self.key = None


class KdNode(KdData, NodeMixin):
    def __init__(self, parent=None, children=None, axis: int = 0) -> None:
        super().__init__()
        self.axis = axis
        self.parent = parent
        self.trace = False
        self.found = False
        self.canvas = None
        self.img_container = None
        self.img = None
        self.tk_root = None
        self.decoy = False
        self.dis = None
        if children:
            self.children = children

    def set_key(self, key):
        self.key = key
        if self.key is not None:
            self.children = [
                KdNode(axis=(self.axis + 1) % self.key.shape[0]),
                KdNode(axis=(self.axis + 1) % self.key.shape[0]),
                KdNode(axis=(self.axis + 1) % self.key.shape[0])
            ]
            self.get_left_node().set_canvas(self.canvas)
            self.get_left_node().img_container = self.img_container
            self.get_right_node().set_canvas(self.canvas)
            self.get_right_node().img_container = self.img_container

    def get_key(self):
        return self.key

    def get_left_node(self):
        if self.key is None:
            return None
        return self.children[0]

    def get_right_node(self):
        if self.key is None:
            return None
        return self.children[2]

    def insert(self, key, trace=False):
        self.reset_decoy()
        self.reset_found()
        self.reset_trace()
        self.trace = trace
        if trace and self.key is not None:
            time.sleep(1)
        if self.key is None:
            self.set_key(key)
        elif key[self.axis] < self.key[self.axis]:
            self.get_left_node().insert(key, trace)
        else:
            self.get_right_node().insert(key, trace)

    def reset_trace(self):
        self.trace = False
        if self.key is None:
            return
        self.get_left_node().reset_trace()
        self.get_right_node().reset_trace()

    def get_smallest_in_axis(self, axis=0) -> NodeMixin:
        result = self
        left_node = self.get_left_node()
        right_node = self.get_right_node()
        if self.axis == axis and left_node is not None:
            temp = left_node.get_smallest_in_axis(axis)
            if temp.key is not None and temp.key[axis] < result.key[axis]:
                result = temp
        if right_node is not None:
            temp = right_node.get_smallest_in_axis(axis)
            if temp.key is not None and temp.key[axis] < result.key[axis]:
                result = temp
        return result

    def get_largest_in_axis(self, axis=0) -> NodeMixin:
        result = self
        left_node = self.get_left_node()
        right_node = self.get_right_node()
        if self.axis == axis and right_node is not None:
            temp = right_node.get_largest_in_axis(axis)
            if temp.key is not None and temp.key[axis] > result.key[axis]:
                result = temp
        if left_node is not None:
            temp = left_node.get_largest_in_axis(axis)
            if temp.key is not None and temp.key[axis] < result.key[axis]:
                result = temp
        return result

    def is_leaf(self):
        left_node = self.get_left_node()
        right_node = self.get_right_node()
        if left_node.key is None and right_node.key is None:
            return True
        return False

    def delete(self, key, trace=False):
        self.reset_decoy()
        self.reset_found()
        self.reset_trace()
        target_node = self.find_exact(key, trace)
        if target_node is None:
            return
        if trace:
            time.sleep(1)
        left_node = target_node.get_left_node()
        right_node = target_node.get_right_node()
        target_node.found = False
        if target_node.is_leaf():
            # Detach children
            target_node.children = []
            # Remove key
            target_node.key = None
            return
        elif right_node.key is not None:
            new_target_node = right_node.get_smallest_in_axis(target_node.axis)
            new_target_node.decoy = trace
            if trace:
                time.sleep(1)
            # Swap key
            target_node.key, new_target_node.key = new_target_node.key, target_node.key
            if trace:
                time.sleep(1)
            # Delete recursively
            new_target_node.delete(key, trace)
            return
        elif left_node.key is not None:
            new_target_node = left_node.get_largest_in_axis(target_node.axis)
            new_target_node.decoy = trace
            if trace:
                time.sleep(1)
            # Swap key
            target_node.key, new_target_node.key = new_target_node.key, target_node.key
            if trace:
                time.sleep(1)
            # Delete recursively
            new_target_node.delete(key, trace)
            return

    def find_exact(self, key, trace=False):
        self.reset_decoy()
        self.reset_found()
        self.reset_trace()
        if self.key is None:
            return None
        self.trace = trace
        if trace:
            time.sleep(1)
        target_key = np.array(key)
        if np.all(self.key == target_key):
            self.set_found()
            if trace:
                time.sleep(1)
            return self
        left_node = self.get_left_node()
        right_node = self.get_right_node()
        if target_key[self.axis] < self.key[self.axis]:
            return left_node.find_exact(key, trace)
        else:
            return right_node.find_exact(key, trace)

    def find_range(self, axis, left_bound, right_bound, trace=False) -> list[NodeMixin]:
        self.reset_decoy()
        self.reset_found()
        result = list()
        self.reset_trace()
        self.trace = trace
        if self.key is None:
            return result
        if trace:
            time.sleep(1)
        left_node = self.get_left_node()
        right_node = self.get_right_node()
        if left_bound <= self.key[axis] <= right_bound:
            self.set_found()
            if trace:
                time.sleep(1)
            result.append(self)
            result += right_node.find_range(axis, left_bound, right_bound, trace)
            result += left_node.find_range(axis, left_bound, right_bound, trace)
        elif self.axis == axis:
            if self.key[axis] < left_bound:
                result += right_node.find_range(axis, left_bound, right_bound, trace)
            else:
                result += left_node.find_range(axis, left_bound, right_bound, trace)
        else:
            result += right_node.find_range(axis, left_bound, right_bound, trace)
            result += left_node.find_range(axis, left_bound, right_bound, trace)
        return result

    def set_found(self):
        self.found = True

    def set_found_range(self):
        self.set_found()
        if self.key is None:
            return
        self.get_left_node().set_found_range()
        self.get_right_node().set_found_range()

    def reset_decoy(self):
        self.decoy = False
        if self.key is None:
            return
        self.get_left_node().reset_decoy()
        self.get_right_node().reset_decoy()

    def reset_found(self):
        self.found = False
        if self.key is None:
            return
        self.get_left_node().reset_found()
        self.get_right_node().reset_found()

    def set_canvas(self, canvas: Canvas):
        self.canvas = canvas
        if self.key is None:
            return
        self.get_left_node().set_canvas(canvas)
        self.get_right_node().set_canvas(canvas)

    def update_tree(self):
        dots = UniqueDotExporter(self.root, options=[],
                                 nodeattrfunc=lambda node: _node_attribute(node),
                                 edgeattrfunc=lambda src, tgt: _edge_att(src, tgt))
        dots.to_picture('kd_tree/temp.png')
        assert isinstance(self.canvas, Canvas)
        img_n = Image.open('kd_tree/temp.png')
        img = ImageTk.PhotoImage(img_n)
        self.img = img
        if len(self.canvas.find_all()) == 0:
            self.img_container = self.canvas.create_image(500, 0,
                                                          anchor='n', image=self.img)
        else:
            self.canvas.itemconfigure(self.img_container, image=self.img)

    def set_dis(self, value):
        self.dis = value

    def reset_dis(self):
        self.dis = None
        if self.key is None:
            return
        self.get_left_node().reset_dis()
        self.get_right_node().reset_dis()


def _node_attribute(node):
    att = "shape=plaintext"
    att += " fixedsize=true"
    att += " width=1"
    att += " height=1"
    if node.key is None:
        att += " label=\"\""
    else:
        att += f" label=\"{node.key}\naxis = {node.axis}\ndis = {node.dis}\""
    if node.found:
        att += " fontcolor=\"#04AF70\""
    elif node.trace:
        att += " fontcolor=\"#FF0000\""
    elif node.decoy:
        att += " fontcolor=\"#0000FF\""
    return att


def _edge_att(source, target):
    att = ""
    if target.key is None:
        att += "style=invis"
    elif target.trace:
        att += "color=\"#FF0000\""
    return att


def _median(arr, axis):
    sorted_arr = arr[np.argsort(arr[:, axis])]
    if len(sorted_arr) % 2 == 1:
        result = sorted_arr[len(sorted_arr) // 2]
    else:
        result = sorted_arr[len(sorted_arr) // 2 - 1]
    return result


def _median_of_median(arr, axis):
    new_arr = np.zeros((0, 2), dtype=int)
    i = 0
    while i < len(arr):
        if i + 5 < len(arr):
            new_arr = np.append(new_arr, [_median(arr[i: i + 5], axis)], axis=0)
        else:
            new_arr = np.append(new_arr, [_median(arr[i: len(arr)], axis)], axis=0)
        i += 5
    if len(new_arr) <= 5:
        return _median(new_arr, axis)
    else:
        return _median_of_median(new_arr, axis)


def _distance(key1, key2):
    return abs(key1[0]-key2[0]) + abs(key1[1]-key2[1])


class KDTree:
    def __init__(self, n_axis=2) -> None:
        self.root = KdNode()
        self.n_axis = n_axis

    def set_canvas(self, canvas: Canvas):
        self.root.set_canvas(canvas)

    def insert(self, key, trace=False):
        assert len(key) == self.n_axis
        self.root.insert(key, trace)

    def delete(self, key, trace=False):
        assert len(key) == self.n_axis
        self.root.delete(key, trace)

    def build_balanced_tree(self, curr, data, axis):
        if len(data) < 1:
            return None

        right_arr = np.zeros((0, 2), dtype=int)
        left_arr = np.zeros((0, 2), dtype=int)

        x = _median(data, axis)

        curr.set_key(x)

        indices_to_delete = np.where(np.all(data == x, axis=1))
        data = np.delete(data, indices_to_delete, axis=0)

        for i in range(len(data)):
            if data[i][axis] <= x[axis]:
                left_arr = np.append(left_arr, [data[i]], axis=0)
            else:
                right_arr = np.append(right_arr, [data[i]], axis=0)

        self.build_balanced_tree(curr.get_left_node(), left_arr, (axis + 1) % self.n_axis)
        self.build_balanced_tree(curr.get_right_node(), right_arr, (axis + 1) % self.n_axis)

    def balance(self, data):
        self.build_balanced_tree(self.root, data, 0)

    def find_exact(self, key, trace=False):
        assert len(key) == self.n_axis
        self.root.find_exact(key, trace)

    def find_range(self, axis, left_bound, right_bound, trace=False):
        self.root.find_range(axis, left_bound, right_bound, trace)

    def build_random(self, n_nodes=20, seed=None, node_size=2):
        np.random.seed(seed)
        data = np.random.randint(100, size=(n_nodes, node_size))
        self.balance(data)

    def find_k_nearest_neighbors(self, key, curr, result, curr_min, curr_node_min, k=1, axis=0, trace=False):
        if curr.get_key() is None:
            return

        x = curr.get_key()

        dis = _distance(key, x)
        curr.set_dis(dis)

        curr.trace = trace
        if trace:
            time.sleep(1)

        if not result:
            result.append(x)
            curr_min.append(dis)
            curr_node_min.append(curr)
            curr.found = True
            if trace:
                time.sleep(1)

        elif dis < curr_min[-1] or len(result) < k:
            i = 0
            while curr_min[i] < dis and i < len(result)-1:
                i += 1
            result.insert(i, x)
            curr_min.insert(i, dis)
            curr_node_min.insert(i, curr)
            curr.found = True
            if trace:
                time.sleep(1)
            if len(result) > k:
                result.pop(-1)
                curr_min.pop(-1)
                curr_node_min[-1].found = False
                curr_node_min.pop(-1)
                if trace:
                    time.sleep(1)

        to_boundary = key[axis] - x[axis]

        if to_boundary < 0:
            self.find_k_nearest_neighbors(key, curr.get_left_node(), result, curr_min, curr_node_min,
                                          k, (axis + 1) % self.n_axis, trace)
            if len(result) < k or abs(to_boundary) < curr_min[-1]:
                self.find_k_nearest_neighbors(key, curr.get_right_node(), result, curr_min, curr_node_min,
                                              k, (axis + 1) % self.n_axis, trace)
        else:
            self.find_k_nearest_neighbors(key, curr.get_right_node(), result, curr_min, curr_node_min,
                                          k, (axis + 1) % self.n_axis, trace)
            if len(result) < k or abs(to_boundary) < curr_min[-1]:
                self.find_k_nearest_neighbors(key, curr.get_left_node(), result, curr_min, curr_node_min,
                                              k, (axis + 1) % self.n_axis, trace)

    def k_nearest_neighbors(self, key, k=1, trace=False):
        self.root.reset_dis()
        self.root.reset_decoy()
        self.root.reset_trace()
        self.root.reset_found()
        result = list()
        list_min = list()
        current_node_min = list()
        self.find_k_nearest_neighbors(key, self.root, result, list_min, current_node_min, k, 0, trace)
        return result, list_min, current_node_min
