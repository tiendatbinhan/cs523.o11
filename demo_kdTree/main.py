import tkinter as tk
from threading import Thread
import ctypes
from kd_tree.tree import KDTree
import numpy as np

ctypes.windll.shcore.SetProcessDpiAwareness(1)

if __name__ == '__main__':
    my_tree = KDTree(n_axis=2)

    root = tk.Tk()
    root.geometry("1700x1000")
    root.resizable(False, False)
    root.title("Demo K-dimensional Tree")
    root.grid_rowconfigure(index=0, weight=1)
    for i in range(3):
        if i != 0:
            root.grid_columnconfigure(index=i, weight=1)

    tree_frame = tk.Frame(root, padx=10, pady=10)
    tree_frame.grid(row=0, column=0, sticky='n w s e')
    tree_canvas = tk.Canvas(tree_frame, width=1400, height=900, borderwidth=5, bg="#FFFFFF")
    tree_canvas.pack(side=tk.TOP, expand=True, fill="both")
    width_scroll = tk.Scrollbar(tree_frame, orient="horizontal")
    width_scroll.pack(side=tk.TOP, expand=True, fill="x")
    loc = width_scroll.get()
    tree_canvas.configure(xscrollcommand=width_scroll.set, scrollregion=(-1000, -1000, 2000, 2000))
    width_scroll.configure(command=tree_canvas.xview)

    my_tree.set_canvas(tree_canvas)
    my_tree.build_random(10, seed=1303)

    def loop_update_tree():
        my_tree.root.update_tree()
        root.after(500, loop_update_tree)

    loop_update_tree()

    main_frame = tk.Frame(root, padx=10, pady=10)
    main_frame.grid(row=0, column=1, sticky='n s')

    # scatter_frame = tk.Frame(root, padx=10, pady=10)
    # scatter_frame.grid(row=0, column=2, sticky='n w s')
    # scatter_canvas = tk.Canvas(scatter_frame, borderwidth=5)
    # scatter_canvas.pack(expand=True, fill="both")

    tree_modification_frame = tk.Frame(main_frame)
    tree_modification_frame.pack()

    tk.Label(tree_modification_frame, text="Tree modification", padx=10, pady=10, font=("TkDefaultFont", 14, "bold")).pack()
    entry_x = tk.Entry(tree_modification_frame)
    entry_y = tk.Entry(tree_modification_frame)
    entry_x.pack()
    entry_y.pack()

    def tree_thread(function=my_tree.find_exact):
        x = int(entry_x.get())
        y = int(entry_y.get())
        thread = Thread(target=function, kwargs={"key": np.array([x, y]),
                                                 "trace": True})
        thread.start()

    tk.Button(tree_modification_frame, text="Find", width=10,
              command=lambda: tree_thread(my_tree.find_exact)).pack()
    tk.Button(tree_modification_frame, text="Insert", width=10,
              command=lambda: tree_thread(my_tree.insert)).pack()
    tk.Button(tree_modification_frame, text="Delete", width=10,
              command=lambda: tree_thread(my_tree.delete)).pack()

    tk.Label(main_frame, text="Range query", padx=10, pady=10, font=("TkDefaultFont", 14, "bold")).pack()
    range_query_frame = tk.Frame(main_frame, padx=10, pady=10)
    range_query_frame.pack()
    tk.Label(range_query_frame, text="Start", padx=5, pady=5).grid(row=1, column=0)
    tk.Label(range_query_frame, text="End", padx=5, pady=5).grid(row=2, column=0)
    entry_left_bound = tk.Entry(range_query_frame)
    entry_left_bound.grid(row=1, column=1)
    entry_right_bound = tk.Entry(range_query_frame)
    entry_right_bound.grid(row=2, column=1)
    button_axis = tk.IntVar(value=0)
    button_x_axis = tk.Radiobutton(range_query_frame, variable=button_axis, text="x axis", value=0)
    button_x_axis.grid(row=3, column=0)
    button_y_axis = tk.Radiobutton(range_query_frame, variable=button_axis, text="y axis", value=1)
    button_y_axis.grid(row=4, column=0)

    def range_query_thread():
        left_bound = int(entry_left_bound.get())
        right_bound = int(entry_right_bound.get())
        axis = button_axis.get()
        Thread(target=my_tree.find_range, kwargs={"axis": axis, "left_bound": left_bound,
                                                  "right_bound": right_bound, "trace": True}).start()
    tk.Button(range_query_frame, text="Go", width=4, command=range_query_thread).grid(row=5, column=0)

    knn_frame = tk.Frame(main_frame, padx=10, pady=10)
    knn_frame.pack()
    tk.Label(knn_frame, text="KNN search", padx=10, pady=10, font=("TkDefaultFont", 14, "bold")).pack()
    entry_x_knn = tk.Entry(knn_frame)
    entry_y_knn = tk.Entry(knn_frame)
    entry_x_knn.pack()
    entry_y_knn.pack()
    tk.Label(knn_frame, text="k", padx=10, pady=10, anchor='nw', width=100).pack()
    entry_k_knn = tk.Entry(knn_frame)
    entry_k_knn.pack()

    def knn_thread():
        x = int(entry_x_knn.get())
        y = int(entry_y_knn.get())
        k = int(entry_k_knn.get())
        Thread(target=lambda: my_tree.k_nearest_neighbors(np.array([x, y]), k, True)).start()
    tk.Button(knn_frame, text="Go", width=4, command=knn_thread).pack()

    root.mainloop()
