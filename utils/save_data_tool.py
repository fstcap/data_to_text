import os
import pickle


def save_pkl(path, data):
    """
    数据保存成pkl
    """
    root_path = path.split('/')
    if len(root_path) > 0:
        root_path = '/'.join(root_path[:-1])
        if not os.path.exists(root_path):
            os.makedirs(root_path)

    with open(path, "wb") as f:
        pickle.dump(data, f)
