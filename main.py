import os
import numpy as np

PWD = os.getcwd()


if __name__ == "__main__":

    res = np.load(os.path.join(PWD, 'datasets', 'rl_tables_single2023-02-13_18-35-02.pkl'), allow_pickle=True)
    print()
