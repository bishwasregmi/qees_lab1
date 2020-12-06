import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os
from pathlib import Path
from numpy import loadtxt

from python_scripts.definitions import *

EVAL_NUM = 3000


if __name__ == "__main__":
    a = set_RT(RT)
    b = set_EVAL_NUM(EVAL_NUM)
    rebuild = a or b
    if rebuild:
        rebuild_nodes()
    run_meas(dds, EVAL_NUM, RT, LD)
    meas_data = load_meas_data_q1(eval_nums)
    make_group_boxplot(meas_data)





