import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os
from pathlib import Path
from numpy import loadtxt

from python_scripts.definitions import *

EVAL_NUM = 3000
RT = 'ON'
dds = 'fastrtps'
rebuild = False
q = 'q1'


if __name__ == "__main__":
    a = set_RT(RT)
    b = set_EVAL_NUM(EVAL_NUM)
    rebuild = a or b
    if rebuild:
        rebuild_nodes()
    run_meas(dds)
    meas_data = load_meas_data()
    make_boxplot(meas_data, q, dds)





