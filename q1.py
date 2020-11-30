import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os
from pathlib import Path
from numpy import loadtxt

from python_scripts.definitions import *

EVAL_NUM = 3000

set_EVAL_NUM(intr_proc_lstnr_path, intr_proc_tlkr_path, EVAL_NUM)
run_meas()
make_boxplot(EVAL_NUM)





