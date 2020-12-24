from python_scripts.definitions import *

EVAL_NUM = 500
RT = "ON"
LD = "ON"
FQ_SCL = "ON"
q = 'q5'

ddses = ['fastrtps', 'opensplice', 'connext']
dds = ddses[0]

print(f"Current config. : {q} {dds} {EVAL_NUM}, RT:{RT}, LD:{LD}, FQ SCL:{FQ_SCL}")

rebuild_flag = False
a = set_EVAL_NUM(EVAL_NUM)
b = set_RT(RT)

rebuild_flag = a or b
if rebuild_flag:
    rebuild_nodes()

run_meas_q5(dds, EVAL_NUM, RT, LD, FQ_SCL)

meas_data = load_meas_data_q5(dds, EVAL_NUM, RT, LD, FQ_SCL)
make_boxplot(meas_data, dds, RT, LD, FQ_SCL)
# make_histogram(meas_data, dds, RT, LD)

