from python_scripts.definitions import *

EVAL_NUM = 300
RT = "OFF"
LD = "ON"
q = 'q3'

ddses = ['fastrtps', 'opensplice', 'connext']
dds = ddses[0]

print(f"Current config. : {q} {dds} {EVAL_NUM}, RT:{RT}, LD:{LD}")

rebuild_flag = False
a = set_EVAL_NUM(EVAL_NUM)
b = set_RT(RT)

rebuild_flag = a or b
if rebuild_flag:
    rebuild_nodes()

run_meas(dds, EVAL_NUM, RT, LD)

meas_data = load_meas_data_q3(dds, EVAL_NUM, RT, LD)
make_boxplot(meas_data, dds, RT, LD)
# make_histogram(meas_data, dds, RT, LD)

