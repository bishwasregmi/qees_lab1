import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os
from pathlib import Path
from numpy import loadtxt
import pickle
from distutils.dir_util import copy_tree

main_dir_path = Path(__file__).resolve().parents[1]
meas_path = main_dir_path.joinpath("evaluation/transport_time")
intr_proc_lstnr_path = main_dir_path.joinpath("src/interprocess_eval/src/listener_interprocess.cpp")
intr_proc_tlkr_path = main_dir_path.joinpath("src/interprocess_eval/src/talker_interprocess.cpp")

file_sizes = ['256B', '512B', '1kB', '2kB', '4kB', '8kB', '16kB', '32kB', '64kB', '128kB', '256kB', '512kB', '1MB',
              '2MB', '4MB']

index = {
    'transport_time_256byte.txt': 0,
    'transport_time_512byte.txt': 1,
    'transport_time_1Kbyte.txt': 2,
    'transport_time_2Kbyte.txt': 3,
    'transport_time_4Kbyte.txt': 4,
    'transport_time_8Kbyte.txt': 5,
    'transport_time_16Kbyte.txt': 6,
    'transport_time_32Kbyte.txt': 7,
    'transport_time_64Kbyte.txt': 8,
    'transport_time_128Kbyte.txt': 9,
    'transport_time_256Kbyte.txt': 10,
    'transport_time_512Kbyte.txt': 11,
    'transport_time_1Mbyte.txt': 12,
    'transport_time_2Mbyte.txt': 13,
    'transport_time_4Mbyte.txt': 14
}


# check EVAL_NUM is the same as set value. If not, change it
def set_EVAL_NUM(value):
    rebuild = False
    paths = [intr_proc_lstnr_path, intr_proc_tlkr_path]

    for path in paths:
        with open(path, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if "#define EVAL_NUM" in line:
                    curr_value = [int(s) for s in line.split() if s.isdigit()][0]
                    if curr_value != value:
                        rebuild = True
                        lines[i] = "#define EVAL_NUM " + str(value) + " \n"
                        print(f"In {path} line {i}: {line} changed to {lines[i]}")
                        break

        with open(path, 'w') as f:
            f.writelines(lines)
            f.close()

    return rebuild


def set_RT(value):
    rebuild = False
    paths = [intr_proc_lstnr_path, intr_proc_tlkr_path]

    for path in paths:
        with open(path, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if "#define RUN_REAL_TIME" in line and value == "OFF":
                    rebuild = True
                    lines[i] = "//#define RUN_REAL_TIME \n"
                    print(f"In {path} line {i}: {line} changed to {lines[i]}")
                    break
                if "//#define RUN_REAL_TIME" in line and value == "ON":
                    rebuild = True
                    lines[i] = "#define RUN_REAL_TIME \n"
                    print(f"In {path} line {i}: {line} changed to {lines[i]}")
                    break
        with open(path, 'w') as f:
            f.writelines(lines)
            f.close()

    return rebuild


def rebuild_nodes():
    print("Rebuilding ...")
    os.chdir(main_dir_path)
    subprocess.run(["colcon", "build", "--symlink-install"], check=True)
    print("Done Rebuilding.")
    subprocess.run(["g++", "calculation_transport_time.cpp", "-o", "calculate"], check=True)


# clean previous measurements
def clean_eval_time():
    print("Cleaning previous measurements..")
    # subprocess.run("rm -f " + os.path.join(path, "evaluation/publish_time/*.txt "), shell=True)
    # subprocess.run("rm -f " + os.path.join(path, "evaluation/subscribe_time/*.txt "), shell=True)
    # subprocess.run("rm -f " + os.path.join(path, "evaluation/transport_time/*.txt "), shell=True)
    # subprocess.run("rm -f " + os.path.join(path, "evaluation/convert_msg/*.txt"), shell=True)
    # subprocess.run("rm -f " + os.path.join(path, "evaluation/dds_time/*.txt"), shell=True)

    os.chdir(main_dir_path)
    subprocess.run("sh clean_evaltime.bash", shell=True)
    print("Done Cleaning.")


def run_meas(dds, num, RT, LD):
    os.chdir(main_dir_path)
    # cmd1 = "ros2 run interprocess_eval listener_interprocess__rmw_fastrtps_cpp"
    # cmd2 = "ros2 run interprocess_eval talker_interprocess__rmw_fastrtps_cpp"
    # os.system("gnome-terminal --title=newWindow -- bash -c  'bash'" )
    # process = os.system("gnome-terminal --title=newWindow -- bash -c  'ros2 run interprocess_eval talker_interprocess__rmw_fastrtps_cpp; bash'")
    # process.wait()

    answer = " "
    while answer != "yes" and answer != "no":
        answer = input("Are you sure you want to clean eval-times? \n Type 'yes' or 'no' : \n")
    if answer == 'yes':
        clean_eval_time()

    while answer != "Done" and answer != "no":
        if LD == 'OFF':
            instr = "Run the following commands and type 'Done' to continue : \n" \
                    "  *open terminal* \n" \
                    "  cd " + str(main_dir_path) + " \n" \
                                                   "  ros2 run interprocess_eval listener_interprocess__rmw_" + dds + "_cpp \n" \
                                                                                                                      "  CTRL+SHIFT+N \n" \
                                                                                                                      "  ros2 run interprocess_eval talker_interprocess__rmw_" + dds + "_cpp \n"
        else:
            instr = "Run the following commands and type 'Done' to continue : \n" \
                    "  *open terminal* \n" \
                    "  cd " + str(main_dir_path) + " \n" \
                                                   "  ros2 run artificial_load CPU_load \n" \
                                                   "  CTRL+SHIFT+N \n" \
                                                   "  ros2 run interprocess_eval listener_interprocess__rmw_" + dds + "_cpp \n" \
                                                                                                                      "  CTRL+SHIFT+N \n" \
                                                                                                                      "  ros2 run interprocess_eval talker_interprocess__rmw_" + dds + "_cpp \n"
        answer = input(instr)

    if answer == "Done":
        print("Calculating transport times ...")
        subprocess.run("./calculate", shell=True)
        print("Done.")

        from_dir = os.path.join(main_dir_path, 'evaluation/transport_time')
        to_dir = os.path.join(main_dir_path,
                              'evaluation/transport_time_' + str(num) + "_" + dds + "_RT_" + RT + "_LD_" + LD)
        copy_tree(from_dir, to_dir)
        print(f"Measurements saved in {str(to_dir)}")


def load_meas_data():
    meas_data = [0] * 15
    meas_files = os.listdir(meas_path)
    for file in meas_files:
        try:
            meas_data[index[file]] = loadtxt(os.path.join(meas_path, file), dtype=np.float64)
        except:
            pass
    return meas_data


def load_meas_data_q3(dds, num, RT, LD):
    meas_data = [0] * 15
    path = os.path.join(main_dir_path, 'evaluation/transport_time_' + str(num) + "_" + dds + "_RT_" + RT + "_LD_" + LD)
    meas_files = os.listdir(path)
    for file in meas_files:
        try:
            meas_data[index[file]] = loadtxt(os.path.join(path, file), dtype=np.float64)
        except:
            pass
    return meas_data


def make_boxplot(meas_data, q, dds):
    num = len(meas_data[0])
    # x = [256*pow(2,i) for i in range(15)]
    x = range(1, 16)
    fig = plt.figure()
    plt.boxplot(meas_data, showfliers=False)
    plt.xticks(x, file_sizes)
    plt.ylabel("latency [s]")
    plt.xlabel("data size")
    plt.title(dds + " EVAL_NUM=" + str(num))
    plt.grid()
    plt.show()
    # f_name = 'python_scripts/figures/'+q+'_box_'+str(num)+'_'+dds+'.pickle'
    # with open(f_name, 'wb') as f:
    #     pickle.dump(fig, f)
    #     f.close()


def open_boxplot():
    q = input('q : ')
    num = input('num : ')
    dds = input('dds : ')
    f_name = 'python_scripts/figures/' + q + '_box_' + str(num) + '_' + dds + '.pickle'
    print(f_name)
    fig = plt.figure()
    fig = pickle.load(open(f_name, 'rb'))
    fig.show()


def make_histogram(meas_data, q, dds):
    num = len(meas_data[0]) + 1
    for i, data in enumerate(meas_data):
        plt.figure()
        plt.hist(data)
        plt.title(dds + " EVAL_NUM=" + str(num) + " " + file_sizes[i])
        plt.xlabel('latency [s]')
        plt.ylabel("count")
        plt.show()
    # f_name = 'python_scripts/figures/'+q+'_hist_'+str(num)+'_'+dds+'.pickle'
    # with open(f_name, 'wb') as f:
    #     pickle.dump(fig, f)
