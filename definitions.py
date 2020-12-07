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
        answer = input("Type 'yes' to Clean eval-times and Re-measure. Type 'no' to just plot. \n : ")
    if answer == 'yes':
        clean_eval_time()

    while answer != "Done" and answer != "no":
        if LD == 'OFF':
            instr = "Run the following commands: \n" \
                    "  *open terminal* \n" \
                    "  cd " + str(main_dir_path) + " \n" \
                    "  ros2 run interprocess_eval listener_interprocess__rmw_" + dds + "_cpp \n" \
                    "  CTRL+SHIFT+N \n" \
                    "  ros2 run interprocess_eval talker_interprocess__rmw_" + dds + "_cpp \n" \
                    "Type 'Done' to continue when the nodes have finished\n:"
        else:
            instr = "Run the following commands and type 'Done' to continue : \n" \
                    "  *open terminal* \n" \
                    "  cd " + str(main_dir_path) + " \n" \
                    "  ros2 run artificial_load CPU_load \n" \
                    "  CTRL+SHIFT+N \n" \
                    "  ros2 run interprocess_eval listener_interprocess__rmw_" + dds + "_cpp \n" \
                    "  CTRL+SHIFT+N \n" \
                    "  ros2 run interprocess_eval talker_interprocess__rmw_" + dds + "_cpp \n" \
                    "Type 'Done' to continue when the nodes have finished\n:"
        answer = input(instr)

    if answer == "Done":
        print("Calculating transport times ...")
        subprocess.run("./calculate", shell=True)
        print("Done.")

        from_dir = os.path.join(main_dir_path, 'evaluation/transport_time')
        to_dir = os.path.join(main_dir_path,
                              'python_scripts/measurements/transport_time_' + str(num) + "_" + dds + "_RT_" + RT + "_LD_" + LD)
        copy_tree(from_dir, to_dir)
        print(f"Measurements saved in {str(to_dir)}")


def load_meas_data_q1(nums):
    RT = 'ON'
    dds = 'fastrtps'
    LD = 'OFF'
    meas_data = [[0]*15 for _ in range(len(nums))]
    for i in range(len(nums)):
        path = os.path.join(main_dir_path, 'python_scripts/measurements/transport_time_' + str(nums[i]) + "_" + dds + "_RT_" + RT + "_LD_" + LD)
        print(f'loading meas data from {path}')
        meas_files = os.listdir(path)
        for file in meas_files:
            try:
                meas_data[i][index[file]] = loadtxt(os.path.join(path, file), dtype=np.float64)
            except:
                pass

    return meas_data


def load_meas_data_q3(dds, num, RT, LD):
    meas_data = [0] * 15
    path = os.path.join(main_dir_path, 'python_scripts/measurements/transport_time_' + str(num) + "_" + dds + "_RT_" + RT + "_LD_" + LD)
    meas_files = os.listdir(path)
    for file in meas_files:
        try:
            meas_data[index[file]] = loadtxt(os.path.join(path, file), dtype=np.float64)
        except:
            pass
    return meas_data


def make_boxplot(meas_data, dds, RT="ON", LD="OFF"):
    num = len(meas_data[0])
    # x = [256*pow(2,i) for i in range(15)]
    x = range(1, 16)
    fig = plt.figure(figsize=[10,7.5])
    plt.boxplot(meas_data, showfliers=False)
    plt.xticks(x, file_sizes)
    plt.ylabel("latency [s]")
    plt.xlabel("data size")
    plt.title(f"{dds} : EVAL_NUM={str(num)}, RT={RT}, CPU LD={LD}")
    plt.grid()
    plt.show()
    # f_name = 'python_scripts/figures/'+q+'_box_'+str(num)+'_'+dds+'.pickle'
    # with open(f_name, 'wb') as f:
    #     pickle.dump(fig, f)
    #     f.close()


def make_histogram(meas_data, dds, RT, LD):
    num = len(meas_data[0])
    idx= range(15)
    idx = [12,11,14]
    for i, data in enumerate(meas_data):
        if i in idx:
            plt.figure()
            plt.hist(data)
            ax=plt.gca()
            xlim = plt.xlim()
            ax.set_xlim([0,xlim[1]])
            plt.title(dds + ": EVAL_NUM=" + str(num) + ", " + file_sizes[i]+ f", RT={RT}, CPU LD={LD}")
            plt.xlabel('latency [s]')
            plt.ylabel("count")
            plt.show()
    # f_name = 'python_scripts/figures/'+q+'_hist_'+str(num)+'_'+dds+'.pickle'
    # with open(f_name, 'wb') as f:
    #     pickle.dump(fig, f)


def make_group_boxplot(data):

    # --- Labels for your data:
    labels_list = file_sizes
    xlocations = range(15)
    width = 0.3


    fig = plt.figure(figsize=[10,7.5])
    ax = plt.gca()
    ax.grid(True, linestyle='dotted')
    ax.set_axisbelow(True)
    ax.set_xticks(xlocations,labels_list)
    plt.ylabel("latency [s]")
    plt.xlabel("data size")
    plt.title(f'Measurements for eval nums 120, 150 and 1000')

    # --- Offset the positions per group:
    positions_group1 = [x - (width + 0.01) for x in xlocations]
    positions_group2 = xlocations
    positions_group3 = [x + (width + 0.01) for x in xlocations]

    plt.boxplot(data[0], boxprops=dict(color='red'), showfliers=False,
                sym='r',
                labels=[''] * len(labels_list),
                positions=positions_group1,
                widths=width
                )
    plt.legend(['120', '500', '1000'])

    plt.boxplot(data[1], boxprops=dict(color='green'), showfliers=False,
                labels=labels_list,
                sym='g',
                positions=positions_group2,
                widths=width
                )
    plt.boxplot(data[2], boxprops=dict(color='blue'), showfliers=False,
                labels=[''] * len(labels_list),
                sym='b',
                positions=positions_group3,
                widths=width
                )

    # plt.savefig('boxplot_grouped.png')
    # plt.savefig('boxplot_grouped.pdf')  # when publishing, use high quality PDFs
    plt.show()