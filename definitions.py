import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os
import difflib
from pathlib import Path
from numpy import loadtxt
import pickle
from distutils.dir_util import copy_tree
from scipy.stats import norm

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


def run_meas_q5(dds, num, RT, LD, FQ_SCL):
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
                              'python_scripts/measurements/transport_time_' + str(num) + "_" + dds + "_RT_" + RT + "_LD_" + LD + "_FqScl_"+FQ_SCL)
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

def load_meas_data_q5(dds, num, RT, LD, FQ_SCL):
    meas_data = [0] * 15
    path = os.path.join(main_dir_path, 'python_scripts/measurements/transport_time_' + str(num) + "_" + dds + "_RT_" + RT + "_LD_" + LD + "_FqScl_"+FQ_SCL)
    meas_files = os.listdir(path)
    for file in meas_files:
        try:
            meas_data[index[file]] = loadtxt(os.path.join(path, file), dtype=np.float64)
        except:
            pass
    return meas_data

def load_meas_data_q4():
    path1 = os.path.join(main_dir_path, 'python_scripts/measurements/transport_time_500_fastrtps_RT_ON_LD_OFF/transport_time_256byte.txt')
    path2 = os.path.join(main_dir_path, 'python_scripts/measurements/transport_time_500_fastrtps_RT_ON_LD_OFF/transport_time_128Kbyte.txt')
    path3 = os.path.join(main_dir_path, 'python_scripts/measurements/transport_time_500_opensplice_RT_ON_LD_OFF/transport_time_256byte.txt')
    path4 = os.path.join(main_dir_path, 'python_scripts/measurements/transport_time_500_opensplice_RT_ON_LD_OFF/transport_time_128Kbyte.txt')
    path5 = os.path.join(main_dir_path, 'python_scripts/measurements/transport_time_500_connext_RT_ON_LD_OFF/transport_time_256byte.txt')
    path6 = os.path.join(main_dir_path, 'python_scripts/measurements/transport_time_500_connext_RT_ON_LD_OFF/transport_time_128Kbyte.txt')

    paths = [[path1, path3, path5], [path2, path4, path6]]
    data = [[0]*3 for i in range(2)]

    for i,_ in enumerate(paths):
        for j,file in enumerate(paths[i]):
            data[i][j] =loadtxt(file, dtype=np.float64)
    return data

def load_meas_data_q7(num):

    meas_data = [[0]*15 for _ in range(num)]
    for i in range(num):
        path = os.path.join(main_dir_path, 'python_scripts/measurements/transport_time_' + str(num) + "nodes/node_"+str(i+1))
        print(f'loading meas data from {path}')
        meas_files = os.listdir(path)
        for file in meas_files:
            try:
                meas_data[i][index[file]] = loadtxt(os.path.join(path, file), dtype=np.float64)
            except:
                pass

    return meas_data

def make_boxplot_q7(num,data):

    plt.figure()
    plt.title(f"{num} listener nodes")
    x = range(1, 16)
    for i in range(num):
        plt.subplot(1,2,i+1)
        plt.boxplot(data[i], showfliers=False)
        plt.title(f"node {i+1}")
        plt.ylim([0.0,0.012])
        plt.grid()
        plt.xticks(x, file_sizes, rotation=90)
        plt.ylabel("latency [s]")
        plt.xlabel("data size")
    plt.show()


def make_boxplot(meas_data, dds, RT="ON", LD="OFF", FQ_SCL=None):
    num = len(meas_data[0])
    # x = [256*pow(2,i) for i in range(15)]
    x = range(1, 16)
    fig = plt.figure(figsize=[10,7.5])
    plt.boxplot(meas_data, showfliers=False)
    plt.xticks(x, file_sizes)
    plt.ylabel("latency [s]")
    plt.xlabel("data size")
    if (FQ_SCL!=None):
        plt.title(f"{dds} : EVAL_NUM={str(num)}, RT={RT}, CPU LD={LD}, Fq Scl={FQ_SCL}")
    else:
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

def plot_errorbars_q7(nums):

    plt.subplot(1,2,1)
    for num in nums:
        dat = load_meas_data_q7(num)
        new_data = [[0] * num for _ in range(15)]

        for i in range(15):
            for j in range(num):
                new_data[i][j] = (dat[j][i])

        x = range(1,16)
        means = []
        stds = []
        for i in range(15):
            print(np.array(new_data).shape)
            mean,std = norm.fit(new_data[i])
            means.append(mean)
            stds.append(std)

        plt.errorbar(x, means, stds)
    plt.xticks(x, file_sizes)
    plt.ylabel("latency [s]")
    plt.xlabel("data size")
    plt.title("Mean and std. of latencies of all listening nodes for \n different datasizes and different no. of multiple listeners  ")
    plt.legend([f'{nums[0]} listeners', f'{nums[1]} listeners', f'{nums[2]} listeners', f'{nums[3]} listeners', f'{nums[4]} listeners'])
    plt.grid()
    # plt.show()


    plt.subplot(1,2,2)
    plot_errorbars_q7_zoomed(nums)
    plt.show()

    # print(new_data[0])
    # print(f"mean {means[1]} std {stds[1]}")
    # plt.boxplot(new_data[1])
    # plt.show()
def plot_errorbars_q7_zoomed(nums):

    for num in nums:
        dat = load_meas_data_q7(num)
        new_data = [[0] * num for _ in range(1,10)]

        for i in range(1,10):
            for j in range(num):
                new_data[i-1][j] = (dat[j][i])

        x = range(1,10)
        means = []
        stds = []
        for i in range(9):
            # print(np.array(new_data).shape)
            mean,std = norm.fit(new_data[i])
            means.append(mean)
            stds.append(std)

        plt.errorbar(x, means, stds)
    plt.xticks(x, file_sizes[1:10])
    plt.ylabel("latency [s]")
    plt.xlabel("data size")
    plt.title("Zoomed in for datasizes between 512B and 128kB ")
    plt.legend([f'{nums[0]} listeners', f'{nums[1]} listeners', f'{nums[2]} listeners', f'{nums[3]} listeners', f'{nums[4]} listeners'])
    plt.grid()
    # plt.show()

def plot_errorbars_q7_pernode(num = 10):
    plt.subplot(1,2,1)
    data = load_meas_data_q7(num)
    x = range(2,10)
    for n in range(10):
        means = []
        stds = []
        for i in range(1,9):
            print(np.array(data).shape)
            # mean,std = norm.fit(data[n][i])
            mean = np.median(data[n][i])
            std = np.std(data[n][i])
            means.append(mean)
            stds.append(std)

        plt.errorbar(x, means, stds)
    plt.xticks(x, file_sizes[1:9])
    plt.ylabel("latency [s]")
    plt.xlabel("data size")
    plt.title("Median and std. of latencies of individual nodes for \n 10 active listeners (data size < 65kB) ")
    plt.legend(['node 1', 'node 2', 'node 3', 'node 4', 'node 5', 'node 6', 'node 7', 'node 8', 'node 9', 'node 10'])
    plt.grid()

    plt.subplot(1,2,2)
    x = range(10, 16)
    for n in range(10):
        means = []
        stds = []
        for i in range(9,15):
            print(np.array(data).shape)
            # mean, std = norm.fit(data[n][i])
            mean = np.median(data[n][i])
            std = np.std(data[n][i])
            means.append(mean)
            stds.append(std)

        plt.errorbar(x, means, stds)
    plt.xticks(x, file_sizes[9:15])
    plt.ylabel("latency [s]")
    plt.xlabel("data size")
    plt.title(
        "Median and std. of latencies of individual nodes for \n 10 active listeners (data size > 65kB) ")
    plt.legend(
        ['node 1', 'node 2', 'node 3', 'node 4', 'node 5', 'node 6', 'node 7', 'node 8', 'node 9', 'node 10'])
    plt.grid()
    plt.show()



def plot_mean_std_medians(num, data):

    medians = np.ndarray([num,15])
    for i in range(num):
        for j in range(15):
            medians[i,j] = (np.median(data[i][j]))
    # plt.boxplot(medians, showfliers=False)

    medians= medians.T
    means = []
    stds = []
    print(medians.shape)
    for i in range(15):
        mean,std= norm.fit(medians[i])
        means.append(mean)
        stds.append(std)
    x = range(1,16)
    plt.errorbar(x,means,stds)
    plt.xticks(x, file_sizes)
    plt.ylabel("latency [s]")
    plt.xlabel("data size")
    plt.title("Mean and std of median latencies of 10 simultaneously \n active listener nodes for different data sizes")
    plt.grid()
    plt.show()
