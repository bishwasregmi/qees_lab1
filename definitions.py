import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os
from pathlib import Path
from numpy import loadtxt

main_dir_path = Path(__file__).resolve().parents[1]
meas_path = main_dir_path.joinpath("evaluation/transport_time")
intr_proc_lstnr_path = main_dir_path.joinpath("src/interprocess_eval/src/listener_interprocess.cpp")
intr_proc_tlkr_path = main_dir_path.joinpath("src/interprocess_eval/src/talker_interprocess.cpp")

file_sizes = ['256B', '512B', '1kB', '2kB', '4kB', '8kB', '16kB', '32kB', '64kB', '128kB', '256kB', '512kB', '1MB',
              '2MB', '4MB']
meas_data = [0] * 15
index = {
    'transport_time_256Kbyte.txt': 0,
    'transport_time_512byte.txt': 1,
    'transport_time_1Kbyte.txt': 2,
    'transport_time_2Kbyte.txt': 3,
    'transport_time_4Kbyte.txt': 4,
    'transport_time_8Kbyte.txt': 5,
    'transport_time_16Kbyte.txt': 6,
    'transport_time_32Kbyte.txt': 7,
    'transport_time_64Kbyte.txt': 8,
    'transport_time_128Kbyte.txt': 9,
    'transport_time_256byte.txt': 10,
    'transport_time_512Kbyte.txt': 11,
    'transport_time_1Mbyte.txt': 12,
    'transport_time_2Mbyte.txt': 13,
    'transport_time_4Mbyte.txt': 14
}


# check EVAL_NUM is the same as set value. If not, change it and rebuild
def set_EVAL_NUM(path1, path2, value):
    rebuild = False
    with open(path1, 'r') as f:  # check in listener
        lines = f.readlines()
        for i, line in enumerate(lines):
            if "#define EVAL_NUM" in line:
                curr_value = [int(s) for s in line.split() if s.isdigit()][0]
                if curr_value != value:
                    rebuild = True
                    lines[i] = "#define EVAL_NUM " + str(value) + " \n"
                    print(f"In {path1} line {i}: {line} changed to {lines[i]}")
    with open(path1, 'w') as f:
        f.writelines(lines)

    with open(path2, 'r') as f:  # check in talker
        lines = f.readlines()
        for i, line in enumerate(lines):
            if "#define EVAL_NUM" in line:
                curr_value = [int(s) for s in line.split() if s.isdigit()][0]
                if curr_value != value:
                    rebuild = True
                    lines[i] = "#define EVAL_NUM " + str(value) + " \n"
                    print(f"In {path2} line {i}: {line} changed to {lines[i]}")
    with open(path2, 'w') as f:
        f.writelines(lines)

    if rebuild == True:
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


def run_meas():
    os.chdir(main_dir_path)
    cmd1 = "ros2 run interprocess_eval listener_interprocess__rmw_fastrtps_cpp"
    cmd2 = "ros2 run interprocess_eval talker_interprocess__rmw_fastrtps_cpp"
    # os.system("gnome-terminal --title=newWindow -- bash -c  'bash'" )
    # process = os.system("gnome-terminal --title=newWindow -- bash -c  'ros2 run interprocess_eval talker_interprocess__rmw_fastrtps_cpp; bash'")
    # process.wait()

    answer = " "
    while answer != "yes" and answer != "no":
        answer = input("Are you sure you want to clean eval-times? \n Type 'yes' or 'no' : \n")
    if answer == 'yes':
        clean_eval_time()

    while answer != "Done" and answer != "no":
        answer = input("Run the ros2 nodes and type 'Done' to continue : \n")

    subprocess.run("./calculate", shell=True)


def make_boxplot(num):
    meas_files = os.listdir(meas_path)
    for file in meas_files:
        try:
            meas_data[index[file]] = loadtxt(os.path.join(meas_path, file), dtype=np.float64)
        except:
            pass

    # x = [256*pow(2,i) for i in range(15)]
    x = range(1, 16)
    plt.boxplot(meas_data)
    plt.xticks(x, file_sizes)
    plt.ylabel("latency [ms]")
    plt.xlabel("data size")
    plt.title("EVAL_NUM = " + str(num))
    plt.show()
    # plt.savefig('q1_' + str(EVAL_NUM) + ".png")
