from python_scripts.definitions import *

num = 10
data = load_meas_data_q7(num)

# plot_mean_std_medians(num,data)

# plot_data = []
# for i in range(num):
#     plot_data.append(data[i][0])
# for i in range(num):
#     plot_data.append(data[i][1])
# for i in range(num):
#     plot_data.append(data[i][2])

# plt.boxplot(plot_data, showfliers=False)
# plt.show()

# for i in range(2):
#     plt.boxplot(data[i], showfliers=False)
#     plt.title(f"node {i+1}")
#     plt.show()

# make_boxplot_q7(num, data)

# print(np.array(data).shape )



# nums = [2,4,6,8,10]
# plot_errorbars_q7(nums)
# plot_errorbars_q7_zoomed(nums)


plot_errorbars_q7_pernode()
