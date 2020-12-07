# load packages
import pandas as pd
from statsmodels.formula.api import ols
import statsmodels.stats.multicomp
from statsmodels.stats.anova import anova_lm
import seaborn as sns
import researchpy as rp
import statsmodels.api as sm
from scipy.stats import t
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.graphics.factorplots import interaction_plot
import matplotlib.pyplot as plt
from scipy import stats
from python_scripts.definitions import *
# load data file
# file_object = open("measurements/transport_time_120_annova analysis", "r")
# data = file_object.readlines()
# print(data)

latency = np.array(load_meas_data_q4())
latency = latency.flatten('C')
print(f"latency shape {latency.shape}")

df = pd.DataFrame({'latency' : latency,
                   'data_size' : np.repeat(['256B', '128kB'], 1500),
                   'dds' : np.r_[np.repeat(['fastrtps', 'opensplice', 'connext'],500),
                                 np.repeat(['fastrtps', 'opensplice', 'connext'],500)]})
# sns.boxplot(x = 'dds', y = 'latency', hue= 'data_size', data = df, showfliers=False)
# plt.grid()
# plt.show()
print(f"DataFrame summary:\n{rp.summary_cont(df.groupby(['data_size', 'dds']))['latency']}")
# print(df)

model = ols('latency ~ C(data_size)*C(dds)', df).fit()

print(f"Overall model: F({model.df_model: .0f},{model.df_resid: .0f}) = {model.fvalue: .3f}, p = {model.f_pvalue: .4f}")

print(f"Model summary:\n {model.summary()}")

res = sm.stats.anova_lm(model, typ= 2)
print(f"Anova table:\n {res}")



# mc = statsmodels.stats.multicomp.MultiComparison(df['latency'],df['dds'])
# mc_results = mc.tukeyhsd()
# print(mc_results)
#
# mc = statsmodels.stats.multicomp.MultiComparison(df['latency'],df['data_size'])
# mc_results = mc.tukeyhsd()
# print(mc_results)


# ------------CALCULATING CONFIDENCE INTERVALS FOR DDSes-----------


latency = load_meas_data_q4()
fastrtps = np.concatenate((latency[0][0], latency[1][0]))
opensplice = np.concatenate((latency[0][1], latency[1][1]))
connext = np.concatenate((latency[0][2], latency[1][2]))

dds_meas = [fastrtps, opensplice, connext]

dds_mean = np.mean(dds_meas)

ssr = res['sum_sq'][3]                  # error variance(ssr) taken from the anova table

dds_std = ssr*np.sqrt(((3-1)/(3*2*500)))   # s_alpha = ssr * sqrt((a-1)/abr)



for i,data in enumerate(dds_meas):
    alpha = 0.95
    dof = len(data) - 1
    mu = np.mean(data)
    std = np.std(data)
    sem = std / np.sqrt(len(data))

    t_val = abs(t.ppf((1 - alpha) / 2, df=dof))

    c = [mu - t_val * sem, mu + t_val * sem]

    print(f"Confidence interval {i} = {c}")


