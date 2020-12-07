# load packages
import pandas as pd
from statsmodels.formula.api import ols
import statsmodels.stats.multicomp
from statsmodels.stats.anova import anova_lm
import seaborn as sns
import researchpy as rp
import statsmodels.api as sm
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
sns.boxplot(x = 'dds', y = 'latency', hue= 'data_size', data = df, showfliers=False)
plt.show()
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


latency = np.array(load_meas_data_q4())
