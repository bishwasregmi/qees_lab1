# load packages
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.graphics.factorplots import interaction_plot
import matplotlib.pyplot as plt
from scipy import stats
# load data file
data = pd.read("measurements/", sep="\t")



d_melt = pd.melt(d, id_vars=['data'], value_vars=['fastrtps', 'opensplice', 'connext'])
