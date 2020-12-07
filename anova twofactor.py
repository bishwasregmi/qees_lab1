# load packages
import pandas as pd
import seaborn as sns
# load data file
d = pd.read_csv("measurements/", sep="\t")


d_melt = pd.melt(d, id_vars=['data'], value_vars=['fastrtps', 'opensplice', 'connext'])
