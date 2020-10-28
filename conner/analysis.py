import pandas as pd
from sqlalchemy import create_engine
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

CATEGORICAL = ["NWCG_REPORTING_AGENCY", "FIRE_SIZE_CLASS", "STATE"]
FEATURES = ["FOD_ID", "NWCG_REPORTING_AGENCY", "FIRE_YEAR", "DISCOVERY_DOY", "STAT_CAUSE_CODE", "CONT_DOY", "FIRE_SIZE", "LATITUDE", "LONGITUDE", "OWNER_CODE"]

engine = create_engine("sqlite:///wildfires.sqlite")
con = engine.connect()
wildfires = pd.read_sql_table("Fires", con, columns=FEATURES + CATEGORICAL)
# wildfires = pd.get_dummies(wildfires, columns=["NWCG_REPORTING_AGENCY"])
#print(wildfires["NWCG_REPORTING_AGENCY"].nunique())
corr_matrix = wildfires.corr()
print(corr_matrix)
sn.heatmap(corr_matrix, annot=True)
plt.show()    
pca = PCA(0.95)
pca.fit(wildfires)
wildfires = pca.transform(wildfires)
principalDf = pd.DataFrame(data = wildfires
             , columns = ['principal component 1', 'principal component 2'])
# PCA bad if large number of variables, need to also scale some elements
# wildfires.plot(kind='scatter',x='LONGITUDE',y='LATITUDE',color='coral',alpha=0.3)
# plt.show()
# Redo correlation matrix by mapping non-numeric data to numeric data
# print(wildfires.info())
# https://pandas.pydata.org/docs/user_guide/10min.html
# https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf
# https://datatofish.com/correlation-matrix-pandas/
# https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9