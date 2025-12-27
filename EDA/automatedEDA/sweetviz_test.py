import pandas as pd
import sweetviz as sv
import numpy as np
print("SweetViz Version : {}".format(sv.__version__))
df=pd.read_csv("http://nucsmb/csv/data/small_csv_files/BuildingConsentsByInstitutionalSectorMonthly.csv")
print(df.head())
print(df.info())
print(df.describe())
report = sv.analyze([df, 'Train'], target_feat='Data_value')
report.show_html("BuildingConsentsByInstitutionalSectorMonthly_sweetviz_report.html")