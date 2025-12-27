import pandas as pd
df=pd.read_csv("http://nucsmb/csv/data/small_csv_files/BuildingConsentsByInstitutionalSectorMonthly.csv")
from ydata_profiling import ProfileReport
profile = ProfileReport(df, title="Profiling Report")
profile.to_notebook_iframe()  # For Jupyter Notebooks
profile.to_file("BuildingConsentsByInstitutionalSectorMonthly_y-data_report.html")  # Save as HTML file