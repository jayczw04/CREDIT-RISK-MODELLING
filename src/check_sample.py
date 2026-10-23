from pathlib import Path
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
df = pd.read_csv(DATA / "loans_sample.csv.gz", low_memory=False)

##print(df.dtypes.value_counts())                      # how many columns of each type?
print(df.select_dtypes(include="str").columns.tolist())   # which columns are text?
##print(df[["term", "emp_length", "issue_d"]].head())  # peek at a few text columns

print(df["open_acc"].isna().sum())