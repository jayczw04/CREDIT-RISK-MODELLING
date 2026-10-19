import pandas as pd

chunks = pd.read_csv("accepted_2007_to_2018Q4.csv.gz", chunksize=200_000, low_memory=False)
sample = pd.concat(c.sample(frac=0.03, random_state=42) for c in chunks)

sample.to_csv("loans_sample.csv.gz", index=False, compression="gzip")
print(sample.shape)