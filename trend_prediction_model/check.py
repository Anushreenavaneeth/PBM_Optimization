import pandas as pd

unique_drugs = set()
chunk_size = 500_000

for chunk in pd.read_csv(
    "dataset/2019/data_2019.csv",
    usecols=["Gnrc_Name"],
    chunksize=chunk_size,
    encoding="latin1",   # remember this fix from the earlier crash
):
    unique_drugs.update(chunk["Gnrc_Name"].dropna().unique())

print(f"Total unique drugs in raw file: {len(unique_drugs)}")