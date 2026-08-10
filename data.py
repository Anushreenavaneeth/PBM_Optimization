import pandas as pd

# Read the Orange Book TXT file
df = pd.read_csv("products.txt", sep="~", dtype=str)

# Save as CSV
df.to_csv("orange_book_products.csv", index=False)

print("Conversion completed!")
print("Rows:", len(df))
print("Columns:", len(df.columns))