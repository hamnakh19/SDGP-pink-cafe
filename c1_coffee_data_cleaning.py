import pandas as pd

# Load coffee sales CSV (skip the product-name row only)
df = pd.read_csv(
    "raw_data/coffee_sales.csv",
    skiprows=[1]
)

# Rename columns FIRST
df = df.rename(columns={
    "Date": "date",
    "Number Sold": "cappuccino_sold",
    "Unnamed: 2": "americano_sold"
})

print(df.head())

# Convert data types
df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
df["cappuccino_sold"] = pd.to_numeric(df["cappuccino_sold"], errors="coerce")
df["americano_sold"] = pd.to_numeric(df["americano_sold"], errors="coerce")

# Drop invalid rows
df = df.dropna()

# Split into two clean datasets
cappuccino_df = df[["date", "cappuccino_sold"]].rename(
    columns={"cappuccino_sold": "number_sold"}
)

americano_df = df[["date", "americano_sold"]].rename(
    columns={"americano_sold": "number_sold"}
)

# Save cleaned files
cappuccino_df.to_csv("cleaned_data/cappuccino_cleaned.csv", index=False)
americano_df.to_csv("cleaned_data/americano_cleaned.csv", index=False)

print("Coffee data cleaned and saved successfully.")
