import pandas as pd

# Load croissant sales data
df = pd.read_csv("raw_data/croissant_sales.csv")

print(df.head())
print(df.columns)

# Rename columns to standard format
df = df.rename(columns={
    "Date": "date",
    "Number Sold": "number_sold"
})

# Convert data types
df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
df["number_sold"] = pd.to_numeric(df["number_sold"], errors="coerce")

# Drop invalid rows
df = df.dropna()

# Sort by date (VERY IMPORTANT for time series)
df = df.sort_values("date")

# Save cleaned data
df.to_csv("cleaned_data/croissants_cleaned.csv", index=False)

print("Croissant data cleaned and saved successfully.")
