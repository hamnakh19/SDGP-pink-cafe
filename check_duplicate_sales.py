import pandas as pd

# Load cleaned datasets
croissants_df = pd.read_csv("cleaned_data/croissants_cleaned.csv")
cappuccino_df = pd.read_csv("cleaned_data/cappuccino_cleaned.csv")

# Merge on date
merged = croissants_df.merge(
    cappuccino_df,
    on="date",
    suffixes=("_croissant", "_cappuccino")
)

# Check how often values are exactly the same
same_ratio = (merged["number_sold_croissant"] == merged["number_sold_cappuccino"]).mean()

print("Proportion of days with identical sales:", same_ratio)
