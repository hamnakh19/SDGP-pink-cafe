import pandas as pd
print("Script started")


def analyse_product(file_path, product_name):
    print(f"\n===== {product_name} ANALYSIS =====")

    # Load dataset
    df = pd.read_csv(file_path)
    df["date"] = pd.to_datetime(df["date"])

    # Basic statistics
    highest_sale = df["number_sold"].max()
    lowest_sale = df["number_sold"].min()
    mean_sale = df["number_sold"].mean()
    median_sale = df["number_sold"].median()
    std_dev = df["number_sold"].std()
    total_sales = df["number_sold"].sum()
    num_days = df["number_sold"].count()

    # Dates of highest and lowest
    highest_date = df.loc[df["number_sold"].idxmax(), "date"]
    lowest_date = df.loc[df["number_sold"].idxmin(), "date"]

    # Print results
    print(f"Number of days analysed: {num_days}")
    print(f"Total sales: {total_sales}")
    print(f"Highest sale: {highest_sale} on {highest_date.date()}")
    print(f"Lowest sale: {lowest_sale} on {lowest_date.date()}")
    print(f"Mean (average) sales: {mean_sale:.2f}")
    print(f"Median sales: {median_sale}")
    print(f"Standard deviation (volatility): {std_dev:.2f}")

# Run analysis for each dataset
analyse_product("cleaned_data/americano_cleaned.csv", "Americano")
analyse_product("cleaned_data/cappuccino_cleaned.csv", "Cappuccino")
analyse_product("cleaned_data/croissants_cleaned.csv", "Croissants")
