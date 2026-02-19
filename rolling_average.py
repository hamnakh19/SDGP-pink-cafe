import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Load datasets
americano = pd.read_csv("cleaned_data/americano_cleaned.csv")
cappuccino = pd.read_csv("cleaned_data/cappuccino_cleaned.csv")
croissants = pd.read_csv("cleaned_data/croissants_cleaned.csv")

# Convert date columns
americano["date"] = pd.to_datetime(americano["date"])
cappuccino["date"] = pd.to_datetime(cappuccino["date"])
croissants["date"] = pd.to_datetime(croissants["date"])

# Calculate rolling averages
americano["rolling_7"] = americano["number_sold"].rolling(7).mean()
cappuccino["rolling_7"] = cappuccino["number_sold"].rolling(7).mean()
croissants["rolling_7"] = croissants["number_sold"].rolling(7).mean()

plt.figure(figsize=(12, 6))
plt.plot(americano["date"], americano["rolling_7"],
         color="#2F5D8C", linewidth=2, label="Americano")
plt.plot(cappuccino["date"], cappuccino["rolling_7"],
         color="#C56A1A", linewidth=2, label="Cappuccino")
plt.plot(croissants["date"], croissants["rolling_7"],
         color="#FF69B4", linewidth=2, label="Croissants")

plt.title("7-Day Rolling Average Sales Comparison (Mar–Oct 2025)")
plt.xlabel("Date")
plt.ylabel("Units Sold")
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # Changed here
plt.xticks(rotation=0)  # No rotation needed now
plt.grid(True, linestyle="--", alpha=0.25)
plt.legend()
plt.tight_layout()
plt.show(block=True)