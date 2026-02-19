import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("cleaned_data/americano_cleaned.csv")

# Convert date column
df["date"] = pd.to_datetime(df["date"])

plt.figure(figsize=(12, 6))

plt.plot(df["date"], df["number_sold"],
         color="#1f77b4",
         linewidth=2)

plt.xlabel("Date")
plt.ylabel("Units Sold")
plt.title("Daily Americano Sales (March–October 2025)")

plt.grid(True, linestyle="--", alpha=0.4)
plt.xticks(rotation=45)


plt.tight_layout()
plt.show()
