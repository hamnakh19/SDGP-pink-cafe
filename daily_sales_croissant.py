import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("cleaned_data/croissants_cleaned.csv")

# Convert date column
df["date"] = pd.to_datetime(df["date"])

# Create figure
plt.figure(figsize=(12, 6))

# Plot pink line
plt.plot(df["date"], df["number_sold"],
         color="hotpink",
         linewidth=2)

plt.ylim(20, 100)
plt.yticks(range(20, 101, 10))

# Labels
plt.xlabel("Date")
plt.ylabel("Units Sold")
plt.title("Daily Croissant Sales (March–October 2025)")

plt.grid(True, linestyle="--", alpha=0.4)
plt.xticks(rotation=45)

plt.ylim(0, 100)
plt.yticks(range(0, 101, 10))

plt.tight_layout()
plt.show()
