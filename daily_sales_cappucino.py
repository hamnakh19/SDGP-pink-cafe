import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Load dataset
df = pd.read_csv("cleaned_data/cappuccino_cleaned.csv")
df["date"] = pd.to_datetime(df["date"])

plt.figure(figsize=(12, 6))

# Warm cappuccino-style orange
plt.plot(df["date"], df["number_sold"],
         color="#C56A1A",   # warm brown-orange
         linewidth=1.8)

plt.title("Daily Cappuccino Sales (March–October 2025)")
plt.xlabel("Date")
plt.ylabel("Units Sold")

plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

plt.xticks(rotation=45)
plt.grid(True, linestyle="--", alpha=0.25)


plt.tight_layout()
plt.show()
