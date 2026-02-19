import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------
# LOAD DATA
# -------------------------------

americano = pd.read_csv("cleaned_data/americano_cleaned.csv")
cappuccino = pd.read_csv("cleaned_data/cappuccino_cleaned.csv")
croissants = pd.read_csv("cleaned_data/croissants_cleaned.csv")

# -------------------------------
# CONVERT DATE
# -------------------------------

americano["date"] = pd.to_datetime(americano["date"])
cappuccino["date"] = pd.to_datetime(cappuccino["date"])
croissants["date"] = pd.to_datetime(croissants["date"])

# -------------------------------
# EXTRACT WEEKDAY
# -------------------------------

americano = americano[americano["date"] < "2025-10-01"]
cappuccino = cappuccino[cappuccino["date"] < "2025-10-01"]
croissants = croissants[croissants["date"] < "2025-10-01"]


americano["weekday"] = americano["date"].dt.day_name()
cappuccino["weekday"] = cappuccino["date"].dt.day_name()
croissants["weekday"] = croissants["date"].dt.day_name()

weekday_order = [
    "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday", "Sunday"
]

# -------------------------------
# GROUP BY WEEKDAY (AVERAGE)
# -------------------------------

amer_weekday = americano.groupby("weekday")["number_sold"].mean().reindex(weekday_order)
capp_weekday = cappuccino.groupby("weekday")["number_sold"].mean().reindex(weekday_order)
crois_weekday = croissants.groupby("weekday")["number_sold"].mean().reindex(weekday_order)

# -------------------------------
# PLOT
# -------------------------------

x = np.arange(len(weekday_order))
width = 0.25

plt.figure(figsize=(12,6))

plt.bar(x - width, amer_weekday, width=width,
        color="#1f77b4", label="Americano")

plt.bar(x, capp_weekday, width=width,
        color="#ff7f0e", label="Cappuccino")

plt.bar(x + width, crois_weekday, width=width,
        color="#ff69b4", label="Croissants")

plt.xticks(x, weekday_order, rotation=30)
plt.ylabel("Average Units Sold")
plt.title("Average Sales by Day of Week (March–September 2025)")


plt.ylim(0, 110)  # adjust max if needed
plt.yticks(np.arange(0, 111, 10))


# ----- Styling -----
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)

plt.legend()
plt.tight_layout()
plt.show(block=True)
