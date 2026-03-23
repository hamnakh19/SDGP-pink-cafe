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
americano["date"] = pd.to_datetime(americano["date"])
cappuccino["date"] = pd.to_datetime(cappuccino["date"])
croissants["date"] = pd.to_datetime(croissants["date"])

# REMOVE OCTOBER
americano = americano[americano["date"].dt.month <= 9]
cappuccino = cappuccino[cappuccino["date"].dt.month <= 9]
croissants = croissants[croissants["date"].dt.month <= 9]

# EXTRACT MONTH
americano["month"] = americano["date"].dt.to_period("M")
cappuccino["month"] = cappuccino["date"].dt.to_period("M")
croissants["month"] = croissants["date"].dt.to_period("M")

# GROUP
amer_monthly = americano.groupby("month")["number_sold"].sum()
capp_monthly = cappuccino.groupby("month")["number_sold"].sum()
crois_monthly = croissants.groupby("month")["number_sold"].sum()

# Convert PeriodIndex to datetime for formatting
months = pd.to_datetime(amer_monthly.index.astype(str))
month_labels = months.strftime("%b")  # Mar, Apr, May...

print(crois_monthly)

# -------------------------------
# PLOT
# -------------------------------
x = np.arange(len(month_labels))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 6))

# Create bars
bars1 = ax.bar(x - width, amer_monthly, width=width,
               color="#1f77b4", label="Americano")
bars2 = ax.bar(x, capp_monthly, width=width,
               color="#ff7f0e", label="Cappuccino")
bars3 = ax.bar(x + width, crois_monthly, width=width,
               color="#ff69b4", label="Croissants")

# ADD DATA LABELS on top of bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(month_labels)
ax.set_ylabel("Total Units Sold")
ax.set_title("Monthly Sales Comparison (March–September 2025)")

# Clean grid
ax.grid(axis="y", linestyle="--", alpha=0.4)

# Remove top and right borders
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend()
plt.tight_layout()
plt.show(block=True)