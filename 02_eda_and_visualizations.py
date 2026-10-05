"""Step 2: Exploratory Data Analysis + Visualizations for the logistics dataset."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({"figure.dpi": 110, "axes.titleweight": "bold"})
OUT = "charts/"

df = pd.read_csv("logistics_data.csv", parse_dates=["order_date"])
df["month"] = df.order_date.dt.month

# ---------- Data quality ----------
print("SHAPE:", df.shape)
print("\nMISSING VALUES:\n", df.isna().sum()[df.isna().sum() > 0])
df["warehouse_hours"] = df.warehouse_hours.fillna(df.warehouse_hours.median())

# Outlier handling: IQR rule applied WITHIN each transport mode (air freight is legitimately costlier than sea)
def iqr_flag(s):
    q1, q3 = s.quantile([.25, .75]); i = q3 - q1
    return s > q3 + 3 * i
df["cost_outlier"] = df.groupby("transport_mode")["cost_per_km"].transform(iqr_flag)
n_out = int(df.cost_outlier.sum())
print(f"\nCOST-PER-KM EXTREME OUTLIERS (3xIQR within mode): {n_out}")
clean = df[~df.cost_outlier].drop(columns="cost_outlier").copy()
print("ROWS AFTER OUTLIER REMOVAL:", len(clean))

# ---------- Descriptive statistics ----------
num_cols = ["distance_km", "weight_kg", "shipment_volume_m3", "warehouse_hours",
            "planned_days", "delay_days", "delivery_days", "transport_cost", "fuel_cost", "cost_per_km"]
desc = clean[num_cols].describe().T
desc["median"] = clean[num_cols].median()
desc["skew"] = clean[num_cols].skew()
print("\nDESCRIPTIVE STATISTICS:\n", desc.round(2).to_string())

print("\nON-TIME RATE: %.1f%%" % (clean.on_time.mean() * 100))
print("AVG DELAY (days): %.2f" % clean.delay_days.mean())

# ---------- Group summaries ----------
by_mode = clean.groupby("transport_mode").agg(shipments=("shipment_id", "count"), avg_cost=("transport_cost", "mean"),
        avg_cost_km=("cost_per_km", "mean"), avg_delay=("delay_days", "mean"), on_time=("on_time", "mean")).round(2)
by_carrier = clean.groupby("carrier").agg(shipments=("shipment_id", "count"), avg_delay=("delay_days", "mean"),
        on_time=("on_time", "mean"), avg_cost=("transport_cost", "mean"), rating=("customer_rating", "mean")).round(2)
by_region = clean.groupby("region").agg(shipments=("shipment_id", "count"), avg_delay=("delay_days", "mean"),
        on_time=("on_time", "mean"), avg_cost=("transport_cost", "mean")).round(2)
by_weather = clean.groupby("weather").agg(shipments=("shipment_id", "count"), avg_delay=("delay_days", "mean"),
        on_time=("on_time", "mean")).round(2)
by_priority = clean.groupby("priority").agg(shipments=("shipment_id", "count"), avg_cost=("transport_cost", "mean"),
        avg_delay=("delay_days", "mean")).round(2)
monthly = clean.groupby("month").agg(shipments=("shipment_id", "count"), total_volume=("weight_kg", "sum"),
        avg_delay=("delay_days", "mean"), on_time=("on_time", "mean"), avg_cost=("transport_cost", "mean")).round(2)
for name, t in [("MODE", by_mode), ("CARRIER", by_carrier), ("REGION", by_region),
                ("WEATHER", by_weather), ("PRIORITY", by_priority), ("MONTHLY", monthly)]:
    print(f"\nBY {name}:\n", t.to_string())

# ---------- Correlations ----------
corr = clean[["distance_km", "weight_kg", "warehouse_hours", "delay_days", "delivery_days",
              "transport_cost", "fuel_cost", "customer_rating"]].corr()
print("\nCORRELATION MATRIX:\n", corr.round(2).to_string())

# ---------- Visualizations ----------
# 1. Histogram + KDE: delivery time distribution
fig, ax = plt.subplots(figsize=(8, 4.5))
sns.histplot(clean.delivery_days, bins=35, kde=True, ax=ax, color="#2a6fbb")
ax.axvline(clean.delivery_days.mean(), color="red", ls="--", label=f"Mean = {clean.delivery_days.mean():.2f}")
ax.axvline(clean.delivery_days.median(), color="green", ls=":", label=f"Median = {clean.delivery_days.median():.2f}")
ax.set(title="Distribution of Delivery Time", xlabel="Delivery time (days)", ylabel="Shipments"); ax.legend()
fig.tight_layout(); fig.savefig(OUT + "01_delivery_time_distribution.png"); plt.close()

# 2. Box plot: cost per km by transport mode
fig, ax = plt.subplots(figsize=(8, 4.5))
sns.boxplot(data=clean, x="transport_mode", y="cost_per_km", order=["Sea", "Rail", "Road", "Air"], ax=ax)
ax.set(title="Cost per Kilometre by Transport Mode", xlabel="Transport mode", ylabel="Cost per km")
fig.tight_layout(); fig.savefig(OUT + "02_cost_per_km_by_mode.png"); plt.close()

# 3. Line chart: monthly shipment volume and on-time rate (dual axis)
fig, ax1 = plt.subplots(figsize=(8, 4.5))
ax1.bar(monthly.index, monthly.shipments, color="#9ecae1", label="Shipments")
ax1.set(xlabel="Month (2025)", ylabel="Number of shipments", title="Monthly Shipments vs On-Time Delivery Rate")
ax1.set_xticks(range(1, 13))
ax2 = ax1.twinx()
ax2.plot(monthly.index, monthly.on_time * 100, color="crimson", marker="o", label="On-time %")
ax2.set_ylabel("On-time rate (%)"); ax2.grid(False)
h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="lower left")
fig.tight_layout(); fig.savefig(OUT + "03_monthly_trend.png"); plt.close()

# 4. Correlation heatmap
fig, ax = plt.subplots(figsize=(7.5, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True, ax=ax)
ax.set_title("Correlation Heatmap of Key Logistics Variables")
fig.tight_layout(); fig.savefig(OUT + "04_correlation_heatmap.png"); plt.close()

# 5. Scatter: distance vs cost, coloured by mode
fig, ax = plt.subplots(figsize=(8, 4.8))
sns.scatterplot(data=clean, x="distance_km", y="transport_cost", hue="transport_mode", alpha=0.6, s=25, ax=ax)
ax.set(title="Transportation Cost vs Distance", xlabel="Distance (km)", ylabel="Transport cost")
fig.tight_layout(); fig.savefig(OUT + "05_distance_vs_cost.png"); plt.close()

# 6. Bar chart: on-time rate by carrier
fig, ax = plt.subplots(figsize=(8, 4.5))
order = by_carrier.sort_values("on_time", ascending=False).index
sns.barplot(x=by_carrier.loc[order].index, y=by_carrier.loc[order].on_time * 100, ax=ax, hue=by_carrier.loc[order].index, legend=False)
for i, v in enumerate(by_carrier.loc[order].on_time * 100):
    ax.text(i, v + 0.8, f"{v:.1f}%", ha="center")
ax.set(title="On-Time Delivery Rate by Carrier", xlabel="Carrier", ylabel="On-time rate (%)", ylim=(0, 100))
fig.tight_layout(); fig.savefig(OUT + "06_ontime_by_carrier.png"); plt.close()

# 7. Heatmap: average delay by region x weather
pivot = clean.pivot_table(index="region", columns="weather", values="delay_days", aggfunc="mean")[["Clear", "Rain", "Fog", "Storm"]]
fig, ax = plt.subplots(figsize=(7.5, 4.5))
sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax)
ax.set_title("Average Delay (days): Region x Weather")
fig.tight_layout(); fig.savefig(OUT + "07_delay_region_weather.png"); plt.close()

# 8. Pareto: delay by weather contribution & cost drivers (feature importance via correlation with cost)
cost_corr = clean[["distance_km", "weight_kg", "warehouse_hours", "delay_days", "fuel_cost"]].corrwith(clean.transport_cost).sort_values()
fig, ax = plt.subplots(figsize=(8, 4.2))
cost_corr.plot(kind="barh", ax=ax, color="#31a354")
ax.set(title="Correlation of Variables with Transport Cost", xlabel="Pearson correlation")
fig.tight_layout(); fig.savefig(OUT + "08_cost_drivers.png"); plt.close()

# 9. Violin: delay by priority
fig, ax = plt.subplots(figsize=(8, 4.5))
sns.violinplot(data=clean, x="priority", y="delay_days", order=["Standard", "Express", "Critical"], inner="quartile", ax=ax, hue="priority", legend=False)
ax.set(title="Delay Distribution by Shipment Priority", xlabel="Priority", ylabel="Delay (days)")
fig.tight_layout(); fig.savefig(OUT + "09_delay_by_priority.png"); plt.close()

# 10. Customer rating vs delay
fig, ax = plt.subplots(figsize=(8, 4.5))
sns.boxplot(data=clean, x="customer_rating", y="delay_days", ax=ax)
ax.set(title="Customer Rating vs Delay", xlabel="Customer rating (1-5)", ylabel="Delay (days)")
fig.tight_layout(); fig.savefig(OUT + "10_rating_vs_delay.png"); plt.close()

print("\nAll charts saved to", OUT)
