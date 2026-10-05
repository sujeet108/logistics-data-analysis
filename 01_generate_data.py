"""Step 1: Simulate a hypothetical logistics dataset (2,000 shipments, 2025)."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 2000

regions = ["North", "South", "East", "West", "Central"]
carriers = ["SwiftMove", "RoadKing", "CargoPro", "BlueLine"]
modes = ["Road", "Rail", "Air", "Sea"]
priorities = ["Standard", "Express", "Critical"]
weather = ["Clear", "Rain", "Fog", "Storm"]

df = pd.DataFrame({
    "shipment_id": [f"SHP{100000 + i}" for i in range(N)],
    "order_date": pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, N), unit="D"),
    "region": rng.choice(regions, N, p=[0.2, 0.2, 0.2, 0.25, 0.15]),
    "carrier": rng.choice(carriers, N, p=[0.3, 0.3, 0.25, 0.15]),
    "transport_mode": rng.choice(modes, N, p=[0.55, 0.2, 0.15, 0.10]),
    "priority": rng.choice(priorities, N, p=[0.6, 0.3, 0.1]),
    "weather": rng.choice(weather, N, p=[0.6, 0.2, 0.12, 0.08]),
})

# Distance depends on mode
dist_base = {"Road": 450, "Rail": 900, "Air": 1800, "Sea": 2500}
df["distance_km"] = [max(30, rng.normal(dist_base[m], dist_base[m] * 0.35)) for m in df.transport_mode]
df["weight_kg"] = np.round(rng.lognormal(mean=5.5, sigma=0.8, size=N), 1)
df["shipment_volume_m3"] = np.round(df.weight_kg / rng.uniform(80, 250, N), 2)
df["warehouse_hours"] = np.round(rng.gamma(2.5, 3.0, N) + (df.weight_kg > 600) * 3, 1)

# Speed (km/day) by mode -> planned days
speed = {"Road": 500, "Rail": 600, "Air": 1500, "Sea": 700}
df["planned_days"] = np.ceil(df.distance_km / df.transport_mode.map(speed) + df.warehouse_hours / 24 + 0.5).astype(int)

# Delay drivers: weather, carrier, month (peak season Oct-Dec), warehouse hours
weather_delay = df.weather.map({"Clear": 0, "Rain": 0.4, "Fog": 0.7, "Storm": 1.6})
carrier_delay = df.carrier.map({"SwiftMove": 0.0, "RoadKing": 0.3, "CargoPro": 0.1, "BlueLine": 0.8})
peak = df.order_date.dt.month.isin([10, 11, 12]).astype(int) * 0.6
prio_delay = df.priority.map({"Standard": 0.0, "Express": -0.2, "Critical": -0.4})
delay = rng.gamma(1.5, 0.3, N) + weather_delay + carrier_delay + peak + prio_delay + 0.03 * df.warehouse_hours - 1.15
df["delay_days"] = np.round(np.clip(delay, -1, None), 1)
df["delivery_days"] = np.round(df.planned_days + df.delay_days, 1)
df["on_time"] = (df.delay_days <= 0.5).astype(int)

# Costs
mode_rate = {"Road": 0.9, "Rail": 0.55, "Air": 4.2, "Sea": 0.35}   # per tonne-km factor
prio_mult = df.priority.map({"Standard": 1.0, "Express": 1.35, "Critical": 1.8})
base = df.distance_km * (df.weight_kg / 1000 + 0.3) * df.transport_mode.map(mode_rate)
df["fuel_cost"] = np.round(base * rng.uniform(0.28, 0.38, N), 2)
df["transport_cost"] = np.round((base * prio_mult + 40 + rng.normal(0, 25, N) + df.delay_days * 18).clip(lower=25), 2)
df["cost_per_km"] = np.round(df.transport_cost / df.distance_km, 3)

df["customer_rating"] = np.clip(np.round(4.7 - 0.55 * df.delay_days + rng.normal(0, 0.5, N)), 1, 5).astype(int)

# Inject a few missing values / outliers to make the EDA realistic
df.loc[rng.choice(N, 25, replace=False), "warehouse_hours"] = np.nan
df.loc[rng.choice(N, 8, replace=False), "transport_cost"] *= 3

df["distance_km"] = df.distance_km.round(1)
df.to_csv("logistics_data.csv", index=False)
print(df.shape)
print(df.head())
