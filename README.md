# Advanced Data Analysis and Visualization in Logistics

Week 3 internship project: simulate a logistics dataset, run a full Exploratory Data Analysis (EDA) in Python, build ten visualizations, and derive operational insights and recommendations.

## Project structure
| File | Purpose |
|---|---|
| `01_generate_data.py` | Simulates 2,000 shipments (2025) with realistic relationships and saves `logistics_data.csv` |
| `02_eda_and_visualizations.py` | Data cleaning, descriptive statistics, correlations and all 10 charts |
| `logistics_data.csv` | Generated dataset |
| `charts/` | 10 exported PNG visualizations |
| `eda_output.txt` | Console output of the EDA script |
| `Week3_Logistics_Analysis_Report.docx` | Full written report |

## How to run
```bash
pip install -r requirements.txt
python 01_generate_data.py
python 02_eda_and_visualizations.py
```
The random seed is fixed (42), so all results are reproducible.

## Dataset variables
shipment_id, order_date, region, carrier, transport_mode, priority, weather, distance_km, weight_kg, shipment_volume_m3, warehouse_hours, planned_days, delay_days, delivery_days, on_time, transport_cost, fuel_cost, cost_per_km, customer_rating

## Visualizations
1. Delivery time distribution (histogram + KDE)
2. Cost per km by transport mode (box plot)
3. Monthly shipments vs on-time rate (bar + line)
4. Correlation heatmap
5. Transport cost vs distance by mode (scatter)
6. On-time rate by carrier (bar)
7. Average delay by region and weather (heatmap)
8. Correlation of variables with transport cost (horizontal bar)
9. Delay by priority (violin)
10. Customer rating vs delay (box plot)

## Key findings (simulated data)
- Overall on-time rate is 74.7%; median delivery time is 2.6 days.
- Cost is driven by transport mode, distance and fuel (fuel-cost correlation with total cost: 0.95).
- Air freight costs about 4x Road per km.
- One carrier (BlueLine) delivers only 46% on time versus 85% for the best carrier.
- Storms cut on-time delivery to 4%; Q4 on-time rate drops from about 80% to 52% by December.
- Delay is strongly linked to lower customer ratings (correlation -0.52).

Note: the data is simulated, so findings reflect the assumptions built into the simulation; the workflow can be reused on real shipment data.
