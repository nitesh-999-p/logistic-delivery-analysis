"""
Week 1 - Logistics Data Analysis (Last-Mile Delivery Delays)

Dataset: Olist Brazilian E-Commerce (Kaggle)
Put these CSV files inside a folder called "data":
    olist_orders_dataset.csv
    olist_order_items_dataset.csv
    olist_customers_dataset.csv
    olist_products_dataset.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, r2_score, classification_report
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# ---------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------
orders = pd.read_csv("data/olist_orders_dataset.csv")
items = pd.read_csv("data/olist_order_items_dataset.csv")
customers = pd.read_csv("data/olist_customers_dataset.csv")
products = pd.read_csv("data/olist_products_dataset.csv")

# ---------------------------------------------------------------
# 2. Clean data
# ---------------------------------------------------------------
orders = orders.drop_duplicates()
orders = orders[orders["order_status"] == "delivered"]

date_cols = ["order_purchase_timestamp",
             "order_delivered_customer_date",
             "order_estimated_delivery_date"]
for col in date_cols:
    orders[col] = pd.to_datetime(orders[col])

orders = orders.dropna(subset=["order_delivered_customer_date"])

# Delivery time in days, promised time in days, and late flag
orders["delivery_days"] = (orders["order_delivered_customer_date"]
                           - orders["order_purchase_timestamp"]).dt.days
orders["promised_days"] = (orders["order_estimated_delivery_date"]
                           - orders["order_purchase_timestamp"]).dt.days
orders["is_late"] = (orders["delivery_days"] > orders["promised_days"]).astype(int)

# One row per order: total freight cost and total weight
items = items.merge(products[["product_id", "product_weight_g"]],
                    on="product_id", how="left")
order_info = items.groupby("order_id").agg(
    freight_value=("freight_value", "sum"),
    weight_g=("product_weight_g", "sum"),
    num_items=("order_item_id", "count"),
).reset_index()

df = (orders.merge(order_info, on="order_id")
            .merge(customers[["customer_id", "customer_state"]], on="customer_id"))
df["weight_g"] = df["weight_g"].fillna(df["weight_g"].median())

print("Orders after cleaning:", len(df))

# ---------------------------------------------------------------
# 3. KPIs
# ---------------------------------------------------------------
on_time_rate = (1 - df["is_late"].mean()) * 100
avg_delivery_days = df["delivery_days"].mean()
cost_per_delivery = df["freight_value"].mean()

print("\n--- KPIs ---")
print(f"On-time delivery rate : {on_time_rate:.1f} %")
print(f"Average delivery time : {avg_delivery_days:.1f} days")
print(f"Cost per delivery     : {cost_per_delivery:.2f}")

# ---------------------------------------------------------------
# 4. Exploratory analysis - delays by state
# ---------------------------------------------------------------
state_stats = df.groupby("customer_state").agg(
    orders=("order_id", "count"),
    avg_days=("delivery_days", "mean"),
    late_rate=("is_late", "mean"),
    avg_freight=("freight_value", "mean"),
)
state_stats = state_stats[state_stats["orders"] >= 100]  # ignore tiny states

top = state_stats.sort_values("late_rate", ascending=False).head(10)
top["late_rate"].mul(100).plot(kind="bar", color="steelblue")
plt.title("Late delivery rate by state (top 10)")
plt.ylabel("Late orders (%)")
plt.xlabel("State")
plt.tight_layout()
plt.savefig("late_rate_by_state.png")
plt.close()

# ---------------------------------------------------------------
# 5. Regression - predict delivery time
# ---------------------------------------------------------------
X = pd.get_dummies(df[["promised_days", "freight_value", "weight_g",
                       "num_items", "customer_state"]], drop_first=True)
y = df["delivery_days"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

reg = LinearRegression().fit(X_train, y_train)
pred = reg.predict(X_test)
print("\n--- Regression (delivery days) ---")
print("MAE :", round(mean_absolute_error(y_test, pred), 2), "days")
print("R2  :", round(r2_score(y_test, pred), 3))

# ---------------------------------------------------------------
# 6. Classification - predict late orders
# ---------------------------------------------------------------
y_late = df["is_late"]
Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X, y_late, test_size=0.2, random_state=42, stratify=y_late)

clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
clf.fit(Xc_train, yc_train)
print("\n--- Classification (is the order late?) ---")
print(classification_report(yc_test, clf.predict(Xc_test)))

# ---------------------------------------------------------------
# 7. Clustering - group states by delay behaviour
# ---------------------------------------------------------------
scaled = StandardScaler().fit_transform(state_stats[["avg_days", "late_rate", "avg_freight"]])
state_stats["cluster"] = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(scaled)
print("\n--- State clusters ---")
print(state_stats.sort_values("cluster")[["avg_days", "late_rate", "avg_freight", "cluster"]].round(2))

state_stats.to_csv("state_summary.csv")
print("\nSaved: late_rate_by_state.png, state_summary.csv")
