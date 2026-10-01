# Logistics Delivery Delay Analysis

Week 1 project for the **Logistics Data Analyst Internship**.

## Problem
An e-commerce company has late deliveries and high delivery costs. This project uses
Python to find where and why delays happen, and to predict them.

## KPIs
| KPI | Formula |
|---|---|
| On-time delivery rate | On-time orders / Total orders x 100 |
| Average delivery time | Sum of delivery days / Number of orders |
| Cost per delivery | Total freight cost / Number of orders |

## Dataset
[Olist Brazilian E-Commerce Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (Kaggle).
Download these files and place them in a `data/` folder:
- `olist_orders_dataset.csv`
- `olist_order_items_dataset.csv`
- `olist_customers_dataset.csv`
- `olist_products_dataset.csv`

## What the code does (`week1_analysis.py`)
1. **Cleaning:** removes duplicates, keeps delivered orders, fixes date formats, creates `delivery_days` and `is_late`.
2. **KPIs:** calculates the three KPIs above.
3. **EDA:** late-delivery rate by state (saves `late_rate_by_state.png`).
4. **Regression:** predicts delivery time in days (MAE and R2).
5. **Classification:** predicts whether an order will be late (logistic regression).
6. **Clustering:** groups states by delay and cost using K-Means (saves `state_summary.csv`).

## How to run
```
pip install pandas scikit-learn matplotlib
python week1_analysis.py
```

## Tools
Python, pandas, scikit-learn, matplotlib

## Documents
`Week1_Logistics_Strategic_Planning_Report.docx` - the full strategic planning report.
