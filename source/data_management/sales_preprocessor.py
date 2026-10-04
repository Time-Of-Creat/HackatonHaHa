import os
import pandas as pd

class SalesPreprocessor:
    def __init__(self, raw_dir='data/raw'):
        self.raw_dir = raw_dir
        self.sales_df = None

    def load_sales(self):
        sales_path = os.path.join(self.raw_dir, 'sales_daily.csv')
        if os.path.exists(sales_path):
            self.sales_df = pd.read_csv(sales_path)
            self.sales_df['match_id_str'] = self.sales_df['match_id'].astype(str).str.strip()
            self.sales_df['zone_str'] = self.sales_df['zone'].astype(str).str.strip()
            self.sales_df['date'] = pd.to_datetime(self.sales_df['date'])
            self.sales_df = self.sales_df.sort_values('date')

    def generate_daily_snapshots(self, match_id, zone, max_capacity):
        if self.sales_df is None:
            return [(0.0, 0.0, 0)]

        mask = (self.sales_df['match_id_str'] == str(match_id).strip()) & \
               (self.sales_df['zone_str'] == str(zone).strip())
        sub = self.sales_df[mask]

        if sub.empty:
            return [(0.0, 0.0, 0)]

        start_date = sub['date'].min()
        cum_tickets = 0
        snapshots = []

        for _, row in sub.iterrows():
            days_since_start = float((row['date'] - start_date).days)
            cum_tickets += row['tickets']
            sold_ratio = float(cum_tickets / max_capacity) if max_capacity > 0 else 0.0
            snapshots.append((days_since_start, sold_ratio, cum_tickets))

        return snapshots

    def get_latest_sales_state(self, match_id, zone, max_capacity):
        snapshots = self.generate_daily_snapshots(match_id, zone, max_capacity)
        days, ratio, _ = snapshots[-1]
        return days, ratio
