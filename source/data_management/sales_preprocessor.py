import os
import pandas as pd


class SalesPreprocessor:
    def __init__(self, raw_dir='data/raw', min_days_before=0):
        self.raw_dir = raw_dir
        self.min_days_before = min_days_before
        self.sales_df = None
        self.max_span_days = 1.0

    def load_sales(self, train_match_ids=None):
        sales_path = os.path.join(self.raw_dir, 'sales_daily.csv')
        if os.path.exists(sales_path):
            self.sales_df = pd.read_csv(sales_path)
            self.sales_df['match_id_str'] = self.sales_df['match_id'].astype(str).str.strip()
            self.sales_df['zone_str'] = self.sales_df['zone'].astype(str).str.strip()
            self.sales_df['date'] = pd.to_datetime(self.sales_df['date'])
            self.sales_df = self.sales_df.sort_values('date')

            spans_df = self.sales_df
            if train_match_ids is not None:
                spans_df = spans_df[spans_df['match_id_str'].isin(train_match_ids)]

            spans = spans_df.groupby(['match_id_str', 'zone_str'])['date'].agg(['min', 'max'])
            self.max_span_days = float(max((spans['max'] - spans['min']).dt.days.max(), 1))

    def generate_daily_snapshots(self, match_id, zone, max_capacity):
        if self.sales_df is None:
            return []

        mask = (self.sales_df['match_id_str'] == str(match_id).strip()) & \
               (self.sales_df['zone_str'] == str(zone).strip())
        sub = self.sales_df[mask]
        sub = sub[sub['days_before'] >= self.min_days_before]

        if sub.empty:
            return []

        start_date = sub['date'].min()
        sum_tickets = 0
        snapshots = []

        for _, row in sub.iterrows():
            days_since_start = float((row['date'] - start_date).days)
            sum_tickets += row['tickets']
            sold_ratio = float(sum_tickets / max_capacity) if max_capacity > 0 else 0.0
            snapshots.append((days_since_start / self.max_span_days, sold_ratio, sum_tickets))

        return snapshots

    def get_latest_sales_state(self, match_id, zone, max_capacity):
        snapshots = self.generate_daily_snapshots(match_id, zone, max_capacity)
        if not snapshots:
            return 0.0, 0.0
        days, ratio, _ = snapshots[-1]
        return days, ratio
