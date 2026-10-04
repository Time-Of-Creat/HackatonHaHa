import os  # Модуль для работы с путями к файлам
import pandas as pd  # Библиотека Pandas для обработки табличных данных

"""Отвечает за загрузку и обработку временных рядов динамики продаж билетов"""
class SalesPreprocessor:
    def __init__(self, raw_dir):
        self.raw_dir = raw_dir
        self.sales_df = None
        self.tickets_col = None

    def load_sales(self):
        """Загружает файл с историей ежедневных продаж и находит колонку проданных билетов."""
        path = os.path.join(self.raw_dir, 'sales_daily.csv')
        if os.path.exists(path):
            self.sales_df = pd.read_csv(path)
            self.sales_df['date'] = pd.to_datetime(self.sales_df['date'])

            possible_cols = ['tickets_sold', 'tickets', 'sold', 'tickets_count', 'count']
            for col in possible_cols:
                if col in self.sales_df.columns:
                    self.tickets_col = col
                    break

            if not self.tickets_col:
                numeric_cols = self.sales_df.select_dtypes(include=['number']).columns
                exclude = ['match_id', 'zone', 'zone_id']
                candidates = [c for c in numeric_cols if c not in exclude]
                if candidates:
                    self.tickets_col = candidates[0]
        else:
            self.sales_df = pd.DataFrame()

    def get_latest_sales_state(self, match_id, zone, max_capacity):
        """Возвращает актуальное состояние продаж: (days_since_start, sold_ratio)."""
        if self.sales_df.empty or not self.tickets_col:
            return 0.0, 0.0

        mask = (self.sales_df['match_id'] == match_id) & (self.sales_df['zone'].astype(str) == str(zone))
        match_sales = self.sales_df[mask].sort_values('date')

        if match_sales.empty:
            return 0.0, 0.0

        start_date = match_sales['date'].min()
        latest_date = match_sales['date'].max()

        days_since_start = (latest_date - start_date).days
        total_sold = match_sales[self.tickets_col].sum()
        sold_ratio = total_sold / max_capacity if max_capacity > 0 else 0.0

        return float(days_since_start), float(min(sold_ratio, 1.0))

    def generate_daily_snapshots(self, match_id, zone, max_capacity):
        """Генерирует список всех дневных точек (days, ratio, cumulative_sold) для обучения."""
        if self.sales_df.empty or not self.tickets_col:
            return [(0.0, 0.0, 0)]

        mask = (self.sales_df['match_id'] == match_id) & (self.sales_df['zone'].astype(str) == str(zone))
        match_sales = self.sales_df[mask].sort_values('date')

        if match_sales.empty:
            return [(0.0, 0.0, 0)]

        start_date = match_sales['date'].min()
        snapshots = []
        cumulative_sold = 0

        snapshots.append((0.0, 0.0, 0))

        for _, row in match_sales.iterrows():
            days = (row['date'] - start_date).days
            cumulative_sold += row[self.tickets_col]
            ratio = cumulative_sold / max_capacity if max_capacity > 0 else 0.0
            snapshots.append((float(days), float(min(ratio, 1.0)), cumulative_sold))

        return snapshots
