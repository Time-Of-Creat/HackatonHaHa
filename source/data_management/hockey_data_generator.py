import os
import pandas as pd
from source.data_management.feature_encoder import FeatureEncoder
from source.data_management.sales_preprocessor import SalesPreprocessor
from source.file_manager.folders_data import folders_data


class HockeyDataGenerator:
    # В тестовых матчах продажи обрезаны за 10 дней до игры. Если оставить в
    # обучении более поздние снапшоты, sold_ratio на последнем из них совпадает
    # с таргетом, и модель учится копировать признак вместо прогнозирования.
    MIN_DAYS_BEFORE = 10

    def __init__(self, val_ratio=0.2):
        self.raw_dir = str(folders_data.DATA_DIR)
        self.encoder = FeatureEncoder(self.raw_dir)
        self.sales = SalesPreprocessor(self.raw_dir, min_days_before=self.MIN_DAYS_BEFORE)
        self.val_ratio = val_ratio

        self.targets_cache = {}
        self.train_data = ([], [])
        self.val_data = ([], [])
        self.prepare()

    def prepare(self):
        self.encoder.fit_transform()
        self.sales.load_sales(train_match_ids=self.encoder.get_train_match_ids())
        self._build_targets_cache()
        self._build_datasets()

    def _build_targets_cache(self):
        sub_path = os.path.join(self.raw_dir, 'submission_template.csv')
        if os.path.exists(sub_path):
            sub_df = pd.read_csv(sub_path)
            if 'tickets_total' in sub_df.columns:
                for _, row in sub_df.iterrows():
                    if pd.isna(row['tickets_total']):
                        continue
                    key = (str(row['match_id']).strip(), str(row['zone']).strip())
                    self.targets_cache[key] = float(row['tickets_total'])

        if self.sales.sales_df is not None:
            grouped = self.sales.sales_df.groupby(['match_id_str', 'zone_str'])['tickets'].sum().reset_index()
            for _, row in grouped.iterrows():
                key = (row['match_id_str'], row['zone_str'])
                if key not in self.targets_cache or self.targets_cache[key] == 0:
                    self.targets_cache[key] = float(row['tickets'])

    def get_match_zone_features(self, match_id, zone):
        static_vec = self.encoder.get_static_features(match_id, zone)
        max_cap = self.encoder.get_capacity(match_id, zone)
        days, sold_ratio = self.sales.get_latest_sales_state(match_id, zone, max_cap)
        return [days, sold_ratio] + static_vec

    def _split_match_ids(self, train_df):
        first_dates = pd.to_datetime(train_df.groupby('match_id')['date'].first())
        ordered = first_dates.sort_values().index.astype(str).str.strip().tolist()
        val_count = int(round(len(ordered) * self.val_ratio))
        split = len(ordered) - val_count
        return set(ordered[:split]), set(ordered[split:])

    def _build_datasets(self):
        train_df = self.encoder.encoded_df[self.encoder.encoded_df['part'] == 'train']
        train_ids, val_ids = self._split_match_ids(train_df)

        train_rows, val_rows = [], []
        for _, row in train_df.iterrows():
            m_id = str(row['match_id']).strip()
            z_name = str(row['zone_id_orig']).strip()
            max_cap = row['max_capacity']

            snapshots = self.sales.generate_daily_snapshots(row['match_id'], row['zone_id_orig'], max_cap)
            if not snapshots:
                continue

            total_target = self.targets_cache.get((m_id, z_name), 0.0)
            target_ratio = float(total_target / max_cap) if max_cap > 0 else 0.0
            static_vec = self.encoder.get_static_features(row['match_id'], row['zone_id_orig'])

            bucket = train_rows if m_id in train_ids else val_rows
            for days, ratio, _ in snapshots:
                bucket.append(([days, ratio] + static_vec, target_ratio))

        self.train_data = ([r[0] for r in train_rows], [r[1] for r in train_rows])
        self.val_data = ([r[0] for r in val_rows], [r[1] for r in val_rows])

    def get_train_data(self):
        return self.train_data

    def get_val_data(self):
        return self.val_data
