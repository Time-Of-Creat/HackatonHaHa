import os
import pandas as pd
from feature_encoder import FeatureEncoder
from sales_preprocessor import SalesPreprocessor

class HockeyDataGenerator:
    def __init__(self, raw_dir='data/raw', val_ratio=0.2):
        self.raw_dir = raw_dir
        self.encoder = FeatureEncoder(raw_dir)
        self.sales = SalesPreprocessor(raw_dir)
        self.val_ratio = val_ratio
        
        self.targets_cache = {}
        self.prepare()

    def prepare(self):
        self.encoder.fit_transform()
        self.sales.load_sales()
        self._build_targets_cache()

    def _build_targets_cache(self):
        sub_path = os.path.join(self.raw_dir, 'submission_template.csv')
        if os.path.exists(sub_path):
            sub_df = pd.read_csv(sub_path)
            if 'tickets_total' in sub_df.columns:
                for _, row in sub_df.iterrows():
                    key = (str(row['match_id']).strip(), str(row['zone']).strip())
                    self.targets_cache[key] = float(row['tickets_total'])

        sales_path = os.path.join(self.raw_dir, 'sales_daily.csv')
        if os.path.exists(sales_path):
            sales_df = pd.read_csv(sales_path)
            sales_df['match_id_str'] = sales_df['match_id'].astype(str).str.strip()
            sales_df['zone_str'] = sales_df['zone'].astype(str).str.strip()
            
            grouped = sales_df.groupby(['match_id_str', 'zone_str'])['tickets'].sum().reset_index()
            for _, row in grouped.iterrows():
                key = (row['match_id_str'], row['zone_str'])
                if key not in self.targets_cache or self.targets_cache[key] == 0:
                    self.targets_cache[key] = float(row['tickets'])

    def get_match_zone_features(self, match_id, zone):
        static_vec = self.encoder.get_static_features(match_id, zone)
        
        mask = (self.encoder.encoded_df['match_id'] == match_id)
        max_cap = self.encoder.encoded_df[mask]['max_capacity'].values[0]
        
        days, sold_ratio = self.sales.get_latest_sales_state(match_id, zone, max_cap)
        return [days, sold_ratio] + static_vec

    def get_train_data(self):
        X, Y = [], []
        
        df = self.encoder.encoded_df
        type_col = next((c for c in ['train_or_test', 'data_type', 'stage'] if c in df.columns), None)
        train_df = df[df[type_col] == 'train'] if type_col else df
        
        zone_col = 'zone_id_orig' if 'zone_id_orig' in train_df.columns else 'zone'
        
        for _, row in train_df.iterrows():
            m_id = str(row['match_id']).strip()
            z_name = str(row[zone_col]).strip()
            max_cap = row['max_capacity']
            
            static_vec = self.encoder.get_static_features(row['match_id'], row[zone_col])
            
            if hasattr(self.sales, 'generate_daily_snapshots'):
                snapshots = self.sales.generate_daily_snapshots(row['match_id'], row[zone_col], max_cap)
            else:
                snapshots = self.sales.generate_snapshots(row['match_id'], row[zone_col], max_cap)
            
            total_target = self.targets_cache.get((m_id, z_name), 0.0)
            
            for days, ratio, _ in snapshots:
                X.append([days, ratio] + static_vec)
                Y.append(total_target)

        split_idx = int(len(X) * (1 - self.val_ratio))
        return X[:split_idx], Y[:split_idx]

    def get_val_data(self):
        X_all, Y_all = self.get_train_data()
        split_idx = int(len(X_all) * (1 - self.val_ratio))
        return X_all[split_idx:], Y_all[split_idx:]
