import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class FeatureEncoder:
    def __init__(self, raw_dir='data/raw'):
        self.raw_dir = raw_dir
        self.scaler = MinMaxScaler()
        self.encoded_df = None
        self.feature_cols = []

    def fit_transform(self):
        matches_path = os.path.join(self.raw_dir, 'matches.csv')
        zones_path = os.path.join(self.raw_dir, 'zones.csv')
        
        matches_df = pd.read_csv(matches_path)
        zones_df = pd.read_csv(zones_path)
        
        df = pd.merge(zones_df, matches_df, on='match_id', how='left')
        
        df['zone_id_orig'] = df['zone'].astype(str)
        df['max_capacity'] = df['seats']
        
        if 'weekday' in df.columns:
            day_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
            df['day_of_week'] = df['weekday'].map(day_names).fillna('Unknown')
            
        df = pd.get_dummies(df, columns=['day_of_week', 'opponent', 'zone'], prefix=['day_of_week', 'opponent', 'zone'], dtype=float)
        
        exclude_cols = ['match_id', 'zone_id_orig', 'part', 'tickets_total', 'season', 'date', 'time', 'sales_open', 'opponent_khl']
        self.feature_cols = [c for c in df.columns if c not in exclude_cols]
        
        df[self.feature_cols] = self.scaler.fit_transform(df[self.feature_cols])
        
        self.encoded_df = df
        return df

    def get_static_features(self, match_id, zone):
        mask = (self.encoded_df['match_id'] == match_id) & (self.encoded_df['zone_id_orig'].astype(str) == str(zone))
        sub = self.encoded_df[mask]
        if sub.empty:
            return [0.0] * len(self.feature_cols)
        return sub[self.feature_cols].iloc[0].tolist()
