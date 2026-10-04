import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

"""Отвечает за загрузку и обработку всех статических характеристик матчей и зон."""

class FeatureEncoder:
    def __init__(self, raw_dir):
        self.raw_dir = raw_dir
        self.scaler = MinMaxScaler()
        self.encoded_df = None
        self.feature_cols = []

    def _load_and_merge(self):
        """Загружает таблицы matches.csv и zones.csv, создает day_of_week и считает лимит мест."""
        matches = pd.read_csv(os.path.join(self.raw_dir, 'matches.csv'))
        zones = pd.read_csv(os.path.join(self.raw_dir, 'zones.csv'))

        matches['date_dt'] = pd.to_datetime(matches['date'])
        matches['day_of_week'] = matches['date_dt'].dt.day_name()

        zones['zone_id_orig'] = zones['zone']

        df = pd.merge(zones, matches, on='match_id', how='inner')
        df['max_capacity'] = df['seats'] - df['season_tickets']

        return df

    def fit_transform(self):
        """Выполняет One-Hot Encoding и нормализацию числовых параметров."""
        df = self._load_and_merge()

        num_cols = ['seats', 'season_tickets', 'max_capacity']
        df[num_cols] = self.scaler.fit_transform(df[num_cols])

        cat_cols = ['day_of_week', 'opponent', 'zone']
        actual_cat_cols = [c for c in cat_cols if c in df.columns]
        df_encoded = pd.get_dummies(df, columns=actual_cat_cols, dtype=float)

        exclude = [
            'match_id', 'zone_id_orig', 'date', 'date_dt', 'time', 
            'season', 'train_or_test', 'data_type', 'tickets_total', 'stage'
        ]

        numeric_df = df_encoded.select_dtypes(include=['number', 'bool'])
        self.feature_cols = [c for c in numeric_df.columns if c not in exclude]
        
        self.encoded_df = df_encoded
        return self.encoded_df

    def get_static_features(self, match_id, zone):
        """Безопасно возвращает статический вектор (фичи) для конкретного матча и зоны."""
        mask = (self.encoded_df['match_id'] == match_id) & (self.encoded_df['zone_id_orig'].astype(str) == str(zone))
        row = self.encoded_df[mask]

        if row.empty:
            row = self.encoded_df[self.encoded_df['match_id'] == match_id]
            if row.empty:
                raise ValueError(f"Матч {match_id} не найден в данных.")

        return row[self.feature_cols].values[0].tolist()
