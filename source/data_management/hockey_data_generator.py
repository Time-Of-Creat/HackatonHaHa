from source.data_management.feature_encoder import FeatureEncoder
from source.data_management.sales_preprocessor import SalesPreprocessor

"""высокоуровневый интерфейс для работы с данными"""


class HockeyDataGenerator:
    def __init__(self, raw_dir, val_ratio=0.2):
        self.raw_dir = raw_dir
        self.encoder = FeatureEncoder(raw_dir)
        self.sales = SalesPreprocessor(raw_dir)

        self.val_ratio = val_ratio

        self.prepare()

    def prepare(self):
        """Загружает и подготавливает все статические и динамические данные."""
        self.encoder.fit_transform()
        self.sales.load_sales()

    def get_match_zone_features(self, match_id, zone):
        """Возвращает итоговый вектор для нейросети по ID матча и зоне."""
        static_vec = self.encoder.get_static_features(match_id, zone)

        mask = (self.encoder.encoded_df['match_id'] == match_id)
        max_cap = self.encoder.encoded_df[mask]['max_capacity'].values[0]

        days, sold_ratio = self.sales.get_latest_sales_state(match_id, zone, max_cap)
        return [days, sold_ratio] + static_vec

    def get_past_data(self):
        """Собирает выборку данных прошлого для обучения и валидации"""
        X, Y = [], []

        df = self.encoder.encoded_df
        type_col = None
        for col in ['train_or_test', 'data_type', 'stage']:
            if col in df.columns:
                type_col = col
                break

        train_df = df[df[type_col] == 'train'] if type_col else df

        for _, row in train_df.iterrows():
            m_id, z_name = row['match_id'], row['zone_id_orig']
            max_cap = row['max_capacity']
            static_vec = self.encoder.get_static_features(m_id, z_name)

            snapshots = self.sales.generate_daily_snapshots(m_id, z_name, max_cap)
            total_target = row.get('tickets_total', 0)

            for days, ratio, _ in snapshots:
                X.append([days, ratio] + static_vec)
                Y.append(total_target)

        return X, Y

    def get_train_data(self):
        """Собирает обучающую выборку (X, Y) по всем дням продаж прошлых матчей."""
        X, Y = self.get_past_data()
        split_idx = int(len(X) * (1 - self.val_ratio))
        return X[:split_idx], Y[:split_idx]

    def get_val_data(self):
        """Возвращает валидационную выборку (последние N% матчей)."""
        X, Y = self.get_past_data()
        split_idx = int(len(X) * (1 - self.val_ratio))
        return X[split_idx:], Y[split_idx:]
