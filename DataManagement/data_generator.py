import os  # Модуль для работы с файлами
import pandas as pd  # Библиотека для обработки таблиц
from feature_encoder import FeatureEncoder  # Импортируем наш кодировщик статических фичей
from sales_preprocessor import SalesPreprocessor  # Импортируем обработчик динамики продаж

"""Главный класс-фасад 
(высокоуровневый интерфейс) для работы с данными.
Объединяет работу FeatureEncoder и SalesPreprocessor. Предоставляет нейросети готовый интерфейс для получения вектора признаков под одиночный 
прогноз (get_match_zone_features) или полного датасета под обучение (get_train_data / get_val_data)."""


class HockeyDataGenerator:
    def __init__(self, raw_dir, val_ratio=0.2):
        self.raw_dir = raw_dir  # Сохраняем путь к папке с сырыми CSV
        self.encoder = FeatureEncoder(raw_dir)  # Инициализируем объект FeatureEncoder
        self.sales = SalesPreprocessor(raw_dir)  # Инициализируем объект SalesPreprocessor

        self.val_ratio = val_ratio

        self.prepare()

    def prepare(self):
        """Загружает и подготавливает все статические и динамические данные."""
        # Запускаем полный цикл обработки статики (OHE + MinMaxScaler)
        self.encoder.fit_transform()
        # Загружаем и индексируем файл продаж sales_daily.csv
        self.sales.load_sales()

    def get_match_zone_features(self, match_id, zone):
        """Возвращает итоговый вектор для нейросети по ID матча и зоне."""
        # Извлекаем статический вектор фичей (44 элемента) для данного матча и зоны
        static_vec = self.encoder.get_static_features(match_id, zone)
        
        # Находим строку матча для получения параметра max_capacity
        mask = (self.encoder.encoded_df['match_id'] == match_id)
        max_cap = self.encoder.encoded_df[mask]['max_capacity'].values[0]
        
        # Получаем актуальную динамику продаж (days_since_start, sold_ratio)
        days, sold_ratio = self.sales.get_latest_sales_state(match_id, zone, max_cap)
        # Объединяем 2 динамических и 44 статических элемента в 1 вектор из 46 чисел
        return [days, sold_ratio] + static_vec

    def get_train_data(self):
        """Собирает обучающую выборку (X, Y) по всем дням продаж прошлых матчей."""
        X, Y = [], []  # Создаем пустые списки под признаки X и ответы Y
        
        # Берем обработанную таблицу со статическими данными
        df = self.encoder.encoded_df
        type_col = None
        # Автоматически определяем столбцы разделения на train/test
        for col in ['train_or_test', 'data_type', 'stage']:
            if col in df.columns:
                type_col = col
                break
        
        # Фильтруем только строки обучающей выборки (train)
        train_df = df[df[type_col] == 'train'] if type_col else df
        
        # Проходимся по каждой строке прошлых матчей и зон
        for _, row in train_df.iterrows():
            m_id, z_name = row['match_id'], row['zone_id_orig']
            max_cap = row['max_capacity']
            # Получаем статические признаки этой зоны
            static_vec = self.encoder.get_static_features(m_id, z_name)
            
            # Получаем ежедневные снимки динамики продаж по дням
            snapshots = self.sales.generate_daily_snapshots(m_id, z_name, max_cap)
            # Извлекаем итоговую целевую переменную (сколько всего билетов было продано)
            total_target = row.get('tickets_total', 0)
            
            # Разворачиваем историю по дням: для каждого дня создаем обучающий пример
            for days, ratio, _ in snapshots:
                # Добавляем вектор из 46 элементов в матрицу X
                X.append([days, ratio] + static_vec)
                # Добавляем целевой ответ в список Y
                Y.append(total_target)

        split_idx = int(len(X) * (1 - self.val_ratio))

        return X[:split_idx], Y[:split_idx]

    def get_val_data(self):
        """Возвращает валидационную выборку (последние N% матчей)."""
        # Сначала собираем все данные обучающей выборки
        X_all, Y_all = self.get_train_data()
        # Вычисляем индекс отсечения для валидационной части (например, 80% train / 20% val)
        split_idx = int(len(X_all) * (1 - self.val_ratio))
        # Возвращаем последние 20% записей для проверки качества модели
        return X_all[split_idx:], Y_all[split_idx:]
