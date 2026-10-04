import os  # Модуль для работы с файловой системой и путями к файлам
import pandas as pd  # Библиотека Pandas для работы с таблицами 
from sklearn.preprocessing import MinMaxScaler  # Инструмент для приведения чисел к диапазону от 0 до 1

"""Отвечает за загрузку и обработку всех статических характеристик матчей и зон.
Класс объединяет таблицы матчей и зон, рассчитывает реальный лимит мест (max_capacity), кодирует категориальные признаки с помощью One-Hot Encoding и нормализует числовые данные с помощью MinMaxScaler."""
class FeatureEncoder:
    def __init__(self, raw_dir):
        self.raw_dir = raw_dir  # Сохраняем путь к папке с исходными CSV - файлами
        self.scaler = MinMaxScaler()  # Инициализируем масштабировщик чисел (MinMaxScaler)
        self.encoded_df = None  # Переменная для хранения итоговой обработанной таблицы
        self.feature_cols = []  # Список имен столбцов, которые пойдут напрямую в нейросеть

    def _load_and_merge(self):
        """Загружает таблицы matches.csv и zones.csv, создает day_of_week и считает лимит мест."""
        # Читаем файл с информацией о матчах
        matches = pd.read_csv(os.path.join(self.raw_dir, 'matches.csv'))
        # Читаем файл с информацией о зонах стадиона
        zones = pd.read_csv(os.path.join(self.raw_dir, 'zones.csv'))
        
        # Преобразуем текстовую дату матча в формат datetime библиотеки pandas
        matches['date_dt'] = pd.to_datetime(matches['date'])
        # Извлекаем текстовое название дня недели (Monday, Tuesday и т.д.)
        matches['day_of_week'] = matches['date_dt'].dt.day_name()
        
        # Сохраняем исходное название/ID зоны до того, как One-Hot Encoding разбьет его на новые столбцы
        zones['zone_id_orig'] = zones['zone']
        
        # Объединяем зоны и матчи в одну общую таблицу по ключу match_id
        df = pd.merge(zones, matches, on='match_id', how='inner')
        
        # Считаем физический лимит доступных мест: общие места минус выкупленные абонементы
        df['max_capacity'] = df['seats'] - df['season_tickets']
        # Возвращаем объединенную таблицу
        return df

    def fit_transform(self):
        """Выполняет One-Hot Encoding и нормализацию числовых параметров."""
        # Загружаем и объединяем данные с помощью предыдущего метода
        df = self._load_and_merge()
        
        # Определяем список числовых столбцов для нормализации
        num_cols = ['seats', 'season_tickets', 'max_capacity']
        # Масштабируем числовые характеристики к диапазону [0, 1] с помощью MinMaxScaler
        df[num_cols] = self.scaler.fit_transform(df[num_cols])
        
        # Определяем список категориальных столбцов для One-Hot Encoding
        cat_cols = ['day_of_week', 'opponent', 'zone']
        # Кодируем категории в бинарные столбцы (0.0 и 1.0)
        df_encoded = pd.get_dummies(df, columns=cat_cols, dtype=float)
        
        # Список служебных и текстовых столбцов, которые НЕ ДОЛЖНЫ попадать в нейросеть
        exclude = [
            'match_id', 'zone_id_orig', 'date', 'date_dt', 'time', 
            'season', 'train_or_test', 'data_type', 'tickets_total', 'stage'
        ]
        # Фильтруем только те столбцы, которые пойдут в нейросеть как фичи
        self.feature_cols = [c for c in df_encoded.columns if c not in exclude]
        
        # Сохраняем закодированную таблицу в атрибут класса
        self.encoded_df = df_encoded
        # Возвращаем готовую таблицу
        return self.encoded_df

    def get_static_features(self, match_id, zone):
        """Безопасно возвращает статический вектор (фичи) для конкретного матча и зоны."""
        # Создаем маску для поиска нужной строки по ID матча и исходному ID зоны
        mask = (self.encoded_df['match_id'] == match_id) & (self.encoded_df['zone_id_orig'].astype(str) == str(zone))
        # Применяем фильтр к таблице
        row = self.encoded_df[mask]
        
        # Если совпадений не найдено, делаем фоллбэк на первый попавшийся сегмент этого матча
        if row.empty:
            row = self.encoded_df[self.encoded_df['match_id'] == match_id]
            # Если матч вовсе отсутствует в базе, выбрасываем ошибку
            if row.empty:
                raise ValueError(f"Матч {match_id} не найден в данных.")
            
        # Вырезаем только числовые признаковые столбцы и возвращаем их в виде обычного списка float
        return row[self.feature_cols].values[0].tolist()
