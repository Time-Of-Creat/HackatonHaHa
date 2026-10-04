import os  # Модуль для работы с путями к файлам
import pandas as pd  # Библиотека Pandas для обработки табличных данных

"""Отвечает за загрузку и обработку временных рядов динамики продаж билетов.
Класс автоматически находит данные о ежедневных продажах из sales_daily.csv,рассчитывает количество дней со старта продаж (days_since_start) и текущую 
долю выкупленных мест (sold_ratio) как для актуального состояния, так и в виде ежедневных снимков (snapshots) для истории."""
class SalesPreprocessor:
    def __init__(self, raw_dir):
        self.raw_dir = raw_dir  # Путь к папке с CSV - файлами
        self.sales_df = None  # Переменная под DataFrame ежедневных продаж
        self.tickets_col = None  # Переменная для хранения распознанного названия столбца с билетами

    def load_sales(self):
        """Загружает файл с историей ежедневных продаж и находит колонку проданных билетов."""
        # Формируем полный путь к файлу sales_daily.csv
        path = os.path.join(self.raw_dir, 'sales_daily.csv')
        # Проверяем, существует ли файл по указанному пути
        if os.path.exists(path):
            # Загружаем файл продаж в DataFrame
            self.sales_df = pd.read_csv(path)
            # Переводим колонку с датами в формат datetime
            self.sales_df['date'] = pd.to_datetime(self.sales_df['date'])
            
            # Список возможных вариантов названия колонки с количеством проданных билетов
            possible_cols = ['tickets_sold', 'tickets', 'sold', 'tickets_count', 'count']
            # Ищем, какое из этих названий присутствует в загруженном файле
            for col in possible_cols:
                if col in self.sales_df.columns:
                    self.tickets_col = col  # Сохраняем найденное имя
                    break  # Прерываем поиск при первом совпадении
            
            # Если совпадений по имени нет, берем первую подходящую числовую колонку
            if not self.tickets_col:
                numeric_cols = self.sales_df.select_dtypes(include=['number']).columns
                exclude = ['match_id', 'zone', 'zone_id']
                candidates = [c for c in numeric_cols if c not in exclude]
                if candidates:
                    self.tickets_col = candidates[0]
        else:
            # Если файла нет, создаем пустой DataFrame
            self.sales_df = pd.DataFrame()

    def get_latest_sales_state(self, match_id, zone, max_capacity):
        """Возвращает актуальное состояние продаж: (days_since_start, sold_ratio)."""
        # Если таблицы нет или не найден столбец билетов, возвращаем дефолтные нули
        if self.sales_df.empty or not self.tickets_col:
            return 0.0, 0.0

        # Фильтруем продажи по конкретному матчу и зоне
        mask = (self.sales_df['match_id'] == match_id) & (self.sales_df['zone'].astype(str) == str(zone))
        # Применяем фильтр и сортируем записи по хронологии дат
        match_sales = self.sales_df[mask].sort_values('date')

        # Если истории продаж для данной зоны нет, возвращаем 0.0, 0.0
        if match_sales.empty:
            return 0.0, 0.0

        # Находим дату начала продаж
        start_date = match_sales['date'].min()
        # Находим последнюю зафиксированную дату продаж
        latest_date = match_sales['date'].max()
        
        # Вычисляем разницу в днях между последней датой и стартом продаж
        days_since_start = (latest_date - start_date).days
        # Считаем суммарно проданные билеты к последней дате
        total_sold = match_sales[self.tickets_col].sum()
        # Вычисляем долю проданных билетов от лимита max_capacity
        sold_ratio = total_sold / max_capacity if max_capacity > 0 else 0.0

        # Возвращаем пару значений: количество дней и ограниченную долю [0.0, 1.0]
        return float(days_since_start), float(min(sold_ratio, 1.0))

    def generate_daily_snapshots(self, match_id, zone, max_capacity):
        """Генерирует список всех дневных точек (days, ratio, cumulative_sold) для обучения."""
        # Если таблицы нет, отдаем базовую стартовую точку
        if self.sales_df.empty or not self.tickets_col:
            return [(0.0, 0.0, 0)]

        # Фильтруем записи по матчу и зоне
        mask = (self.sales_df['match_id'] == match_id) & (self.sales_df['zone'].astype(str) == str(zone))
        # Сортируем записи строго по порядку дат
        match_sales = self.sales_df[mask].sort_values('date')

        # Если данных нет, отдаем начальную точку
        if match_sales.empty:
            return [(0.0, 0.0, 0)]

        # Фиксируем дату первого дня продаж
        start_date = match_sales['date'].min()
        snapshots = []  # Список для накопительных ежедневных снимков
        cumulative_sold = 0  # Накопительный счетчик проданных билетов

        # Добавляем начальное состояние на момент старта продаж (день 0, продано 0%)
        snapshots.append((0.0, 0.0, 0))

        # Итерируемся по каждому дню продаж из таблицы
        for _, row in match_sales.iterrows():
            # Вычисляем, сколько дней прошло с первого дня продаж
            days = (row['date'] - start_date).days
            # Прибавляем продажи за текущий день к общей сумме
            cumulative_sold += row[self.tickets_col]
            # Вычисляем накопительную долю проданных билетов
            ratio = cumulative_sold / max_capacity if max_capacity > 0 else 0.0
            # Сохраняем кортеж с параметрами дня в итоговый список
            snapshots.append((float(days), float(min(ratio, 1.0)), cumulative_sold))

        # Возвращаем историю продаж по дням
        return snapshots
