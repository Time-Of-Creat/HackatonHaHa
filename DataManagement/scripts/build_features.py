import os
import pandas as pd
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
raw_dir = os.path.join(current_dir, '../data/raw')
processed_dir = os.path.join(current_dir, '../data/processed')

# На всякий случай создаем папку processed
os.makedirs(processed_dir, exist_ok=True)

# Загружаем данные
sales = pd.read_csv(os.path.join(raw_dir, 'sales_daily.csv'))
matches = pd.read_csv(os.path.join(raw_dir, 'matches.csv'))
submission = pd.read_csv(os.path.join(raw_dir, 'submission_template.csv'))
zones = pd.read_csv(os.path.join(raw_dir, 'zones.csv'))

# Считаем фактические продажи таргет для всех прошедших матчей
train_sales = sales.groupby(['match_id', 'zone'])['tickets'].sum().reset_index()
train_sales = train_sales.rename(columns={'tickets': 'tickets_total'})

# Считаем среднее количество продаж для каждой зоны базовый прогноз
baseline_predictions = train_sales.groupby('zone')['tickets_total'].mean().reset_index()
baseline_predictions['tickets_total'] = np.round(baseline_predictions['tickets_total']).astype(int)

# Заполняем шаблон для ответа
submission_baseline = pd.merge(
    submission.drop(columns=['tickets_total']),
    baseline_predictions,
    on='zone',
    how='left'
)

# Проверяем жесткое ограничение - вместимость зоны минус абонементы
submission_baseline = pd.merge(
    submission_baseline, 
    zones[['match_id', 'zone', 'seats', 'season_tickets']], 
    on=['match_id', 'zone'], 
    how='left'
)
submission_baseline['max_capacity'] = submission_baseline['seats'] - submission_baseline['season_tickets']

# Обрезаем прогноз, если он превышает вместимость зоны
submission_baseline['tickets_total'] = np.minimum(
    submission_baseline['tickets_total'], 
    submission_baseline['max_capacity']
)

# Сохраняем итоговый файл ответа
output_path = os.path.join(processed_dir, 'predictions_baseline.csv')
final_baseline = submission_baseline[['match_id', 'zone', 'tickets_total']]
final_baseline.to_csv(output_path, index=False)

