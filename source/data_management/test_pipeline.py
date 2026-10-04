from hockey_data_generator import HockeyDataGenerator

generator = HockeyDataGenerator(raw_dir='data/raw')
generator.prepare()

first_row = generator.encoder.encoded_df.iloc[0]
sample_match_id = first_row['match_id']
sample_zone = first_row['zone_id_orig']

print(f"Тестируем на реальных данных -> match_id: {sample_match_id}, zone: {sample_zone}")

sample_vector = generator.get_match_zone_features(match_id=sample_match_id, zone=sample_zone)
print(f"Успех! Размерность вектора для нейросети: {len(sample_vector)}")
print(f"Первые 5 значений вектора: {sample_vector[:5]}")

X_train, Y_train = generator.get_train_data()
print(f"Всего примеров для обучения нейросети (развёрнутых по дням): {len(X_train)}")
print(f"Всего признаков в энкодере: {len(generator.encoder.feature_cols)}")
print("\nСписок столбцов, которые идут в нейросеть:")
for i, col_name in enumerate(generator.encoder.feature_cols):
    print(f"[{i+2}] {col_name}")
clean_vector = generator.get_match_zone_features(match_id=sample_match_id, zone=sample_zone)

print(clean_vector)

print(f"Длина вектора: {len(clean_vector)}")
print(f"Уникальные типы элементов в векторе: {set(type(x) for x in clean_vector)}")
