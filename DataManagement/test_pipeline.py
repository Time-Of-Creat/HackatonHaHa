from data_generator import HockeyDataGenerator  # Импортируем класс-фасад

# 1. Инициализируем наш генератор с указанием папки с исходными файлами
generator = HockeyDataGenerator(raw_dir='data/raw')
# Запускаем подготовку данных
generator.prepare()

# 2. Берем первый реальный матч и зону из загруженной таблицы для теста
first_row = generator.encoder.encoded_df.iloc[0]
sample_match_id = first_row['match_id']
sample_zone = first_row['zone_id_orig']

# Выводим инфо о тестовом матче
print(f"Тестируем на реальных данных -> match_id: {sample_match_id}, zone: {sample_zone}")

# 3. Запрашиваем одиночный вектор для инференса
sample_vector = generator.get_match_zone_features(match_id=sample_match_id, zone=sample_zone)
# Выводим подтверждение размерности вектора
print(f"Успех! Размерность вектора для нейросети: {len(sample_vector)}")
# Печатаем первые 5 элементов вектора для визуальной проверки
print(f"Первые 5 значений вектора: {sample_vector[:5]}")

# 4. Проверяем генерацию обучающего датасета
X_train, Y_train = generator.get_train_data()
# Печатаем общее количество развернутых обучающих примеров
print(f"Всего примеров для обучения нейросети (развёрнутых по дням): {len(X_train)}")
