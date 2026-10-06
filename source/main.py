from source.application import Application


def learn_and_validate():
    """Функция для обучения и валидации нейросети"""
    train_data = app.data_generator.get_train_data()
    validation_data = app.data_generator.get_val_data()

    app.nn.fit(train_data[0], train_data[1])
    validation_res = app.nn.validate(validation_data[0], validation_data[1])['rmse']
    print(f"Результат валидации (RMSE): {validation_res}")

    return validation_res


def form_predictions_table(start_game: int = 86, end_game: int = 102):
    """Функция для формирования таблицы предсказаний"""
    pred_table = [
        ["match_id", "zone", "tickets_total"]
    ]

    for game in range(start_game, end_game+1):
        for zone in range(1, 9):
            pred = app.get_game_zone_prediction(f"M{game:03d}", str(zone))
            pred_table.append([f"M{game:03d}", str(zone), pred])

    print("Предсказания получены, формируем таблицу predictions.csv")
    app.table_creator.form_csv_table(pred_table)
    print("Таблица предсказаний сформирована!")


if __name__ == "__main__":
    app = Application()
    
    learn_and_validate()
    form_predictions_table()

