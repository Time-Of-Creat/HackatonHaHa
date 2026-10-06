import csv

from source.file_manager.folders_data import folders_data


class CSVTableCreator:
    @staticmethod
    def form_csv_table(table: list[list]):
        with open(str(folders_data.PROJECT_DIR) + "\\predictions.csv", "w", newline="", encoding="utf-8-sig") as file:
            writer = csv.writer(file)
            writer.writerows(table)
