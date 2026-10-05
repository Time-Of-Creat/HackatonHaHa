from pathlib import Path


class FoldersData:
    PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
    DATA_DIR = PROJECT_DIR / "data"
    LOGS_FILE = PROJECT_DIR / "top_architectures.txt"


folders_data = FoldersData()
