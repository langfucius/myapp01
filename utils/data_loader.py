import pandas as pd


def load_data_csv(file):
    try:
        df = pd.read_csv(file)
        return df, None
    except Exception as e:
        return None, str(e)


def load_data_excel(file):
    try:
        df = pd.read_excel(file)
        return df, None
    except Exception as e:
        return None, str(e)
