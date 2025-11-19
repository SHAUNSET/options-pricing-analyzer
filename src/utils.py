import pandas as pd

def make_datetime_naive(df, columns):
    """
    Convert timezone-aware datetime columns to naive datetimes for Excel.
    """
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col]).dt.tz_localize(None)
    return df
