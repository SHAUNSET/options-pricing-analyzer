# src/utils.py
import pandas as pd

def make_datetime_naive(df, columns):
    """
    Convert timezone-aware datetime columns to naive datetimes for Excel.
    
    Args:
        df (pd.DataFrame): DataFrame containing datetime columns.
        columns (list): List of column names to convert.
    
    Returns:
        pd.DataFrame: DataFrame with naive datetime columns.
    """
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col]).dt.tz_localize(None)
    return df
