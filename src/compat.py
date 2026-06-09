import pandas as pd


def _normalize_legacy_merge_series(series):
    return series.map(lambda value: '' if pd.isna(value) else str(value).strip().lower())


def _collapse_legacy_merge_suffixes(df, log_fn):
    legacy_bases = sorted({
        col[:-2] for col in df.columns
        if col.endswith('_x') and f'{col[:-2]}_y' in df.columns
    })
    for base in legacy_bases:
        x_col = f'{base}_x'
        y_col = f'{base}_y'
        mismatch = _normalize_legacy_merge_series(df[x_col]) != _normalize_legacy_merge_series(df[y_col])
        if mismatch.any():
            log_fn(f'WARNING: {int(mismatch.sum())} rows have {x_col} != {y_col}; dropping those rows')
            df = df.loc[~mismatch].reset_index(drop=True)
        df[base] = df[x_col]
        df.drop(columns=[x_col, y_col], inplace=True)

    orphan_x_cols = sorted(col for col in df.columns if col.endswith('_x') and col[:-2] not in df.columns)
    for x_col in orphan_x_cols:
        base = x_col[:-2]
        df.rename(columns={x_col: base}, inplace=True)
        y_col = f'{base}_y'
        if y_col in df.columns:
            df.drop(columns=[y_col], inplace=True)
    return df
