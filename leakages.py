def time_leakage(df):
    id_cols = [c for c in df.columns if c.startswith('id')]
    dt_cols = [c for c in df.columns if c.startswith('datetime')]
    if id_cols and dt_cols:
        num_cols = [c for c in df if c.startswith('number')]
        for id_col in id_cols:
            group = df.groupby(by=id_col).get_group(df[id_col].iloc[0])
            for dt_col in dt_cols:
                sorted_group = group.sort_values(dt_col)
                for lag in range(-1, -10, -1):
                    for col in num_cols:
                        corr = sorted_group['target'].corr(sorted_group[col].shift(lag))
                        if corr >= 0.99:
                            return {'is_leakage': True, 'num_col': col, 'lag': lag, 'id_col': id_col, 'dt_col': dt_col}

    return {'is_leakage': False}


def use_time_leakage(df, leak_params):
    if 'prediction' not in df:
        df['prediction'] = 0

    for name, group in df.groupby(by=leak_params['id_col']):
        gr = group.sort_values(leak_params['dt_col'])
        df.loc[gr.index, 'prediction'] = gr[leak_params['num_col']].shift(leak_params['lag'])

    return df
