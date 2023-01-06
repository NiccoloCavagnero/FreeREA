import os
import re
import pickle
import numpy as np
import pandas as pd
import gzip


def load_metrics(dataset_metric_folder: str, include_accuracies: bool = False, include_times: bool = True):
    # List of CSV files
    paths = [os.path.join(dataset_metric_folder, f) for f in os.listdir(dataset_metric_folder)]

    # Load all of them into a DataFrame
    df = None

    for file_ in paths:
        # Do not include accuracy if not required
        if not re.search(r'accuracies\.(?:csv|pkl)(?:\.gz)?$', file_) or include_accuracies:
            # Read metric
            open_fn = open
            if file_.endswith('.gz'):
                open_fn = gzip.open

            with open_fn(file_, 'rb') as fp:
                if re.search(r'\.csv(?:\.gz)?$', file_):
                    other = pd.read_csv(fp)
                elif re.search(r'\.pkl(?:\.gz)?$', file_):
                    other = pickle.load(fp)
                else:
                    continue

            if len(other.columns) > 2:
                metric = other.columns[1]
                if include_times:
                    other.rename(columns={'time=[s]': metric + '_time=[s]'}, inplace=True)

                # Remove time 'pure' columns
                other.drop(columns=[cn for cn in other.columns if cn.startswith('time')], inplace=True)

            # Merge
            if df is not None:
                df = df.merge(other, on='index')
            else:
                df = other

    df.drop(columns='index', inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.min(), inplace=True)

    return df
