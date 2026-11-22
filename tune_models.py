from pathlib import Path
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile
from itertools import product
from time import perf_counter
import warnings
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

ROOT = Path(__file__).resolve().parent
HORIZONS = {'D': 61, 'W': 8, 'M': 2}
OFFSETS = {'D': [42, 21, 0], 'W': [8, 4, 0], 'M': [1, 0]}
FEATURES = {'D': ([1, 2, 7, 14, 28], [7, 28]),
            'W': ([1, 2, 4, 8], [4, 8]), 'M': ([1, 2], [2])}

def make_row(history, date, freq):
    lags, windows = FEATURES[freq]
    row = {f'lag_{lag}': float(history[-lag]) for lag in lags}
    for window in windows:
        recent = np.asarray(history[-window:], dtype=float)
        row[f'mean_{window}'] = recent.mean()
        row[f'std_{window}'] = recent.std()
    row['time_index'] = len(history)
    if freq == 'D':
        row['dow_sin'] = np.sin(2 * np.pi * date.day_of_week / 7)
        row['dow_cos'] = np.cos(2 * np.pi * date.day_of_week / 7)
        row['saturday'] = int(date.day_of_week == 5)
    return row

def fit_forecast(name, train, dates, freq, config):
    history = train.iloc[-config['window']:] if config['window'] else train
    y = np.log1p(history) if config['log'] else history
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        if name == 'ETS':
            model = ExponentialSmoothing(y, initialization_method='estimated', **config['params'])
            fit = model.fit(use_brute=False)
            if not fit.mle_retvals.success:
                raise ValueError('ETS convergence')
            pred = np.asarray(fit.forecast(len(dates)))
        elif name == 'ARIMA':
            fit = ARIMA(y, **config['params']).fit(method_kwargs={'maxiter': 250, 'disp': 0})
            if not fit.mle_retvals.get('converged', True):
                raise ValueError('ARIMA convergence')
            pred = np.asarray(fit.forecast(len(dates)))
        else:
            lags, windows = FEATURES[freq]
            warmup = max(lags + windows)
            values = y.to_numpy()
            x = pd.DataFrame([make_row(values[:i], y.index[i], freq)
                              for i in range(warmup, len(y))])
            model = CatBoostRegressor(**config['params'], random_seed=123,
                                     thread_count=2, verbose=False, allow_writing_files=False)
            model.fit(x, y.iloc[warmup:])
            values, pred = y.to_list(), []
            for date in dates:
                value = max(0, float(model.predict(pd.DataFrame([make_row(values, date, freq)]))[0]))
                pred.append(value)
                values.append(value)
            pred = np.asarray(pred)
        if config['log']:
            pred = np.expm1(pred)
    pred = np.maximum(0, pred)
    if not np.isfinite(pred).all():
        raise ValueError('nonfinite prediction')
    return pred, ' | '.join(sorted({str(w.message) for w in caught}))

def grid(name, freq):
    windows = [None, 112] if freq == 'D' else [None, 26] if freq == 'W' else [None]
    if name == 'ARIMA':
        orders = [(1, 0, 0), (1, 0, 1), (0, 1, 0), (0, 1, 1), (1, 1, 0), (1, 1, 1)]
        if freq == 'M':
            orders = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 1, 1)]
        seasons = [(0, 0, 0, 0), (1, 0, 0, 7), (0, 1, 1, 7)] if freq == 'D' else [(0, 0, 0, 0)]
        params = [{'order': order, 'seasonal_order': seasonal,
                   'trend': 'c' if order[1] + seasonal[1] == 0 else 'n'}
                  for order, seasonal in product(orders, seasons)]
    elif name == 'ETS':
        params = []
        for trend, seasonal in product([None, 'add'], [None, 'add'] if freq == 'D' else [None]):
            p = {'trend': trend, 'seasonal': seasonal}
            if trend:
                p['damped_trend'] = True
            if seasonal:
                p['seasonal_periods'] = 7
            params.append(p)
    else:
        base = ([{'depth': 1, 'iterations': 100, 'learning_rate': 0.05, 'l2_leaf_reg': 3},
                 {'depth': 2, 'iterations': 200, 'learning_rate': 0.03, 'l2_leaf_reg': 10},
                 {'depth': 2, 'iterations': 80, 'learning_rate': 0.03, 'l2_leaf_reg': 20}]
                if freq == 'M' else
                [{'depth': 2, 'iterations': 200, 'learning_rate': 0.05, 'l2_leaf_reg': 3},
                 {'depth': 3, 'iterations': 400, 'learning_rate': 0.03, 'l2_leaf_reg': 10},
                 {'depth': 2, 'iterations': 120, 'learning_rate': 0.05, 'l2_leaf_reg': 10},
                 {'depth': 3, 'iterations': 200, 'learning_rate': 0.03, 'l2_leaf_reg': 10}])
        params = [{**p, 'loss_function': loss} for p, loss in product(base, ['RMSE', 'MAE'])]
    return [{'window': window, 'log': log, 'params': p}
            for window, log, p in product(windows, [False, True], params)]

def load_series():
    path = ROOT / 'data' / 'raw' / 'online+retail.zip'
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        url = 'https://archive.ics.uci.edu/static/public/352/online+retail.zip'
        with urlopen(url, timeout=90) as response:
            path.write_bytes(response.read())
    with ZipFile(BytesIO(path.read_bytes())) as archive:
        frame = pd.read_excel(archive.open('Online Retail.xlsx'), dtype={
            'InvoiceNo': 'string', 'StockCode': 'string',
        }, parse_dates=['InvoiceDate'])
    sales = frame.loc[frame.StockCode.eq('47566') & frame.Quantity.gt(0)
                      & ~frame.InvoiceNo.str.startswith('C', na=False)]
    calendar = pd.date_range(frame.InvoiceDate.min().normalize(),
                             frame.InvoiceDate.max().normalize(), freq='D')
    daily = (sales.set_index('InvoiceDate').Quantity.resample('D').sum()
             .reindex(calendar, fill_value=0).astype(float)).iloc[:-1]
    return {'D': daily, 'W': daily.resample('W-SUN').sum().iloc[1:-1],
            'M': daily.resample('MS').sum().iloc[:-1]}

def main():
    started = perf_counter()
    for freq, series in load_series().items():
        train = series.iloc[:-HORIZONS[freq]]
        cutoffs = [len(train) - HORIZONS[freq] - offset for offset in OFFSETS[freq]]
        for name in ['ETS', 'ARIMA', 'CatBoost']:
            candidates = grid(name, freq)
            print(freq, name, len(candidates), 'candidates', flush=True)
            successful = []
            for candidate, config in enumerate(candidates):
                scores = []
                for fold, cutoff in enumerate(cutoffs, 1):
                    fit = train.iloc[:cutoff]
                    valid = train.iloc[cutoff:cutoff + HORIZONS[freq]]
                    try:
                        pred, messages = fit_forecast(name, fit, valid.index, freq, config)
                        mae = float(np.abs(valid.to_numpy() - pred).mean())
                        scores.append(mae)
                    except (ValueError, RuntimeError, np.linalg.LinAlgError):
                        scores = []
                        break
                if len(scores) == len(cutoffs):
                    successful.append((float(np.mean(scores)), candidate, config))
                if (candidate + 1) % 12 == 0:
                    print('  completed', candidate + 1, 'elapsed', round(perf_counter() - started, 1), flush=True)
            score, candidate, config = min(successful, key=lambda item: (item[0], item[1]))
            print('BEST', round(score, 3), config, flush=True)
    print('COMPLETED', round(perf_counter() - started, 1), flush=True)

if __name__ == '__main__':
    main()
