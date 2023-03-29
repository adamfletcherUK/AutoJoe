'''
Workflow:

Detect Number of String:
    - If String:
        -
    - If Number:
        -
#todo:
- handle null values and error values differently
- ability to output most likely and second most likely column type
- for the second method (scanning the data) can we detect errors etc - not needed for first pass

'''
import pandas as pd
import numpy as np
import tdda
import datetime
import iisignature
from joblib import load

from iisig_model.string_encoding import one_hot_input


def most_frequent(List):
    return max(set(List), key=List.count)


def get_date_format(s_date):
    #todo: change this function! I hate it so much
    date_patterns = ["%d-%m-%Y", "%Y-%m-%d", "%m-%d-%Y", "%d/%m/%Y", "%Y/%m/%d", "%m/%d/%Y"]
    for pattern in date_patterns:
        try:
          datetime.datetime.strptime(s_date, pattern).date()
          return pattern
        except:
            pass

def determine_sample_fract(speed):
    if speed == 3:
        sampling_frac = 0.001
    elif speed == 2:
        sampling_frac = 0.01
    elif speed == 1:
        sampling_frac = 0.1
    return sampling_frac


def default_generator(column: pd.Series):
    column = column.astype('str')
    max_len = max(column.apply(len))
    print(f'    Generator: regex')
    print(f'    Regex: [A-Z0-9]*{max_len}')
    print(f'    Notes: Unable to full resolve column type')


def column_detection(df, clf, speed=2):
    threshold = 0.6
    #todo: if error is > a threshold create a default generator
    n_rows = df.shape[0]
    sampling_frac = determine_sample_fract(speed)
    sampling_number = int(np.ceil(n_rows * sampling_frac))
    if sampling_number <= 100:
        sampling_number = 100
    print(f'Sampling Size: {sampling_number}')

    for col in df:
        print(f'{col}:')
        col_type = most_frequent([type(df[col][i]) for i in np.random.randint(n_rows, size=sampling_number)])

        if len(df[col].value_counts()) <= 10:
            print('    Generator: Choice')
            print(f'    options: {df[col].unique().astype(str)}')

        elif (col_type in [np.int64]) & (len(df[col].unique()) == len(df[col])):
            print('    Generator: Index')
            print(f'    lbound: {min(df[col])}')
            print(f'    ubound: {max(df[col])}')

        elif col_type in [np.int64]:  # todo: there are multiple int formats!
            int_col = pd.to_numeric(df[col], errors='coerce')
            count_na = len(int_col.loc[int_col.isna()])
            error = round(count_na / n_rows, 2)
            if error > threshold:
                default_generator(df[col])
            else:
                print('    Generator: Int')
                print(f'    lbound: {min(int_col)}')
                print(f'    ubound: {max(int_col)}')
                print(f'    error: {error}')

        elif col_type in [float, np.float64]:
            float_col = pd.to_numeric(df[col], errors='coerce')
            count_na = len(float_col.loc[float_col.isna()])
            error = round(count_na / n_rows, 2)
            if error > threshold:
                default_generator(df[col])
            else:
                print('    Generator: float')
                print(f'    lbound: {min(float_col)}')
                print(f'    ubound: {max(float_col)}')
                print(f'    error: {error}')

        #todo: refactor this!!
        elif (col_type == str) & (most_frequent([get_date_format(df[col][i]) for i in range(sampling_number)]) != None):
            date_format = most_frequent([get_date_format(df[col][i]) for i in range(sampling_number)])
            print('    Generator: Date')
            print(f'    format: {date_format}')
            print(f'    lbound: {min(df[col])}')
            print(f'    ubound: {max(df[col])}')

        elif col_type == str:
            df[col] = df[col].astype(str)
            X = [iisignature.sig(one_hot_input(df.iloc[i][col].lower(), 41), 2) for i in range(sampling_number)]
            ypred = clf.predict(X).tolist()
            col_pred = most_frequent(ypred)
            print(f'    Generator: {col_pred}')
            if col_pred == 'regex':
                regex = tdda.rexpy.rexpy.Extractor(df[col]).results.rex[0]
                print(f'    Regex: {regex}')

        else:
            default_generator(df[col])

def find_most_recent_model():
    ...


def pipeline(df, speed=2):
    clf = load('../models/iisig_clf.joblib')
    column_detection(df, clf, speed)


if __name__ == '__main__':
    from detector.make_test_data import make_test_data
    print('Synthetic Data')
    df = make_test_data()
    pipeline(df, speed=1)
    print()
    print('Titanic data')
    titanic = pd.read_csv('../testing_data/titanic.csv')
    pipeline(titanic, 2)
