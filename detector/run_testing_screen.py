import pandas as pd
import glob

from detector.make_test_data import make_test_data
from detector.column_detector import pipeline

def scan_for_data(filepath):
    glob.glob(filepath)

if __name__ == '__main__':

    print('Synthetic Data')
    df = make_test_data()
    pipeline(df)
    print()

    for file in glob.glob('../testing_data/*.csv'):
        print(file)
        df = pd.read_csv(file)
        pipeline(df)
        print()
    for file in glob.glob('../testing_data/*/*.csv'):
        print(file)
        df = pd.read_csv(file)
        pipeline(df)
        print()

    # df = pd.read_csv('../testing_data/BasicCompanyData-2023-03-01-part1_7.csv')
    # pipeline(df, 3)