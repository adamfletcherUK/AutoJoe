import iisignature
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
import joblib
from datetime import datetime
from iisig_model.generate_training_data import TrainingDataMaker
from iisig_model.string_encoding import one_hot_input


class TrainModel:
    def __init__(self):
        self.training_data = TrainingDataMaker().main()
        self.max_value = 41
        self.logging = True

    def calculate_iisignature(self):
        X = [iisignature.sig(one_hot_input(self.training_data.iloc[i]['Value'].lower(),
                                           self.max_value),
                             2) for i in range(self.training_data.shape[0])]
        return X

    def test_train_split(self, X):
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            self.training_data['Label'],
                                                            test_size = 0.33,
                                                            random_state = 42)
        return {'X_train': X_train, 'y_train': y_train,
                'X_test': X_test, 'y_test': y_test}

    def train_model(self, X_train, y_train):
        print('Training Model')
        clf = make_pipeline(#StandardScaler(),
                            SVC(gamma='scale', verbose=True))
        clf.fit(X_train, y_train)
        print('Finished Training')
        return clf

    def test_model(self, clf, X_test, y_test, timestamp):
        print('Generating Accuracy Statistics')
        ypred = clf.predict(X_test)
        with open(f'../models/config-{timestamp}.txt', 'w') as f:
            f.write(f'Configuration for iisig_clf_{timestamp}.joblib\n')
            f.write(f'    Python Version: {sys.version}\n')
            f.write(f'    Pandas Version: {pd.__version__}\n')
            f.write(f'    Joblib Version: {joblib.__version__}\n\n')
            f.write(f'Classification Report\n')
            f.write(classification_report(y_test, ypred))
            f.write('\n\n')
            f.write('Model Performance (average == macro)\n')
            f.write(f'    Accuracy Score:  {accuracy_score(y_test, ypred)}\n')
            f.write(f'    F1 Score:        {f1_score(y_test, ypred, average="macro")}\n')
            f.write(f'    Precision Score: {precision_score(y_test, ypred, average="macro")}\n')
            f.write(f'    Recall Score:    {recall_score(y_test, ypred, average="macro")}\n')

        print(classification_report(y_test, ypred))

    def export_model(self, clf, timestamp):
        joblib.dump(clf, f'../models/iisig_clf_{timestamp}.joblib')
        print('Finished dumping model')


    def get_timestamp(self):
        timestamp_str = datetime.now()
        timestamp_str = str(timestamp_str)
        timestamp_str = timestamp_str.split('.')[0]
        timestamp_str = timestamp_str.replace(' ', '_')
        timestamp_str = timestamp_str.replace(':', '-')
        return timestamp_str

    def pipeline(self):
        timestamp = self.get_timestamp()
        X = self.calculate_iisignature()
        X_dict = self.test_train_split(X)
        clf = self.train_model(X_dict['X_train'], X_dict['y_train'])
        self.test_model(clf, X_dict['X_test'], X_dict['y_test'], timestamp)
        self.export_model(clf, timestamp)
        return clf


if __name__ == '__main__':
    tm = TrainModel()
    tm.pipeline()