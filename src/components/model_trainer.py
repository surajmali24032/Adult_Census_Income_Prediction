import sys
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
#from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object
from dataclasses import dataclass


@dataclass
class ModelTrainerconfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerconfig()

    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info('Splitting Independent and dependent variables form train and test array')
            X_train, y_train, X_test, y_test =(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            model = RandomForestClassifier(n_estimators=100, random_state=42)  # Initialize Random Forest classifier
            model.fit(X_train, y_train)  # Fit the model to the training data


            y_pred = model.predict(X_test)
            test_score = accuracy_score(y_test,y_pred)
            report = classification_report(y_test,y_pred)

            print(f'Model Name: Random Forest, Accuracy Score: {test_score}')
            logging.info(f'Model Name: Random Forest, Accuracy Score: {test_score}')


            print('\n====================================================================================\n')
            print(f'classification Report: {report}')

            logging.info(f'classification Report: {report}')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = model
            )

        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise CustomException(e,sys)