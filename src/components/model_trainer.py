from dataclasses import dataclass
import os
import sys
from src.utils import evaluate_model, save_object

from catboost import CatBoostRegressor
from matplotlib.widgets import Lasso
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.logger import logging
from src.exception import CustomException

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformer


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Doing the train and test split")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Linear Regression" : LinearRegression(),
                "Lasso" : Lasso(),
                "Ridge" : Ridge(),
                "K-Neighbors Regressor":KNeighborsRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "Random Forest Regressor" : RandomForestRegressor(),
                "XGB Regressor" : XGBRegressor(),
                "AdaBoost Regressor":AdaBoostRegressor(),
                "CatBoost Regressor": CatBoostRegressor(verbose=False),
            }

            params = {

                "Linear Regression": {},
                "Lasso": {
                    'alpha': [0.001, 0.01, 0.1, 1, 10, 100],
                    'max_iter': [10000, 20000]
                },
                "Ridge": {
                    'alpha': [0.01, 0.1, 1, 10, 100]
                },
                "K-Neighbors Regressor": {
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree'],
                    'p': [1, 2]
                },
                "Decision Tree": {
                    'criterion': ['squared_error', 'poisson', 'absolute_error'],
                    'splitter': ['best', 'random'],
                    'max_depth': [None, 5, 10, 15, 20],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },

                "Random Forest Regressor": {
                    'n_estimators': [100, 200, 300],
                    'criterion': ['squared_error', 'absolute_error'],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None]
                },

                "XGB Regressor": {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 1.0],
                    'colsample_bytree': [0.8, 1.0]
                },

                "AdaBoost Regressor": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.05, 0.1, 1.0],
                    'loss': ['linear', 'square', 'exponential']
                },

                "CatBoost Regressor": {
                    'iterations': [100, 200, 300],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'depth': [4, 6, 8, 10],
                    'l2_leaf_reg': [1, 3, 5, 10]
                }
            }

            model_report:dict = evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                               models=models,params=params)
            best_model_name = max(model_report,key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            
            logging.info("Best model on both training and testing dataset")

            save_object(self.model_trainer_config.trained_model_file_path,obj=best_model)

            return best_model_score

        except Exception as e:
            raise CustomException(e,sys)


if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformer()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))
