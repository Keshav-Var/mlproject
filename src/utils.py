import os
import sys
import pickle
from src.exception import CustomException
from sklearn.metrics import r2_score

def save_object(file_path,obj):
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)

        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)

def load_obj(file_path):
    try:
        with open(file_path,'rb') as f:
            return pickle.load(f)
    except Exception as e:
        raise CustomException(e,sys)

def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        model_list = []
        r2_list = []

        for name,model in models.items():
            model.fit(X_train,y_train)

            #make train and test predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            #evalutions of the models
            train_r2 = r2_score(y_train,y_train_pred)
            test_r2 = r2_score(y_test,y_test_pred)

            model_list.append(name)
            r2_list.append(test_r2)

        return dict(zip(model_list,r2_list))
    except Exception as e:
        raise CustomException(e,sys)