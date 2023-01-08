
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, classification_report, roc_auc_score,roc_curve,auc

import pandas as pd

def fit_predict(X, y, ts, rs, model):
    #train test split
    (X_train, X_test, y_train, y_test) = get_train_test_split(X, y, ts, rs)
    
    
    #fit on data
    model_out = model.fit(X_train, y_train)
    
    #prediction
    pred = model.predict(X_test)

    result = ""

    #performance of model
    result += "Classification Report: \n", classification_report(y_test, pred)
    result += "-" * 100 + "\n"
    
    #accuracy of model
    acc = accuracy_score(y_test, pred)

    result += "Accuracy Score: " + str(acc)
    result += "-" * 100 + "\n"

    #f1-score of model
    f1 = f1_score(y_test, pred)
    f1_list.append(f1)
    result += "F1 Score: " + str(f1)
    result += "-" * 100 + "\n"

     #roc-auc curve of model
    fpr,tpr,threshold = roc_curve(y_test,pred)
    auc_value = auc(fpr,tpr)
    rocauc_score = roc_auc_score(y_test, pred)

    result += "ROC-AUC Score: " + rocauc_score
    result += "-" * 100 + "\n"

    return result
    
def get_train_test_split(x_input, y_target, test_size, random_state):
    return train_test_split(x_input, y_target, test_size=test_size, stratify=y_target, random_state=random_state)