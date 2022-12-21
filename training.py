
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

def fit(X, y, ts, rs, model,accuracy_list ,f1_list):
    #train test split
    (X_train, X_test, y_train, y_test) = get_train_test_split(X, y, ts, rs)
    
    
    #fit on data
    model_out = model.fit(X_train, y_train)
    
    #prediction
    pred = model.predict(X_test)

    #performance of model
    print("Classification Report: \n", classification_report(y_test, pred))
    print("-" * 100)
    print()
    
    #accuracy of model
    acc = accuracy_score(y_test, pred)
    accuracy_list.append(acc)
    print("Accuracy Score: ", acc)
    print("-" * 100)
    print()

    #f1-score of model
    f1 = f1_score(y_test, pred)
    f1_list.append(f1)
    print("F1 Score: ", f1)
    print("-" * 100)
    print()

def get_train_test_split(x_input, y_target, test_size, random_state):
    return train_test_split(x_input, y_target, test_size=test_size, stratify=y_target, random_state=random_state)
    # return train_test_split(X, y, test_size=ts, random_state=rs)