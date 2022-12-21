
from sklearn.model_selection import train_test_split

def fit(X, y, ts, rs, model):
    #train test split
    (X_train, X_test, y_train, y_test) = get_train_test_split(X, y, ts, rs)
    
    
    #fit on data
    model_out = model.fit(X_train, y_train)
    
    #prediction
    pred = model.predict(X_test)

def get_train_test_split(x_input, y_target, test_size, random_state):
    return train_test_split(x_input, y_target, test_size=test_size, stratify=y_target, random_state=random_state)
    # return train_test_split(X, y, test_size=ts, random_state=rs)