

def fit(X, y, ts, rs, model):
    #train test split
    (X_train, X_test, y_train, y_test) = get_train_test_split(X, y, ts, rs)
    
    
    #fit on data
    model_out = model.fit(X_train, y_train)
    
    #prediction
    pred = model.predict(X_test)