
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, classification_report, roc_auc_score,roc_curve,auc
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd

def fit(X, y, ts, rs, model,accuracy_list ,f1_list,roc_auc_list):
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

     #roc-auc curve of model
    fpr,tpr,threshold = roc_curve(y_test,pred)
    auc_value = auc(fpr,tpr)
    rocauc_score = roc_auc_score(y_test, pred)
    roc_auc_list.append(rocauc_score)
    plt.figure(figsize=(5,5),dpi=100)
    print("ROC-AUC Score: ", f1)
    print("-" * 100)
    print()
    plt.plot(fpr,tpr,linestyle='-',label = "(auc_value = %0.3f)" % auc_value)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()
    print()
    
    #confusion matrix for model
    print("Confusion Matrix: ")
    plt.figure(figsize=(10, 5))
    sns.heatmap(confusion_matrix(y_test, pred), annot=True, fmt='g');
    plt.title('Confusion Matrix', fontsize=20)

def get_train_test_split(x_input, y_target, test_size, random_state):
    return train_test_split(x_input, y_target, test_size=test_size, stratify=y_target, random_state=random_state)
    # return train_test_split(X, y, test_size=ts, random_state=rs)

#Load Data
df_train = pd.read_csv('hotel.csv')

df_train.dtypes

df_train = df_train.drop(['reservation_status_date'], axis=1)

#Preprocessing
def preprocessing(data: pd.DataFrame,
                  completion_rate: float = 0.5,
                  min_nb_unique: int = 1,
                  columns_transformations: dict = {}
                  ) -> pd.DataFrame:
    
    logging.info(f"[COMMON] - START data preprocessing \n"
                 f"Keep columns which have completion rate >= {completion_rate} \n"
                 f"Drop columns which have a nb of unique values (excluding missing values) <= {min_nb_unique}")
    if data.empty:
        raise ValueError("[COMMON] - Data frame is empty !")
    # drop all columns which have a number of unique values <= ``min_nb_unique``
    data = data.loc[:, data.nunique(dropna=True) > min_nb_unique]

    # Apply the transformation on columns if they exist
    for column, transformation in columns_transformations.items():
        try:
            logging.info(
                f"[COMMON] - Applying {transformation} to: {column}")
            data[column] = data[column].apply(transformation, errors='coerce')
        except KeyError:
            pass

    # drop columns which have completion rate under the threshold
    data = data.loc[:, data.notnull().mean() >= completion_rate]
    logging.info(f"[COMMON] - List of selected features: {data.columns}")

    # convert object to str type
    logging.info("[COMMON] - Start of transformation of object columns to ``str``")
    str_columns = data.select_dtypes(include=["object"]).columns
    data[str_columns] = data[str_columns].astype(str)
    logging.info("[COMMON] - DONE data preprocessing")
    return data

# data cleaning
df_train_prepro = preprocessing(df_train, completion_rate=MIN_COMPLETION_RATE)

# target & features extraction
label = "is_canceled"
y_category_target = df_train_prepro[label]
x_input = df_train_prepro.drop([label], axis=1)

num_cols = x_input.select_dtypes(include=np.number).columns.tolist()
num_cols

NUMERIC_TRANSFORMERS = [('SimpleImputer', SimpleImputer(strategy="median")), ('MinMaxScaler', MinMaxScaler())]
CATEGORICAL_TRANSFORMERS = [('SimpleImputer', SimpleImputer(strategy="constant", fill_value="missing")),
                            ('OneHotEncoder', OneHotEncoder(handle_unknown="ignore", drop=None))]

numeric_transformer = Pipeline(NUMERIC_TRANSFORMERS)
categorical_transformer = Pipeline(CATEGORICAL_TRANSFORMERS)

preprocessors = ColumnTransformer(
            transformers=[
                (
                    "number",
                    numeric_transformer,
                    make_column_selector(dtype_include=["number", "float64"]),
                ),
                (
                    "category",
                    categorical_transformer,
                    make_column_selector(dtype_include=["object", "bool"]),
                ),
            ]
        )

