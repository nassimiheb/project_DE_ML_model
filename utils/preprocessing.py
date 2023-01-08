from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
import pandas as pd

def preprocessing(data: pd.DataFrame,
                  target_variable: str,
                  completion_rate: float = 0.5,
                  min_nb_unique: int = 1,
                  columns_transformations: dict = {}
                  ) -> pd.DataFrame:
    """ Data preparation for modeling

    Args:
        data (pd.DatFrame): data frame
        completion_rate (float): the expected value range is 0 to 1
            It's the threshold for completion rate
            We delete all columns that have completion rate less than this threshold
        min_nb_unique (int): number of unique values in feature to retain

    Returns:
        pd.DatFrame

    Notes:
        delete all columns with a completion rate less than ``completion_rate``
        also, remove columns which have a number of unique values <= ``min_nb_unique``

    """
    print(f"[UTILS] - START data preprocessing \n"
                 f"Keep columns which have completion rate >= {completion_rate} \n"
                 f"Drop columns which have a nb of unique values (excluding missing values) <= {min_nb_unique}")
    if data.empty:
        raise ValueError("[UTILS] - Data frame is empty !")
    # drop all columns which have a number of unique values <= ``min_nb_unique``
    data = data.loc[:, data.nunique(dropna=True) > min_nb_unique]

    # Apply the transformation on columns if they exist
    for column, transformation in columns_transformations.items():
        try:
            print(
                f"[UTILS] - Applying {transformation} to: {column}")
            data[column] = data[column].apply(transformation, errors='coerce')
        except KeyError:
            pass

    # drop columns which have completion rate under the threshold
    data = data.loc[:, data.notnull().mean() >= completion_rate]
    print(f"[UTILS] - List of selected features: {data.columns}")

    # convert object to str type
    print("[UTILS] - Start of transformation of object columns to ``str``")
    str_columns = data.select_dtypes(include=["object"]).columns
    data[str_columns] = data[str_columns].astype(str)
    print("[UTILS] - DONE data preprocessing")

    y_category_target = data[target_variable]
    x_input = data.drop([target_variable], axis=1)

    x_train = transform_columns(x_input)

    return (x_train, y_category_target)

def transform_columns(x_input):
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
    
    x_train = preprocessors.fit_transform(x_input)

    return x_train