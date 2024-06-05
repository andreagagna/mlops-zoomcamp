import logging
import functools
import pandas as pd
import pickle
from pathlib import Path
import xgboost as xgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, root_mean_squared_error
import mlflow

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# MLflow setup
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("nyc-taxi-experiment")

search_space = {
    'max_depth': scope.int(hp.quniform('max_depth', 4, 100, 1)),
    'learning_rate': hp.loguniform('learning_rate', -3, 0),
    'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
    'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
    'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
    'objective': 'reg:linear',
    'seed': 42
}


def log_function_call(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        args_repr = [repr(a) for a in args]
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)
        logging.info(f"Calling {func.__name__}({signature})")
        value = func(*args, **kwargs)
        logging.info(f"{func.__name__} returned {value!r}")
        return value
    return wrapper


@log_function_call
def read_dataframe(filename):
    """
    Read and preprocess the dataframe from a Parquet file.

    Parameters:
        filename (Path): The path to the Parquet file.

    Returns:
        pd.DataFrame: Preprocessed dataframe.
    """
    try:
        df = pd.read_parquet(filename)
        df['lpep_dropoff_datetime'] = pd.to_datetime(df['lpep_dropoff_datetime'])
        df['lpep_pickup_datetime'] = pd.to_datetime(df['lpep_pickup_datetime'])
        df['duration'] = (df['lpep_dropoff_datetime'] - df['lpep_pickup_datetime']).dt.total_seconds() / 60
        df = df[(df['duration'] >= 1) & (df['duration'] <= 60)]
        df[['PULocationID', 'DOLocationID']] = df[['PULocationID', 'DOLocationID']].astype(str)
    except Exception as e:
        logging.error(f"Error reading or preprocessing the dataframe: {e}")
        raise
    return df



@log_function_call
def prepare_features(df, categorical, numerical, train=True, dv=None):
    """
    Prepare model features using DictVectorizer.

    Parameters:
        df (pd.DataFrame): Dataframe containing the features.
        categorical (list): List of categorical column names.
        numerical (list): List of numerical column names.
        train (bool, optional): Whether it is training or dev dataframe.
        dv (DictVectorizer, optional): Matrix of vectorizer parameters.

    Returns:
        pd.DataFrame: Train or Dev features dataframe.
        scipy.sparse matrix: Encoded features, if train is True.
    """
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    df_dicts = df[categorical + numerical].to_dict(orient='records')
    if train:
        dv = DictVectorizer()
        X = dv.fit_transform(df_dicts)
        return X, dv
    elif not train and dv is not None:
        return dv.transform(df_dicts)
    else:
        raise Exception("Options not recognized.")



@log_function_call
def objective(params, X_train, X_val, y_train, y_val):
    train = xgb.DMatrix(X_train, label=y_train)
    valid = xgb.DMatrix(X_val, label=y_val)

    with mlflow.start_run():
        mlflow.set_tag("model", "xgboost")
        mlflow.log_params(params)
        booster = xgb.train(
            params=params,
            dtrain=train,
            num_boost_round=100,
            evals=[(valid, 'validation')],
            early_stopping_rounds=50
        )
        y_pred = booster.predict(valid)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)

    return {'loss': rmse, 'status': STATUS_OK}



@log_function_call
def main(model='linear'):
    data_prefix = 'green_tripdata_2021-'
    train_data_name = f"{data_prefix}01.parquet"
    val_data_name = f"{data_prefix}02.parquet"

    train_data_path = Path.cwd().parent.joinpath('data', train_data_name)
    val_data_path = Path.cwd().parent.joinpath('data', val_data_name)

    df_train = read_dataframe(train_data_path)
    df_val = read_dataframe(val_data_path)

    categorical = ['PU_DO']
    numerical = ['trip_distance']

    X_train, dv = prepare_features(df_train, categorical, numerical, train=True)
    X_val = prepare_features(df_val, categorical, numerical, train=False, dv=dv)

    y_train = df_train['duration'].to_numpy()
    y_val = df_val['duration'].to_numpy()

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_val)
    rmse = root_mean_squared_error(y_val, y_pred)

    if model == 'linear':
        model_path = Path.cwd().joinpath('models', 'lin_reg.bin')
        with open(model_path, 'wb') as f_out:
            pickle.dump((dv, lr), f_out)

        with mlflow.start_run():
            mlflow.set_tag("developer", "andrea")
            mlflow.log_params({
                "train_data_path": str(train_data_path),
                "valid_data_path": str(val_data_path),
                "alpha": 0.1
            })
            mlflow.log_metric("rmse", rmse)
            mlflow.log_artifact(local_path=str(model_path), artifact_path="models_pickle")

    elif model == 'xgboost':
        best_result = fmin(
            fn=objective,
            space=search_space,
            algo=tpe.suggest,
            max_evals=50,
            trials=Trials()
        )

        mlflow.xgboost.autolog(disable=True)

        with mlflow.start_run():
            best_params = {
                'learning_rate': 0.09585355369315604,
                'max_depth': 30,
                'min_child_weight': 1.060597050922164,
                'objective': 'reg:linear',
                'reg_alpha': 0.018060244040060163,
                'reg_lambda': 0.011658731377413597,
                'seed': 42
            }

            mlflow.log_params(best_params)

            train = xgb.DMatrix(X_train, label=y_train)
            valid = xgb.DMatrix(X_val, label=y_val)

            booster = xgb.train(
                params=best_params,
                dtrain=train,
                num_boost_round=1000,
                evals=[(valid, 'validation')],
                early_stopping_rounds=50
            )

            y_pred = booster.predict(valid)
            rmse = root_mean_squared_error(y_val, y_pred)
            mlflow.log_metric("rmse", rmse)

            with open("models/preprocessor.b", "wb") as f_out:
                pickle.dump(dv, f_out)
            mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

            mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")


if __name__ == "__main__":
    main(model='xgboost')
