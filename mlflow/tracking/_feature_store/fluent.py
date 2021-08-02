import pandas as pd 

from mlflow.tracking.client import MlflowClient

def ingest(source, entity_name):
    """
    Batch load feature data to publish into offline store.
    Params:
        source (str or pd.Dataframe): Either a file path to parquet file to ingest batch data into offline store.
        features (List[MLFeature]): A list of MLFeature objects that should be ingested into the offline store.
    Returns:
        Dataframe of all data ingested with columns of entity_name and datetime.
    Example usage:
        mlflow.ingest(source=“data/drivers.parquet”, feature_keys=[{"name":"avg_cost", "type": ValueType.INT64}]) 
    """
    return MlflowClient().ingest(source, entity_name)

def retrieve(feature_list, entity_df) -> pd.DataFrame:
    """
    Get features that have been registered already into the offline store.
    Params:
        feature_list (List[str]): A dictionary containg a key of parquet source and 
    value of list of MLFeature objects that should be retrieved from the offline store. 
    Returns:
        Some object with the features that can be used for batch inferencing or training.
    Example usage:
        quality = MLFeature("quality", "int64")
        alcohol = MLFeature("alcohol", "float32")
        feature_keys = [quality, alcohol]
        feature_df = mlflow.retrieve(feature_keys, entity_df)
    """
<<<<<<< HEAD
    return MlflowClient().retrieve(feature_keys, entity_df)

def search_features(database,filter_string):
    return MlflowClient().search_features(database, filter_string)
=======
    return MlflowClient().retrieve(feature_list, entity_df)
>>>>>>> 78291046723eb8da4accb93c55ce6e72c847175b
