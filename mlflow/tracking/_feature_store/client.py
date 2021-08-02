from google.protobuf.duration_pb2 import Duration
from feast.feature_store import FeatureStore
from feast import Entity, FeatureView, Feature, ValueType, FileSource
import pandas as pd
import uuid
import sqlite3
import os 

class FeatureStoreClient(object):
    """
    Client of an MLflow Feature Store that ingests and retrieves batch data.
    """
    def __init__(self):
        self.fs = FeatureStore(
            repo_path="."
        )

    def ingest(self, source, entity_name) -> pd.DataFrame:

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
        # creating Feast objects and infrastructure
        file_stats = FileSource(
            path=source,
            event_timestamp_column="datetime",
            created_timestamp_column="created",
        )

        source_df = pd.read_parquet(source)
        entity_type = source_df.dtypes[entity_name]
        entity_type = self._convertToValueType(entity_type)
        entity = Entity(name=entity_name, value_type=entity_type, description="")

        entity_df = self._create_entity(source, entity_name)

        f_name = os.path.basename(source)
        view_name = os.path.splitext(f_name)[0]

        features = self._get_features(source_df.drop([entity_name, "created", "datetime"], axis=1)) 

        # update metadata with group uuids for Cataloging and Lineage API
        self._update_metadata(features, source, entity_name)
        self._register_dataset(features, source)

        feature_view = FeatureView(
            name=view_name,
            entities=[entity.name],
            ttl=Duration(seconds=86400 * 1),
            features=[Feature(name=feature.name, dtype=feature.type) for feature in features],
            online=False, 
            input=file_stats,
            tags={},
        )

        # creates and updates registry.db metadata
        self.fs.apply([entity, feature_view])
        return entity_df

    def retrieve(self, feature_list, entity_df) -> pd.DataFrame:

        """
        Get features that have been registered already into the offline store.
        Params:
            features (List[str]): A dictionary containg a key of parquet source and 
        value of list of MLFeature objects that should be retrieved from the offline store. 
            entity_df: pandas DataFrame containing entity_name and event_timestamp columns of data to be retrieved.
        Returns:
            pandas DataFrame with the features that can be used for batch inferencing or training.
        Example usage:
            feature_keys = ["alcohol", "quality"]
            feature_df = mlflow.retrieve(feature_keys, entity_df)
        """
        entity_name = list(entity_df.drop(["event_timestamp"], axis=1))[0] # future note: assumes only thing left in df is the entity
        refs = []
        conn = sqlite3.connect('data/metadata.db')
        for feature in feature_list:
            # from metadata.db, grab the view_name whose feature str matches feature AND has the same entity name
            view_query = f"SELECT view_name FROM GROUP_UUID_DATA WHERE feature='{feature.name}' and entity_name='{entity_name}';"
            df = pd.read_sql_query(view_query,conn)
            view = df['feature_uuid'].values[0]
            refs.append("{}:{}".format(view,feature))    
        conn.commit()
        conn.close()

        # retrieving offline data with Feast's get_historical_features
        training_df = self.fs.get_historical_features(
            entity_df=entity_df, 
            feature_refs = refs
        ).to_df()

        return training_df

    def _get_features(self, source_df):
        
        cols = list(source_df)
        features = []
        for col in cols:
            f_type = source_df.dtypes[col]
            feature = MLFeature(col, f_type)
            features.append(feature)
        return features

    def _register_dataset(self, feature_keys, source) -> None:
        """
        Internally registers a data source and ingested features as a dataset within the system, giving the dataset a uuid 
        and creating entries in FEAT_DATA_UUID table which link that dataset_id with all of the feature_ids in it.
        Params:
            feature_keys (List[MLFeature]): A list of MLFeature objects that should be retrieved from the offline store. 
            source (str): Name of dataset file or source from which feature_keys come from (i.e. driver_stats.parquet). 
        """
        # create uuid for dataframe. 
        # register dataframe/uuid into db with the features group uuid
        conn = sqlite3.connect('data/metadata.db')
        curr = conn.cursor() 
        # TODO: CHANGE REGISTER_DATASET TO CALL LINEAGE API TO TAG RUN INSTEAD OF THIS
        # TODO: REMOVE. MOVE LINEAGE METADATA UPDATE TO _UPDATE_METADATA()
        for feature in feature_keys:

            # check if feature is already registered into lineage_table (if: continue, else: register it)
            feat_query = f"SELECT feature_uuid FROM GROUP_UUID_DATA WHERE feature='{feature.name}';"
            df = pd.read_sql_query(feat_query,conn)
            feat_uuid = df['feature_uuid'].values[0]
            type_query = f"SELECT feature_type FROM GROUP_UUID_DATA WHERE feature='{feature.name}';"
            df = pd.read_sql_query(type_query,conn)
            feat_type = df['feature_type'].values[0]   
            curr.execute("select feature from FEAT_DATA_UUID where feature=?", (feature.name,))
            feature_exists = curr.fetchall()

            if not feature_exists:
                # check if dataset doesn't already exist in lineage table and register dataset with new dataset source
                data_query = f"SELECT dataset FROM FEAT_DATA_UUID WHERE dataset='{source}';"
                df = pd.read_sql_query(data_query,conn)
                if df.empty:
                    addData = f"INSERT INTO FEAT_DATA_UUID VALUES('{feature.name}','{feat_uuid}', '{source}', '{feat_type}')"
                    curr.execute(addData)
                    continue           
                # dataset is already in lineage table: register dataset with existing dataset source
                existing_dataset = df['dataset'].values[0]
                addData = f"INSERT INTO FEAT_DATA_UUID VALUES('{feature.name}','{feat_uuid}', '{existing_dataset}', '{feat_type}')"
                curr.execute(addData)  
            # feature does exist but not with this dataset 
            data_query = f"SELECT dataset FROM FEAT_DATA_UUID WHERE dataset='{source}';"
            df = pd.read_sql_query(data_query,conn)
            if df.empty:
                addData = f"INSERT INTO FEAT_DATA_UUID VALUES('{feature.name}','{feat_uuid}', '{source}', '{feat_type}')"
                curr.execute(addData)

        conn.commit()
        conn.close()


    def _update_metadata(self, features, source, entity_name) -> None:

        """
        Internally populate metadata.db with features and their group IDs when ingested together.
        Upstream integration for Cataloging API to be able to discover related features.
        Params:
            feature_keys (List[MLFeature]): A list of MLFeature objects ingested into the offline store.
        """
        conn = sqlite3.connect("data/metadata.db")
        curr = conn.cursor()

        f_name = os.path.basename(source)
        view_name = os.path.splitext(f_name)[0]

        # insert the features if not already in metadata.db for same view_name. 
        for feature in features:
            feat_uuid = uuid.uuid4()
            data_type = feature.type
            feature.type = self._convertToValueType(feature.type)
            feat_query = f"SELECT view_name FROM FEATURE_DATA WHERE feature='{feature.name}';"
            curr.execute(feat_query)
            data = curr.fetchall()
            if not data:
                addData = f"""INSERT INTO FEATURE_DATA VALUES('{feature.name}', '{view_name}','{feat_uuid}', '{data_type}', '{entity_name}', '{f_name}')"""
                curr.execute(addData)

        conn.commit()
        conn.close()

    def _create_entity(self, source, entity_name) -> pd.DataFrame:
        """
        Internal helper method to create a pandas entity dataframe of the data source.
        Params:
            source (str or pd.Dataframe): Either a file path to parquet file to ingest batch data into offline store.
            entity_name (str): Str name to represent primary key of entity.
        Returns:
            pandas Dataframe of entire source with event timestamp and entity columns.
        """
        df = pd.read_parquet(source, engine='auto')
        entity_df = pd.DataFrame.from_dict({
            entity_name: [id for id in df[entity_name]],
            "event_timestamp": [timestamp for timestamp in df["datetime"]]
            })
        return entity_df

    def _convertToValueType(self, dtype) -> ValueType:

        """
        Internal helper method to convert pandas data types to Feast's ValueType.
        ValueType is needed for Entity and Feature instantiation. 
        Params:
            dtype (str): Pandas datatype as a string. 
                Only types supported: 'int32', 'int64', 'str', 'bool, 'float32', 'float64', 'category', 'bytes'
        Returns:
            ValueType conversion of Pandas dtype.
        """
        
        if dtype == "int64":
            return ValueType.INT64
        elif dtype == "int32":
            return ValueType.INT32
        elif dtype == "str":
            return ValueType.STRING
        elif dtype == "bytes":
            return ValueType.BYTES
        elif dtype == "bool":
            return ValueType.BOOL
        elif dtype == "float64":
            return ValueType.DOUBLE
        elif dtype == "float32":
            return ValueType.FLOAT
        else:
            raise Exception("Type does not exist. Acceptable Pandas types: 'int32', 'int64', 'str', 'bool, 'float32', 'float64', 'category', 'bytes'" )
# TODO REMOVE
class MLFeature():

    """
        Feature object that represents a feature. 
        Params:
            name (str): Name of feature
            type (str): Pandas datatype of feature
    """
    
    def __init__(self, name, type) -> None:
        self.name = name
        self.type = type
