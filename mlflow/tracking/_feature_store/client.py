from google.protobuf.duration_pb2 import Duration
from feast.feature_store import FeatureStore
from feast import Entity, FeatureView, Feature, ValueType, FileSource
import pandas as pd
import uuid
import sqlite3

class FeatureStoreClient(object):
    """
    Client of an MLflow Feature Store that ingests and retrieves batch data.
    """
    def __init__(self):
        self.fs = FeatureStore(
            repo_path="."
        )

    def ingest(self, source, feature_keys, entity_name, entity_type):

        """
        Batch load feature data to publish into offline store.
        Params:
            source (str or pd.Dataframe): Either a file path to parquet file to ingest batch data into offline store.
            feature_keys (List[MLFeature]): A list of MLFeature objects that should be ingested into the offline store.
        Returns:
            Dataframe of all data ingested with columns of entity_name and datetime.
        Example usage:
            batch_context.ingest(source=“data/drivers.parquet”, feature_keys=[{"name":"avg_cost", "type": ValueType.INT64}]) 

        """
        # creating Feast objects and infrastructure
        file_stats = FileSource(
            path=source,
            event_timestamp_column="datetime",
            created_timestamp_column="created",
        )
        entity_type = self._convertToValueType(entity_type)
        entity = Entity(name=entity_name, value_type=entity_type, description="")

        entity_df = self._create_entity(source, entity_name)
        # update metadata with group uuids for Cataloging and Lineage API
        self._update_metadata(feature_keys)
        self._register_dataset(feature_keys, source)

        # our parquet files contain sample data that includes a driver_id column, timestamps and
        # three feature column. 
        feature_view = FeatureView(
            name="feature",
            entities=[entity.name],
            ttl=Duration(seconds=86400 * 1),
            features=[Feature(name=feature.name, dtype=feature.type) for feature in feature_keys],
            online=False, 
            input=file_stats,
            tags={},
        )

        # creates and updates registry.db metadata
        self.fs.apply([entity, feature_view])
        return entity_df

    def retrieve(self, feature_keys, entity_df) -> pd.DataFrame:

        """
        Get features that have been registered already into the offline store.
        Params:
            feature_keys (List[MLFeature]): A list of MLFeature objects that should be retrieved from the offline store. 
        Returns:
            Some object with the features that can be used for batch inferencing or training.
        Example usage:
        quality = batch_context.MLFeature("quality", "int64")
        alcohol = batch_context.MLFeature("alcohol", "float32")
        feature_keys = [quality, alcohol]
            feature_df = batch_context.retrieve(feature_keys, entity_df)
        """
        
        refs = []
        for feature_key in feature_keys:
            refs.append("feature:{}".format(feature_key.name))

        # retrieving offline data with Feast's get_historical_features
        training_df = self.fs.get_historical_features(
            entity_df=entity_df, 
            feature_refs = refs
        ).to_df()

        return training_df

    def _register_dataset(self, feature_keys, dataset) -> None:
        """
        Internally registers a data source and ingested features as a dataset within the system, giving the dataset a uuid 
        and creating entries in FEAT_DATA_UUID table which link that dataset_id with all of the feature_ids in it.
        Params:
            feature_keys (List[MLFeature]): A list of MLFeature objects that should be retrieved from the offline store. 
            dataset (str): Name of dataset file or source from which feature_keys come from (i.e. driver_stats.parquet). 
        """
        # create uuid for dataframe. 
        # register dataframe/uuid into db with the features group uuid
        conn = sqlite3.connect('data/metadata.db')
        curr = conn.cursor() 

        data_uuid = uuid.uuid4()
        for feature in feature_keys:

            # check if feature is already registered into lineage_table (if: continue, else: register it)
            feat_query = f"SELECT feature_uuid FROM GROUP_UUID_DATA WHERE feature='{feature.name}';"
            df = pd.read_sql_query(feat_query,conn)
            feat_uuid = df['feature_uuid'].values[0]
            curr.execute("select feature from FEAT_DATA_UUID where feature=?", (feature.name,))
            feature_exists = curr.fetchall()

            if not feature_exists:
                # check if dataset doesn't already exist in lineage table and register dataset with new uuid
                data_query = f"SELECT data_uuid FROM FEAT_DATA_UUID WHERE dataset='{dataset}';"
                df = pd.read_sql_query(data_query,conn)
                if df.empty:
                    addData = f"""INSERT INTO FEAT_DATA_UUID VALUES('{feature.name}','{feat_uuid}', '{dataset}', '{data_uuid}')"""
                    curr.execute(addData)
                    continue           
                # dataset is already in lineage table: register dataset with existing uuid
                existing_data_uuid = df['data_uuid'].values[0]
                addData = f"""INSERT INTO FEAT_DATA_UUID VALUES('{feature.name}','{feat_uuid}', '{dataset}', '{existing_data_uuid}')"""
                curr.execute(addData)   

        conn.commit()
        conn.close()


    def _update_metadata(self, feature_keys) -> None:

        """
        Internally populate metadata.db with features and their group IDs when ingested together.
        Upstream integration for Cataloging API to be able to discover related features.
        Params:
            feature_keys (List[MLFeature]): A list of MLFeature objects ingested into the offline store.
        """
        conn = sqlite3.connect("data/metadata.db")
        curr = conn.cursor()

        group_uuid = uuid.uuid4()

        # first check if any of the feature_keys have already been registered to metadata to keep the group_uuid
        for feature in feature_keys:
            feat_query = f"SELECT group_uuid FROM GROUP_UUID_DATA WHERE feature='{feature.name}';"
            df = pd.read_sql_query(feat_query,conn)
            if not df.empty:
                feat_group_uuid = df['group_uuid'].values[0]
                group_uuid = feat_group_uuid
        
        # insert the features if not already in metadata.db. 
        for feature in feature_keys:
            feat_uuid = uuid.uuid4()
            data_type = feature.type
            feature.type = self._convertToValueType(feature.type)
            feat_query = f"SELECT group_uuid FROM GROUP_UUID_DATA WHERE feature='{feature.name}';"
            curr.execute(feat_query)
            data = curr.fetchall()
            if not data:
                addData = f"""INSERT INTO GROUP_UUID_DATA VALUES('{feature.name}', '{group_uuid}','{feat_uuid}', '{data_type}')"""
                curr.execute(addData)

        conn.commit()
        conn.close()

    def _create_entity(self, source, entity_name):
        
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
        elif dtype == "str" or dtype == "category":
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