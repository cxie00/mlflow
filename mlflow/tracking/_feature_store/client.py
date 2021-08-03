from google.protobuf.duration_pb2 import Duration
from feast.feature_store import FeatureStore
from feast import Entity, FeatureView, Feature, ValueType, FileSource
from mlflow.entities import MLFeature
import pandas as pd
import uuid
import sqlite3
import os 

#lineage imports
from mlflow.entities import run
import numpy as np
import sqlite3
import sys
import os
from mlflow.types.schema import DataType, ColSpec, FeatureColSpec
from mlflow.exceptions import MlflowException
from typing import Dict, Any, List, Union, Optional
from mlflow.models.signature import  ModelSignature, Schema
from mlflow.types.utils import _infer_schema



class FeatureStoreClient(object):
    #Batch Context API
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
        name = os.path.splitext(f_name)[0]

        features = self._get_features(source_df.drop([entity_name, "created", "datetime"], axis=1)) 
        
        # update metadata with group uuids for Cataloging and Lineage API
        self._update_metadata(features, name, entity_name)
        # self._register_dataset(features, source)

        feature_view = FeatureView(
            name=name,
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
            view_query = f"SELECT view_name FROM FEATURE_DATA WHERE feature='{feature}' and entity_name='{entity_name}';"
            df = pd.read_sql_query(view_query,conn)
            view = df['view_name'].values[0]
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
        """
        Internal helper function that reads through a pandas DataFrame to infer features and their types
        into a list of MLFeature objects (defined in mlflow/mlflow/entities/mlfeature.py).
        Params: 
            source_df (pandas DataFrame): DataFrame of dataset ingested. Contains event timestamp, entity id, and feature
                columns.
        Returns:
            List of MLFeature objects to be ingested.
        """
        cols = list(source_df)
        features = []
        for col in cols:
            f_type = source_df.dtypes[col]
            feature = MLFeature(col, f_type)
            features.append(feature)
        return features

    def _register_dataset(self) -> None:
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

        # TODO: USE LINEAGE API TO TAG RUN ID 
        
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
    #Cataloging API
    def search_features(self, database, filter_string):

        conn = sqlite3.connect(database)
        curr = conn.cursor()
        query_string = "SELECT * FROM FEATURE_DATA WHERE " + filter_string

        if curr.execute(query_string):
            rows = curr.fetchall()
            return rows

    def search_by_entity(self, database, filter_string):
    
        conn = sqlite3.connect(database)
        curr = conn.cursor()

        feature_group_uuid = "SELECT entity_name FROM FEATURE_DATA WHERE " + filter_string
    
        curr.execute(feature_group_uuid)
        get_uuid = curr.fetchall()
        my_list = get_uuid
        
        my_tuple = get_uuid
        my_tuple = my_list[0]
        
        table_uuid = "SELECT * FROM  FEATURE_DATA WHERE entity_name = '" + my_tuple[0]+ "'"
        curr.execute(table_uuid)
        rows = curr.fetchall()
        return rows

    def search_related_features(self, database, filter_string):
    
        conn = sqlite3.connect(database)
        curr = conn.cursor()

        feature_group_uuid = "SELECT view_name FROM FEATURE_DATA WHERE " + filter_string
        
        curr.execute(feature_group_uuid)
        get_uuid = curr.fetchall()
        my_list = get_uuid
        
        my_tuple = get_uuid
        my_tuple = my_list[0]
        
        table_uuid = "SELECT * FROM  FEATURE_DATA WHERE view_name = '" + my_tuple[0]+ "'"
        curr.execute(table_uuid)
        rows = curr.fetchall()
        return rows
    #Lineage API
    

    def get_data_type(self, type) -> DataType:
        if type == 'int64' or type == 'int32':
            return DataType.integer
        if type == 'bool':
            return DataType.boolean
        if type == 'float64' or type == 'float32':
            return DataType.float
        if type == 'object':
            return DataType.string
        if type == 'datetime64':
            return DataType.datetime
        #note: pandas does not have 'long', 'double', 'binary' types
    
    def parse_feature_metadata(self) -> List[FeatureColSpec]:
        """This function is called after features have been ingested and 
        the corresponding metadata.db has been created. Here, we parse
        through metadata.db and create a list of feature objects"""

        feature_colspec_list = list()

        directory_path = os.getcwd()
        conn = sqlite3.connect('data\metadata.db')
        curr = conn.cursor()
        fetchData = "SELECT * from FEATURE_DATA"
        curr.execute(fetchData)
        row = curr.fetchone()
        while row is not None:
            #create a new FeatureColSpec for each row
            row_str = ','.join(row)
            type = self.get_data_type(row[3])
            datatype_str = DataType.__repr__(type)
            name = row[0]
            id = row[2]
            feature = FeatureColSpec(type, name, id)
            feature_colspec_list.append(feature)
            row = curr.fetchone()
        return feature_colspec_list
    
        

    def infer_signature_override(self, model_input: Any, model_output: "MlflowInferableDataset" = None
    ) -> ModelSignature:
        print("infer signature override...")
        inputs = Schema(self.parse_feature_metadata())
        outputs = _infer_schema(model_output) if model_output is not None else None
        return ModelSignature(inputs, outputs)
    
        
    def map_run_to_features(self, feature_list: List[FeatureColSpec], feature_runs_dict) ->None:
        '''updates the feature_runs_dict. tags the active ml run id to each 
        corresponding feature in the inputted 
        list'''
        run_id = self.active_run().info.run_id
        for f in feature_list:
            if f.name in feature_list:
                value_set = set()
                x = feature_runs_dict.pop(f.name)
                value_set.update(x)
                value_set.add(run_id)
                feature_runs_dict[f.name] = value_set
                del value_set
                del x
            else:
                value_set = set()
                value_set.add(run_id)
                feature_runs_dict[f.name] = value_set
                del value_set