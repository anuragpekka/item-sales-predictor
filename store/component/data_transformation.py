from cgi import test
from sklearn import preprocessing
from store.exception import StoreException
from store.logger import logging
from store.entity.config_entity import DataTransformationConfig 
from store.entity.artifact_entity import DataIngestionArtifact,\
DataValidationArtifact,DataTransformationArtifact
import sys,os
import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import pandas as pd
from store.constant import *
from store.util.util import read_yaml_file,save_object,save_numpy_array_data,load_data


class ValueReplacer(BaseEstimator, TransformerMixin):

    def __init__(self, columns=None):
        try:
            self.replace_values_in_column_dict = REPLACE_VALUES_IN_COLUMN_DICT
            self.columns = columns
        except Exception as e:
            raise StoreException(e, sys) from e

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        try:            
            for column in self.replace_values_in_column_dict.keys():
                index = self.columns.index(column)
                replace_values_dict = self.replace_values_in_column_dict[column]
                for to_replace, replace_with in replace_values_dict.items():
                    X[index] = np.where(X[index] == to_replace, replace_with, X[index])
                        
            return X
        except Exception as e:
            raise StoreException(e, sys) from e

class DataTransformation:

    def __init__(self, data_transformation_config: DataTransformationConfig,
                 data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_artifact: DataValidationArtifact
                 ):
        try:
            logging.info(f"{'=' * 20}Data Transformation log started.{'=' * 20} ")
            self.data_transformation_config= data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact

        except Exception as e:
            raise StoreException(e,sys) from e

    
    #Creating numerical and catagrical pipeline for data transformation
    def get_data_transformer_object(self)->ColumnTransformer:
        try:
            #schema_file_path = self.data_validation_artifact.schema_file_path
            #dataset_schema = read_yaml_file(file_path=schema_file_path)
            dataset_schema = self.data_validation_artifact.schema_info

            #Getting numerical and catagorical columns from schema.yaml
            numerical_columns = dataset_schema[NUMERICAL_COLUMN_KEY]
            categorical_columns = dataset_schema[CATEGORICAL_COLUMN_KEY]

            #Remove unuseful columns
            for col in dataset_schema[DATASET_DROP_COLUMNS_KEY]:
                if col in numerical_columns:
                    numerical_columns.remove(col)
                if col in categorical_columns:
                    categorical_columns.remove(col)
                     

            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy="median")),
                ('scaler', StandardScaler())
            ]
            )

            cat_pipeline = Pipeline(steps=[
                 ('impute', SimpleImputer(strategy="most_frequent")),                 
                 ('value_replacer', ValueReplacer(columns=categorical_columns)),
                 ('one_hot_encoder', OneHotEncoder()),
                 ('scaler', StandardScaler(with_mean=False))
            ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")


            preprocessing = ColumnTransformer(transformers = [
                ('num_pipeline', num_pipeline, numerical_columns),
                ('cat_pipeline', cat_pipeline, categorical_columns),
            ], sparse_threshold = 0.0)
            
            return preprocessing

        except Exception as e:
            raise StoreException(e,sys) from e   


    def initiate_data_transformation(self)->DataTransformationArtifact:
        try:
            logging.info(f"Obtaining preprocessing object.")
            preprocessing_obj = self.get_data_transformer_object()


            logging.info(f"Obtaining training and test file path.")
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path
            

            #schema_file_path = self.data_validation_artifact.schema_file_path
            
            logging.info(f"Loading training and test data as pandas dataframe.")
            #train_df = load_data(file_path=train_file_path, schema_file_path=schema_file_path)
            
            #test_df = load_data(file_path=test_file_path, schema_file_path=schema_file_path)

            schema_info = self.data_validation_artifact.schema_info
            train_df = load_data(file_path=train_file_path, schema_info=schema_info)
            
            test_df = load_data(file_path=test_file_path, schema_info=schema_info)

            target_column_name = schema_info[TARGET_COLUMN_KEY]


            logging.info(f"Splitting input and target feature from training and testing dataframe.")
            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]
            

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)


            train_arr = np.c_[ input_feature_train_arr, np.array(target_feature_train_df)]

            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            transformed_train_dir = self.data_transformation_config.transformed_train_dir
            transformed_test_dir = self.data_transformation_config.transformed_test_dir

            train_file_name = os.path.basename(train_file_path).replace(".csv",".npz")
            test_file_name = os.path.basename(test_file_path).replace(".csv",".npz")

            transformed_train_file_path = os.path.join(transformed_train_dir, train_file_name)
            transformed_test_file_path = os.path.join(transformed_test_dir, test_file_name)

            logging.info(f"Saving transformed training and testing array.")
            
            save_numpy_array_data(file_path=transformed_train_file_path,array=train_arr)
            save_numpy_array_data(file_path=transformed_test_file_path,array=test_arr)

            preprocessing_obj_file_path = self.data_transformation_config.preprocessed_object_file_path

            logging.info(f"Saving preprocessing object.")
            save_object(file_path=preprocessing_obj_file_path,obj=preprocessing_obj)

            data_transformation_artifact = DataTransformationArtifact(is_transformed=True,
            message="Data transformation successfull.",
            transformed_train_file_path=transformed_train_file_path,
            transformed_test_file_path=transformed_test_file_path,
            preprocessed_object_file_path=preprocessing_obj_file_path

            )
            logging.info(f"Data transformationa artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise StoreException(e,sys) from e

    def __del__(self):
        logging.info(f"{'='*20}Data Transformation log completed.{'='*20} \n\n")
        