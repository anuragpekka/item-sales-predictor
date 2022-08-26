from store.entity.config_entity import DataIngestionConfig
import sys,os
from store.exception import StoreException
from store.logger import logging
from store.entity.artifact_entity import DataIngestionArtifact
from store.constant import DATA_VALIDATION_SCHEMA_TARGET_COLUMN_KEY
#import tarfile
from zipfile import ZipFile
import numpy as np
#from six.moves import urllib
import shutil
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

class DataIngestion:
    #data_ingestion_config: A namedtuple created in store\config\configuration.py
    def __init__(self,data_ingestion_config:DataIngestionConfig):
        try:
            logging.info(f"{'='*20}Data Ingestion log started.{'='*20} ")
            self.data_ingestion_config = data_ingestion_config

        except Exception as e:
            raise StoreException(e,sys)
    

    def download_store_data(self,) -> str:
        try:
            #extraction remote url to download dataset from namedtuple
            #download_url = self.data_ingestion_config.dataset_download_url
            input_data_file = os.path.join(
                self.data_ingestion_config.input_data_dir,
                self.data_ingestion_config.zip_file_name
            )

            #folder location to download file
            zip_download_dir = self.data_ingestion_config.zip_download_dir
            
            #Removing dir to get a clean dir
            if os.path.exists(zip_download_dir):
                os.remove(zip_download_dir)

            #Creating dir
            os.makedirs(zip_download_dir,exist_ok=True)

            #store_file_name = os.path.basename(download_url)
            store_file_name = self.data_ingestion_config.zip_file_name
            zip_file_path = os.path.join(zip_download_dir, store_file_name)
            #zip_file_path = zip_download_dir

            logging.info(f"Downloading file from :[{input_data_file}] into :[{zip_file_path}]")
            #urllib.request.urlretrieve(download_url, zip_file_path)
            shutil.copy(input_data_file, zip_file_path)
            logging.info(f"File :[{zip_file_path}] has been downloaded successfully.")
            return zip_file_path

        except Exception as e:
            raise StoreException(e,sys) from e

    def extract_zip_file(self,zip_file_path:str):
        try:
            raw_data_dir = self.data_ingestion_config.raw_data_dir

            if os.path.exists(raw_data_dir):
                os.remove(raw_data_dir)

            os.makedirs(raw_data_dir,exist_ok=True)

            logging.info(f"Extracting zip file: [{zip_file_path}] into dir: [{raw_data_dir}]")
            #with tarfile.open(zip_file_path) as store_zip_file_obj:
            with ZipFile(zip_file_path, 'r') as store_zip_file_obj:
                store_zip_file_obj.extractall(path=raw_data_dir)
            logging.info(f"Extraction completed")
            extracted_files = os.listdir(raw_data_dir)
            logging.info(f"Extracted files {extracted_files}")

        except Exception as e:
            raise StoreException(e,sys) from e

    def copy_train_test_data(self):
        try:
            raw_data_dir = self.data_ingestion_config.raw_data_dir
            train_file_name = os.path.basename(self.data_ingestion_config.ingested_train_dir) + ".csv"
            test_file_name = os.path.basename(self.data_ingestion_config.ingested_test_dir) + ".csv"
            src_train_file = os.path.join(raw_data_dir, train_file_name)
            src_test_file = os.path.join(raw_data_dir, test_file_name)

            train_file_path = os.path.join(self.data_ingestion_config.ingested_train_dir,
                                            train_file_name)
            test_file_path = os.path.join(self.data_ingestion_config.ingested_test_dir,
                                        test_file_name)

            os.makedirs(self.data_ingestion_config.ingested_train_dir,exist_ok=True)
            logging.info(f"Copying training datset to file: [{train_file_path}]")
            shutil.copy(src_train_file, train_file_path)

            os.makedirs(self.data_ingestion_config.ingested_test_dir, exist_ok= True)
            logging.info(f"Copying test dataset to file: [{test_file_path}]")
            shutil.copy(src_test_file, test_file_path)

            data_ingestion_artifact = DataIngestionArtifact(train_file_path=train_file_path,
                                test_file_path=test_file_path,
                                is_ingested=True,
                                message=f"Data ingestion completed successfully."
                                )
            logging.info(f"Data Ingestion artifact:[{data_ingestion_artifact}]")
            return data_ingestion_artifact
        except Exception as e:
            raise StoreException(e,sys) from e
  
    def split_data_as_train_test(self) -> DataIngestionArtifact:
        try:
            raw_data_dir = self.data_ingestion_config.raw_data_dir
            file_name = os.listdir(raw_data_dir)[0]
            store_file_path = os.path.join(raw_data_dir,file_name)

            schema_info = self.data_ingestion_config.schema_info
            target_column = schema_info[DATA_VALIDATION_SCHEMA_TARGET_COLUMN_KEY]

            logging.info(f"Reading csv file: [{store_file_path}]")
            store_data_frame = pd.read_csv(store_file_path)

            #Seperating to create bins/catagories by factor 1300            
            # (Target column boxplot range)/5 =~ 1300 
            catagory_column = target_column+"_catagoty"
            store_data_frame[catagory_column] = pd.cut(
                store_data_frame[target_column],
                #bins=[0.0, 1300.0, 2600.0, 3900.0, 6400.0, np.inf],
                #labels=[1,2,3,4,5]
                #bins=[0, 650, 1300, 1950, 2600, 3250, 3900, 4550, 5200, 5850, np.inf],
                #labels=[1,2,3,4,5,6,7,8,9,10]
                bins=[0, 325, 650, 975, 1300, 1625, 1950, 2275, 2600, 2925, 3250, 3575, 3900, 4225, 4550, 4875, 5200, 5525, 5850, 6175, np.inf],
                labels=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
                #bins=[0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500,np.inf],
                #labels=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
            )
            

            logging.info(f"Splitting data into train and test")
            strat_train_set = None
            strat_test_set = None

            #Using the StratifiedShuffleSplit to split data
            split = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)

            #X = store_data_frame, y = store_data_frame[catagory_column]
            #train_index ,test_index are Array of index
            for train_index,test_index in split.split(store_data_frame, store_data_frame[catagory_column]):
                strat_train_set = store_data_frame.loc[train_index].drop([catagory_column],axis=1) #Making set with train_index
                strat_test_set = store_data_frame.loc[test_index].drop([catagory_column],axis=1) #Making set with test_index

            train_file_path = os.path.join(self.data_ingestion_config.ingested_train_dir,
                                            file_name)

            test_file_path = os.path.join(self.data_ingestion_config.ingested_test_dir,
                                        file_name)
            
            if strat_train_set is not None:
                os.makedirs(self.data_ingestion_config.ingested_train_dir,exist_ok=True)
                logging.info(f"Exporting training datset to file: [{train_file_path}]")
                strat_train_set.to_csv(train_file_path,index=False)

            if strat_test_set is not None:
                os.makedirs(self.data_ingestion_config.ingested_test_dir, exist_ok= True)
                logging.info(f"Exporting test dataset to file: [{test_file_path}]")
                strat_test_set.to_csv(test_file_path,index=False)
            

            data_ingestion_artifact = DataIngestionArtifact(train_file_path=train_file_path,
                                test_file_path=test_file_path,
                                is_ingested=True,
                                message=f"Data ingestion completed successfully."
                                )
            logging.info(f"Data Ingestion artifact:[{data_ingestion_artifact}]")
            return data_ingestion_artifact

        except Exception as e:
            raise StoreException(e,sys) from e

    def initiate_data_ingestion(self)-> DataIngestionArtifact:
        try:
            zip_file_path =  self.download_store_data()
            self.extract_zip_file(zip_file_path=zip_file_path)
            return self.split_data_as_train_test()
        except Exception as e:
            raise StoreException(e,sys) from e
    


    def __del__(self):
        logging.info(f"{'='*20}Data Ingestion log completed.{'='*20} \n\n")
