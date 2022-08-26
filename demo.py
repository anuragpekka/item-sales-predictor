from store.constant import *
from store.config.configuration import Configuration
from store.pipeline.pipeline import Pipeline
#from store.component.data_validation import DataValidation
#from store.component.data_ingestion import DataIngestion
#from store.entity.store_predictor import StorePredictor, StoreData
from store.logger import logging
import requests
#from zipfile import ZipFile
#from io import BytesIO
import shutil

def main():
    try:
        
        #print('Downloading started')
        #url = 'https://www.kaggle.com/datasets/brijbhushannanda1979/bigmart-sales-data/download?datasetVersionNumber=1'

        #config_path = os.path.join("config","config.yaml")
        #configs = Configuration()
        #ingest = DataIngestion(data_ingestion_config=configs.get_data_ingestion_config())
        '''zip_file_path = ingest.download_store_data()
        ingest.extract_zip_file(zip_file_path=zip_file_path)
        ingest.copy_train_test_data()'''
        #ingest_art = ingest.initiate_data_ingestion()
        #valida = DataValidation(configs.get_data_validation_config(), ingest_art)
        #valida.validate_dataset_schema()
        #valida.initiate_data_validation()
        #pipeline = Pipeline(Configuration(config_file_path=config_path))
        #pipeline.start()
        #logging.info("main() function completed.")

        
        '''
        validation_config = configs.get_data_validation_config()
        print(f"validation_config: {validation_config}")
        '''
        #transformation_config = configs.get_data_transformation_config()
        #print(f"transformation_config: {transformation_config}")
        
        #training_config = configs.get_model_trainer_config()
        #print(f"training_config: {training_config}")

        pipe = Pipeline(config=Configuration(current_time_stamp=get_current_time_stamp()))
        pipe.run_pipeline()

        
        #Testing latest model
        
        '''SAVED_MODELS_DIR_NAME = "saved_models"
        MODEL_DIR = os.path.join(ROOT_DIR, SAVED_MODELS_DIR_NAME)
        store_data = StoreData(Item_Weight = 9.395,
                                Item_Fat_Content = "Low Fat",
                                Item_Visibility = 0.103731617,
                                Item_Type = "Snack Foods",
                                Item_MRP = 236.9932,
                                Outlet_Identifier = "OUT035",
                                Outlet_Establishment_Year = 2004,
                                Outlet_Size = "Small",
                                Outlet_Location_Type = "Tier 2",
                                Outlet_Type = "Supermarket Type1"
                                )
        store_df = store_data.get_store_input_data_frame()
        store_predictor = StorePredictor(model_dir=MODEL_DIR)
        item_outlet_sales_value = store_predictor.predict(X=store_df)
        print(f"item_outlet_sales_value={item_outlet_sales_value}") #4242.4776
'''
    except Exception as e:
        logging.error(f"{e}")
        print(e)

if __name__ == "__main__":
    main()
