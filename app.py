from flask import Flask, request
import os
import json

from store.util.util import read_yaml_file, write_yaml_file
from matplotlib.style import context
from store.logger import logging
from store.config.configuration import Configuration
from store.constant import CONFIG_DIR, get_current_time_stamp
from store.pipeline.pipeline import Pipeline
from store.entity.store_predictor import StorePredictor, StoreData
from flask import send_file, abort, render_template


ROOT_DIR = os.getcwd()
LOG_FOLDER_NAME = "logs"
PIPELINE_FOLDER_NAME = "store"
SAVED_MODELS_DIR_NAME = "saved_models"
MODEL_CONFIG_FILE_PATH = os.path.join(ROOT_DIR, CONFIG_DIR, "model.yaml")
LOG_DIR = os.path.join(ROOT_DIR, LOG_FOLDER_NAME)
PIPELINE_DIR = os.path.join(ROOT_DIR, PIPELINE_FOLDER_NAME)
MODEL_DIR = os.path.join(ROOT_DIR, SAVED_MODELS_DIR_NAME)


from store.logger import get_log_dataframe

STORE_DATA_KEY = "store_data"
ITEM_STORE_SALES_VALUE_KEY = "Item_outlet_sales"

app = Flask(__name__)


@app.route('/artifact', defaults={'req_path': 'store'})
@app.route('/artifact/<path:req_path>')
def render_artifact_dir(req_path):
    os.makedirs("store", exist_ok=True)
    # Joining the base and the requested path
    print(f"req_path: {req_path}")
    abs_path = os.path.join(req_path)
    print(abs_path)
    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        if ".html" in abs_path: # If html file show webpage
            with open(abs_path, "r", encoding="utf-8") as file:
                content = ''
                for line in file.readlines():
                    content = f"{content}{line}"
                return content
        return send_file(abs_path) # If artifact then download file

    # Show directory contents
    #files =  {<root>\\store\\artifact': 'artifact'}
    files = {os.path.join(abs_path, file_name): file_name for file_name in os.listdir(abs_path) if
             "artifact" in os.path.join(abs_path, file_name)}

    result = {
        "files": files,
        "parent_folder": os.path.dirname(abs_path),
        "parent_label": abs_path
    }
    return render_template('files.html', result=result) #Passing result dict to files.html


@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        return str(e)


@app.route('/view_experiment_hist', methods=['GET', 'POST'])
def view_experiment_history():
    experiment_df = Pipeline.get_experiments_status()
    context = {
        "experiment": experiment_df.to_html(classes='table table-striped col-12')
    }
    return render_template('experiment_history.html', context=context)


@app.route('/train', methods=['GET', 'POST'])
def train():
    message = ""
    pipeline = Pipeline(config=Configuration(current_time_stamp=get_current_time_stamp()))
    if not Pipeline.experiment.running_status:
        message = "Training started."
        pipeline.start()
    else:
        message = "Training is already in progress."
    context = {
        "experiment": pipeline.get_experiments_status().to_html(classes='table table-striped col-12'),
        "message": message
    }
    return render_template('train.html', context=context)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    context = {
        STORE_DATA_KEY: None,
        ITEM_STORE_SALES_VALUE_KEY: None
    }

    if request.method == 'POST':
        Item_Weight = float(request.form['Item_Weight'])
        Item_Fat_Content = request.form['Item_Fat_Content']
        Item_Visibility = float(request.form['Item_Visibility'])
        Item_Type = request.form['Item_Type']
        Item_MRP = float(request.form['Item_MRP'])
        Outlet_Identifier = request.form['Outlet_Identifier']
        Outlet_Establishment_Year = int(request.form['Outlet_Establishment_Year'])
        Outlet_Size = request.form['Outlet_Size']
        Outlet_Location_Type = request.form['Outlet_Location_Type']
        Outlet_Type = request.form['Outlet_Type']

        store_data = StoreData(Item_Weight = Item_Weight,
                                Item_Fat_Content = Item_Fat_Content,
                                Item_Visibility = Item_Visibility,
                                Item_Type = Item_Type,
                                Item_MRP = Item_MRP,
                                Outlet_Identifier = Outlet_Identifier,
                                Outlet_Establishment_Year = Outlet_Establishment_Year,
                                Outlet_Size = Outlet_Size,
                                Outlet_Location_Type = Outlet_Location_Type,
                                Outlet_Type = Outlet_Type
                                )
        store_df = store_data.get_store_input_data_frame()
        store_predictor = StorePredictor(model_dir=MODEL_DIR)
        item_outlet_sales_value = store_predictor.predict(X=store_df)
        context = {
            STORE_DATA_KEY: store_data.get_store_data_as_dict(),
            ITEM_STORE_SALES_VALUE_KEY: item_outlet_sales_value,
        }
        return render_template('predict.html', context=context)
    return render_template("predict.html", context=context)


@app.route('/saved_models', defaults={'req_path': 'saved_models'})
@app.route('/saved_models/<path:req_path>')
def saved_models_dir(req_path):
    os.makedirs("saved_models", exist_ok=True)
    # Joining the base and the requested path
    print(f"req_path: {req_path}")
    abs_path = os.path.join(req_path)
    print(abs_path)
    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        return send_file(abs_path) #Download file

    # Show directory contents
    files = {os.path.join(abs_path, file): file for file in os.listdir(abs_path)}

    result = {
        "files": files,
        "parent_folder": os.path.dirname(abs_path),
        "parent_label": abs_path
    }
    return render_template('saved_models_files.html', result=result)


@app.route("/update_model_config", methods=['GET', 'POST'])
def update_model_config():
    try:
        if request.method == 'POST':
            model_config = request.form['new_model_config']
            model_config = model_config.replace("'", '"')
            print(model_config)
            model_config = json.loads(model_config)

            write_yaml_file(file_path=MODEL_CONFIG_FILE_PATH, data=model_config)

        model_config = read_yaml_file(file_path=MODEL_CONFIG_FILE_PATH)
        return render_template('update_model.html', result={"model_config": model_config})

    except  Exception as e:
        logging.exception(e)
        return str(e)


@app.route(f'/logs', defaults={'req_path': f'{LOG_FOLDER_NAME}'})
@app.route(f'/{LOG_FOLDER_NAME}/<path:req_path>')
def render_log_dir(req_path):
    os.makedirs(LOG_FOLDER_NAME, exist_ok=True)
    # Joining the base and the requested path
    logging.info(f"req_path: {req_path}")
    abs_path = os.path.join(req_path)
    print(abs_path)
    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        log_df = get_log_dataframe(abs_path)
        context = {"log": log_df.to_html(classes="table-striped", index=False)}
        return render_template('log.html', context=context)

    # Show directory contents
    files = {os.path.join(abs_path, file): file for file in os.listdir(abs_path)}

    result = {
        "files": files,
        "parent_folder": os.path.dirname(abs_path),
        "parent_label": abs_path
    }
    return render_template('log_files.html', result=result)


if __name__ == "__main__":
    app.run() #Default port = 5000
