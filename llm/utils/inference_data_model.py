class InferenceDataModel(object):
    model_name = str()
    repo_version = str()
    input_path = str()
    gpus = int()
    gen_folder = str()
    mar_filepath = str()
    ts_log_file = str()
    ts_log_config = str()
    ts_config_file = str()
    ts_model_store = str()
    dir_path = str()


def set_data_model(gpus, gen_folder, data="", model_name="", mar_filepath="", repo_version=""):
    data_model = InferenceDataModel()
    data_model.model_name = model_name
    data_model.input_path = data
    data_model.gpus = gpus
    data_model.gen_folder = gen_folder
    data_model.mar_filepath = mar_filepath
    data_model.repo_version = repo_version

    return data_model