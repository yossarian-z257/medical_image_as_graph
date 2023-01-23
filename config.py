MAX_WORKERS = 30
FOLDER_PATH = 'chest_xray'
GRAPH_PATH = 'chest_xray_graphs'
SAVED_MODEL_PATH = 'saved_models'
SAVED_DATA_LOADER = 'saved_data_loader'

feature_size = {
"resnet18" : 512,
"densenet121" : 1024,
"efficientnet-b0" : 1280
}