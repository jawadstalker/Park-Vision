from roboflow import Roboflow

rf = Roboflow(api_key='Api key')
project = rf.workspace('3883zn-gmail-com').project('roadside-parking')
dataset = project.version(12).download('yolov8', location='roadside_parking_raw')