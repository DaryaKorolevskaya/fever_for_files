import argparse
from artifacts.predict_tag import predict_tag
import json

parser = argparse.ArgumentParser(description="Run pipeline for model NL2ML (tag prediction)")
parser.add_argument("-p", "--params", type=str, help="path to request.json file with parameters")

args = parser.parse_args()
with open(args.params) as jf:
    params = json.load(jf)


path_to_model_dir = params["path_to_model_dir"]
data_input_path = params["data_input_path"]
data_output_path = params["data_output_path"]

predict_tag(data_input_path, data_output_path)
