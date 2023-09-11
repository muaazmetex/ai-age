from flask import Flask, request, jsonify, render_template, send_from_directory

import dlib
from argparse import Namespace
import os
import sys
import pprint
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from datasets.augmentations import AgeTransformer
from utils.common import tensor2im
from models.psp import pSp

# import utility functions
from myUtils import exp_data_args, load_model, generate_random_string
from myUtils import run_alignment, run_inference, age_transformer, run_on_batch
from myUtils import get_inference, save_results, get_image_paths
from ipp import get_external_ip

app = Flask(__name__)


IMAGES_FOLDER = "images/"
UPLOAD_FOLDER = "upload_folder/"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
model_path = "pretrained_models/sam_ffhq_aging.pt"

# Load pre-trained model
net = load_model(model_path)


@app.route('/age', methods=["POST", "GET"])
def inference():
    try:
        key = request.form['key']
        if key == "alphabravocharlie1998":
            try:
                if request.method == "POST":

                    file_name_random = generate_random_string()
                    filename = file_name_random
                    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename + ".jpg")


                    try:
                        image_data = request.files['image']
                        if image_data:
                            image_bytes = image_data.read()
                        # Save the image bytes to a file
                            with open(filepath, "wb") as f:
                                f.write(image_bytes)
                            print("Image saved.")
                        else:
                            return "No image data received.", 400

                    except Exception as e:
                        return str(e), 500


                    target_age_limit = 100
                    image_path = filepath

                    
                    myIP = get_external_ip()
                    #print("My Public IP:", myIP)

                    base_url = f"http://{myIP}:5000/" 

                    EXPERIMENT_TYPE = 'ffhq_aging'

                    EXPERIMENT_DATA_ARGS = exp_data_args(image_path)
                    EXPERIMENT_ARGS = EXPERIMENT_DATA_ARGS[EXPERIMENT_TYPE]

                    aligned_image = run_alignment(image_path)

                    img_transforms, input_image = run_inference(aligned_image, EXPERIMENT_ARGS)

                    age_transformers = age_transformer()

                    age_results_dict = get_inference(target_age_limit, aligned_image, input_image, age_transformers, net)

                    saved_image_paths = save_results(age_results_dict, filename)

                    img_json = get_image_paths(base_url, saved_image_paths)

                    print(img_json)
                return jsonify(img_json)

            except Exception as e:
                print("Invalid Method.")
                return jsonify({"invalid_method": "Method not available"})   
        else:
            print("Invalid Key")
            return jsonify({"key_error": "Invalid Key"})
    except Exception as e:
        print("Key Error")
        return jsonify({"key_error": f"\nError:{e}"})        




@app.route('/images/<path:filename>', methods=['GET'])
def serve_image(filename):
    return send_from_directory(IMAGES_FOLDER, filename)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)



 



