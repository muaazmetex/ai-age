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

                    try:
                        aligned_image = run_alignment(image_path)
                    except Exception as e:
                        print("Error:", e)
                        return jsonify({"error": f"Alignment error: {e}"})

                    try:
                        img_transforms, input_image = run_inference(aligned_image, EXPERIMENT_ARGS)
                    except Exception as e:
                        print("Error:", e)
                        return jsonify({"error": f"Run Inference error: {e}"})

                    try:                        
                        age_transformers = age_transformer()
                    except Exception as e:
                        print("Error:", e)
                        return jsonify({"error": f"Transformers error: {e}"})

                    try:                        
                        age_results_dict = get_inference(target_age_limit, aligned_image, input_image, age_transformers, net)
                    except Exception as e:
                        print("Error:", e)
                        return jsonify({"error": f"Get Inference error: {e}"})

                    try:                        
                        saved_image_paths = save_results(age_results_dict, filename)
                    except Exception as e:
                        print("Error:", e)
                        return jsonify({"error": f"Save results error: {e}"})

                    try:                        
                        img_json = get_image_paths(base_url, saved_image_paths)
                    except Exception as e:
                        print("Error:", e)
                        return jsonify({"error": f"Image paths error: {e}"})

                    print(img_json)
                return jsonify(img_json)

            except Exception as e:
                print("Invalid Method.", e)
                return jsonify({"error": "Method not available"})   

        else:
            print("Invalid Key")

    except Exception as e:
        print("Key Error")
        return jsonify({"error": f"Invalid Key:{e}"})        

@app.route('/images/<path:filename>', methods=['GET'])
def serve_image(filename):
    return send_from_directory(IMAGES_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)



