from flask import Flask, request
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


app = Flask(__name__)


# image_path = "tests/images/2468.jpg"

def exp_data_args(image_path):

    # Define Inference Parameters
    EXPERIMENT_DATA_ARGS = {
        "ffhq_aging": {
            "model_path": "pretrained_models/sam_ffhq_aging.pt",
            "image_path": image_path,
            # "image_path": "tests/images/2468.jpg",
            "transform": transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        }
    }

    return EXPERIMENT_DATA_ARGS


# EXPERIMENT_DATA_ARGS = exp_data_args(image_path)

# EXPERIMENT_ARGS = EXPERIMENT_DATA_ARGS[EXPERIMENT_TYPE]

def load_model(EXPERIMENT_ARGS):
# Load Pretrained Model


    # # Check if a GPU is available
    # use_gpu = torch.cuda.is_available()

    # # Define the device to use (CPU or GPU)
    # device = torch.device('cuda' if use_gpu else 'cpu')

    model_path = EXPERIMENT_ARGS['model_path']
    ckpt = torch.load(model_path, map_location='cpu')

    opts = ckpt['opts']
    # pprint.pprint(opts)


    # update the training options
    opts['checkpoint_path'] = model_path

    # Load model
    opts = Namespace(**opts)
    net = pSp(opts)
    net.eval()
    net.cuda()
    print('Model successfully loaded!')


# Visualize image
def Visualize_image(image_path, EXPERIMENT_DATA_ARGS):
    image_path = EXPERIMENT_DATA_ARGS[EXPERIMENT_TYPE]["image_path"]
    original_image = Image.open(image_path).convert("RGB")
    original_image.resize((256, 256))
    # plt.imshow(original_image)


# Align Image
def run_alignment(image_path):

    from scripts.align_all_parallel import align_face
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    aligned_image = align_face(filepath=image_path, predictor=predictor)
    print("Aligned image has shape: {}".format(aligned_image.size))
    return aligned_image

# aligned_image = run_alignment(image_path)
# aligned_image.resize((256, 256))


# Run Inference
def run_inference(aligned_image, EXPERIMENT_ARGS):
    img_transforms = EXPERIMENT_ARGS['transform']
    input_image = img_transforms(aligned_image)
    return img_transforms, input_image

def age_transformer():
    # we'll run the image on multiple target ages
    target_ages = [100]
    # target_ages = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    age_transformers = [AgeTransformer(target_age=age) for age in target_ages]

    return age_transformers

def run_on_batch(inputs, net):
    result_batch = net(inputs.to("cuda").float(), randomize_noise=False, resize=False)
    return result_batch


def get_inference(target_age_limit, aligned_image, input_image, age_transformers):

    try:
        # for each age transformed age, we'll concatenate the results to display them side-by-side
        results = np.array(aligned_image.resize((1024, 1024)))
        for age_transformer in age_transformers:
            print(f"Running on target age: {age_transformer.target_age}")
            with torch.no_grad():
                input_image_age = [age_transformer(input_image.cpu()).to('cuda')]
                input_image_age = torch.stack(input_image_age)
                result_tensor = run_on_batch(input_image_age, net)[0]
                result_image = tensor2im(result_tensor)
                results = np.concatenate([results, result_image], axis=1)
                if age_transformer.target_age == target_age_limit:
                    return result_image
                else:
                    continue

    except Exception as e:
        print(f"The error occured during inference.\n Error: {e}")


def show_image(image):
    plt.imshow(image)


def concatenated_image(results):

    results = Image.fromarray(results)
    return results   # this is a very large image (11*1024 x 1024) so it may take some time to display!   


def save_results(results):
    try:
        # save image at full resolution
        results.save(f"saved/images/age_transformed_image_{results}.jpg")
        print("Results saved successfully!")

    except Exception as e:
        print(f"Error in saving image: {e}")


@app.route('/', methods=["POST"])
def inference():
 
    if requests.method=="POST":


        EXPERIMENT_TYPE = 'ffhq_aging'

        # MODEL_PATHS = {"ffhq_aging": {"id": "1XyumF6_fdAxFmxpFcmPf-q84LU_22EMC", "name": "sam_ffhq_aging.pt"}}
        # path = MODEL_PATHS[EXPERIMENT_TYPE]

        target_age_limit = 100
        image_path = "tests/images/2468.jpg"

        EXPERIMENT_DATA_ARGS = exp_data_args(image_path)
        EXPERIMENT_ARGS = EXPERIMENT_DATA_ARGS[EXPERIMENT_TYPE]

        load_model(EXPERIMENT_ARGS)

 
        aligned_image = run_alignment(image_path)

        img_transforms, input_image = run_inference(aligned_image, EXPERIMENT_ARGS)\

        age_transformers = age_transformer()

        aged_image = get_inference(target_age_limit, aligned_image, input_image, age_transformers)

        save_results(aged_image)







    return 'Inference Done!'

# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port='8080')

    # app.run(port=int(os.environ.get("PORT", 8080)),host='0.0.0.0',debug=True)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)





























