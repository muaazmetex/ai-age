from flask import jsonify
import dlib
from argparse import Namespace
import os
import sys
import pprint
import random
import string
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from datasets.augmentations import AgeTransformer
from utils.common import tensor2im
from models.psp import pSp



IMAGES_FOLDER = "images/"

def generate_random_string(length=10):
    characters = string.ascii_letters + string.digits
    random_string = ''.join(random.choice(characters) for _ in range(length))
    return random_string



def exp_data_args(image_path):

    # Define Inference Parameters
    EXPERIMENT_DATA_ARGS = {
        "ffhq_aging": {
            "model_path": "pretrained_models/sam_ffhq_aging.pt",
            "image_path": image_path,
            "transform": transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        }
    }

    return EXPERIMENT_DATA_ARGS


def load_model(model_path):
    
    
    # Load Pretrained Model
    model_path = model_path
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
    return net


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


# Run Inference
def run_inference(aligned_image, EXPERIMENT_ARGS):
    img_transforms = EXPERIMENT_ARGS['transform']
    input_image = img_transforms(aligned_image)
    return img_transforms, input_image


def age_transformer():
    # we'll run the image on multiple target ages
    #target_ages = [100]
    target_ages = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    age_transformers = [AgeTransformer(target_age=age) for age in target_ages]

    return age_transformers


def run_on_batch(inputs, net):
    result_batch = net(inputs.to("cuda").float(), randomize_noise=False, resize=False)
    return result_batch


def get_inference(target_age_limit, aligned_image, input_image, age_transformers, net):
    try:
        age_results_dict = {}  # Dictionary to store result images for each age target
        results = np.array(aligned_image.resize((1024, 1024)))
        
        for age_transformer in age_transformers:
            print(f"Running on target age: {age_transformer.target_age}")
            with torch.no_grad():
                input_image_age = [age_transformer(input_image.cpu()).to('cuda')]
                input_image_age = torch.stack(input_image_age)
                result_tensor = run_on_batch(input_image_age, net)[0]
                result_image = tensor2im(result_tensor)
                # results = np.concatenate([results, result_image], axis=1)

            age_results_dict[age_transformer.target_age] = result_image

        result_images = list(age_results_dict.values())

        # Return the list of result images and the dictionary of age-based result images
        return result_images, age_results_dict

    except Exception as e:
        print(f"The error occurred during inference.\nError: {e}")

def show_image(image):
    plt.imshow(image)


def concatenated_image(results):

    results = Image.fromarray(results)
    return results   # this is a very large image (11*1024 x 1024) so it may take some time to display!   


def save_results(age_results_dict, filename):
    try:
        saved_image_paths = []

        for age, result_image in age_results_dict.items():
            # Save each image from the age_results_dict
            image_path = f"images/{filename}.jpg"  # Remove age from filename
            result_image.save(image_path)
            saved_image_paths.append(image_path)

        print("Results saved successfully!")

        return saved_image_paths

    except Exception as e:
        print(f"Error in saving images: {e}")


def get_image_paths(base_url, saved_image_paths):
    try:
        image_urls = [f'{base_url}{filename}' for filename in saved_image_paths]

        image_dict = {}
        for age, url in enumerate(image_urls, start=0):
            age = age * 10
            key = f"{age}"
            image_dict[key] = url

        return {'image_urls': image_dict}
    except Exception as e:
        return {"image_error": f"Image not found. \nError:{e}"}







































