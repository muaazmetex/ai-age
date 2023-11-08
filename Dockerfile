#FROM nvcr.io/nvidia/pytorch:23.09-py3

FROM ubuntu:20.04
#FROM nvidia/cuda:12.2.0-devel-ubuntu20.04

# Allow statements and log messages to immediately appear in the logs
ENV PYTHONUNBUFFERED True


# Set CUDA environment variables
ENV CUDA_HOME /usr/local/cuda
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Copy local code to the container image.
ADD . /home/model-server/
WORKDIR /home/model-server/

# Switch to root user for system-level operations
USER root

RUN apt-get update && \
    apt-get install -y curl git && \
    rm -rf /var/lib/apt/lists/*

#Uninstall and install CUDA

#RUN apt autoremove cuda* nvidia* --purge

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
RUN dpkg -i cuda-keyring_1.1-1_all.deb
RUN apt-get update
RUN apt-get -y install cuda
RUN nvidia-smi

RUN pip install torch==1.11.0 torchvision==0.12.0

# Install production dependencies.
RUN apt-get clean && apt-get -y update && apt-get install -y build-essential libopenblas-dev liblapack-dev libopenblas-dev liblapack-dev
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y cmake
RUN pip install --upgrade pip setuptools

RUN apt-get install -y wget
RUN pip install -r requirements.txt
#RUN wget "https://raw.githubusercontent.com/italojs/facial-landmarks-recognition/master/shape_predictor_68_face_landmarks.dat"
#RUN pip install gdown
#RUN gdown "https://drive.google.com/u/0/uc?id=1XyumF6_fdAxFmxpFcmPf-q84LU_22EMC&export=download"
RUN pip install --upgrade pip setuptools
RUN pip install gunicorn
#RUN pip install dlib==19.24.2





# Install PyTorch and related packages using pip
#RUN pip install torch==1.13.1+cu116 torchvision -f https://download.pytorch.org/whl/torch_stable.html


# Switch back to the non-root user
#USER <your-username>

# EXPOSE 5000
CMD exec gunicorn --bind :5000 --workers 1 --threads 8 --timeout 0 app:app
 