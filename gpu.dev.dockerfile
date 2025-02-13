FROM nvcr.io/nvidia/pytorch:23.09-py3
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia
ENV SHELL /bin/bash
ENV IS_GPU_ENABLED TRUE
RUN pip3 install --upgrade pip
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install libyaml-dev -y
RUN pip3 install fvcore iopath
RUN apt update -y
RUN apt install python3-tk -y
RUN pip3 install --upgrade opencv-python opencv-python-headless
COPY requirements.txt /workspaces/requirements.txt
RUN pip3 install -r /workspaces/requirements.txt
RUN apt install build-essential python3-dev libgl1-mesa-dev -y
WORKDIR /workspaces/Workfiles
RUN git clone https://github.com/facebookresearch/pytorch3d & \
    cd pytorch3d && pip install -e . & rm -rf pytorch3d
# RUN pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113

ARG USERNAME=snehashis
ARG USER_UID=1000
ARG USER_GID=$USER_UID
# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

RUN groupmod --gid $USER_GID $USERNAME \
    && usermod --uid $USER_UID --gid $USER_GID $USERNAME \
    && chown -R $USER_UID:$USER_GID /home/$USERNAME