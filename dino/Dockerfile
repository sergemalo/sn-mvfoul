# Utiliser l'image de base NVIDIA CUDA avec Python 3.9
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

# Éviter les interactions pendant l'installation
ENV DEBIAN_FRONTEND=noninteractive

# Installer Python 3.9 et pip
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Définir Python 3.9 comme version par défaut
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1

# Mettre à jour pip
RUN python -m pip install --upgrade pip

# Installer mim
RUN pip install -U openmim

# Créer et définir le répertoire de travail
WORKDIR /app

# Installer les dépendances
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Test CUDA
COPY cuda.py ./
RUN python cuda.py

# Installer mmcv
RUN mim install mmcv-full==1.7.2

# Installer les dépendances source
COPY requirements-source.txt ./
RUN pip install -r requirements-source.txt

# Dirty fixes
# Replace `from mmcv import print_log` by `from mmengine import print_log` in /usr/local/lib/python3.9/dist-packages/mmcv/parallel/distributed.py
# COPY fixes/distributed.py /usr/local/lib/python3.9/dist-packages/mmcv/parallel/distributed.py

# Copier les fichiers Python
COPY main.py depther.py ./

# Commande par défaut
CMD ["python", "main.py"] 
