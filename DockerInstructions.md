# Tensorflow docker installation instructions

- Linux required to run on GPU
- Image: `tensorflow/tensorflow:2.2.2-gpu-py3`
- CUDA > 9.0 driver installed

Instructions from: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker, modified to pull container needed

## Ubuntu

For other Linux distributions, refer to the link above.

### Setting up Docker

```bash
curl https://get.docker.com | sh \
   && sudo systemctl start docker \
   && sudo systemctl enable docker
```

### Setting up NVIDIA Container Toolkit

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
```

Install the `nvidia-docker2` package (and dependencies) after updating the package listing:

```bash
sudo apt-get update
sudo apt-get install -y nvidia-docker2
```

Restart the Docker daemon to complete the installation after setting the default runtime:

```bash
sudo systemctl restart docker
```

At this point, a working setup can be tested by running a base CUDA container:

```bash
sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

This should result in a console output shown below:

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 450.51.06    Driver Version: 450.51.06    CUDA Version: 11.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla T4            On   | 00000000:00:1E.0 Off |                    0 |
| N/A   34C    P8     9W /  70W |      0MiB / 15109MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

### Build Effects Container

```bash
dockerfile_path='./tensorflow/'
docker build -t effects:0.1 ${dockerfile_path}
```

Run container and link the database and repo folders

```bash
docker run -it --gpus all \
   -v ${local/repo/path}:/opt/DNNEffects/ \
   -v ${local/dataset/path}:/db/IDMT-SMT-AUDIO-EFFECTS/ \
   -w /opt/DNNEffects/ \
   effects:0.1 \
   bash
```
