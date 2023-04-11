An extension for stable-diffusion-webui that segment the image elements

## Installation

download this script to scripts/segment_anything.py

```shell
cd stable-diffusion-webui
wget -O scripts/segment_anything.py https://github.com/Erickrus/stable-diffusion-webui-segment-anything/blob/main/scripts/segment_anything.py
```

install dependencies
```shell
pip3 install opencv-python matplotlib onnx onnxruntime
pip3 install 'git+https://github.com/facebookresearch/segment-anything.git'
```

## User Interface

![](about.png)
