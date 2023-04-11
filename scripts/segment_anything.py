import html


# installation
# pip3 install opencv-python matplotlib onnx onnxruntime
# pip3 install 'git+https://github.com/facebookresearch/segment-anything.git'
# wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth



import cv2
import numpy as np
import torch

from PIL import Image

#import matplotlib.pyplot as plt

import os

from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.onnx import SamOnnxModel

import onnxruntime
from onnxruntime.quantization import QuantType
from onnxruntime.quantization.quantize import quantize_dynamic

from modules import script_callbacks, shared

import gradio as gr
import torch

model_dir = "models/sam/"
checkpoint = model_dir + "sam_vit_h_4b8939.pth"
model_type = "vit_h"
if not os.path.exists(checkpoint):
    os.mkdir(model_dir)
    os.system("wget -O %s https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth" % model_dir)



sam = sam_model_registry[model_type](checkpoint=checkpoint)



def extract_onnx():
    import warnings
    onnx_model_path = model_dir + "sam_onnx_example.onnx"
    onnx_model = SamOnnxModel(sam, return_single_mask=True)

    dynamic_axes = {
        "point_coords": {1: "num_points"},
        "point_labels": {1: "num_points"},
    }

    embed_dim = sam.prompt_encoder.embed_dim
    embed_size = sam.prompt_encoder.image_embedding_size
    mask_input_size = [4 * x for x in embed_size]
    dummy_inputs = {
        "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float),
        "point_coords": torch.randint(low=0, high=1024, size=(1, 5, 2), dtype=torch.float),
        "point_labels": torch.randint(low=0, high=4, size=(1, 5), dtype=torch.float),
        "mask_input": torch.randn(1, 1, *mask_input_size, dtype=torch.float),
        "has_mask_input": torch.tensor([1], dtype=torch.float),
        "orig_im_size": torch.tensor([1500, 2250], dtype=torch.float),
    }
    output_names = ["masks", "iou_predictions", "low_res_masks"]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        with open(onnx_model_path, "wb") as f:
            torch.onnx.export(
                onnx_model,
                tuple(dummy_inputs.values()),
                f,
                export_params=True,
                verbose=False,
                opset_version=17,
                do_constant_folding=True,
                input_names=list(dummy_inputs.keys()),
                output_names=output_names,
                dynamic_axes=dynamic_axes,
            )
    onnx_model_quantized_path = model_dir + "sam_onnx_quantized_example.onnx"
    quantize_dynamic(
        model_input=onnx_model_path,
        model_output=onnx_model_quantized_path,
        optimize_model=True,
        per_channel=False,
        reduce_range=False,
        weight_type=QuantType.QUInt8,
    )

onnx_model_quantized_path = model_dir + "sam_onnx_quantized_example.onnx"
if not os.path.exists(onnx_model_quantized_path):
    extract_onnx()
onnx_model_path = onnx_model_quantized_path
ort_session = onnxruntime.InferenceSession(onnx_model_path)

sam.to(device='cuda')
predictor = SamPredictor(sam)

def find_input_points(mask):
    input_points = []
    gray = 255 - np.array(mask)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 100:
            x, y = c[:,:,0].mean(), c[:,:,1].mean()
            input_points.append([int(x), int(y)])
    return input_points

def segment_anything(im):
    img = im['image'].convert("RGB")
    mask = im['mask'].convert("L")
    input_points = find_input_points(mask)

    image = np.array(img)

    predictor.set_image(image)
    image_embedding = predictor.get_image_embedding().cpu().numpy()

    input_point = np.array(input_points)
    input_label = np.ones(len(input_point)).astype(np.int)
    print(input_point)

    onnx_coord = np.concatenate([input_point, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
    onnx_label = np.concatenate([input_label, np.array([-1])], axis=0)[None, :].astype(np.float32)

    onnx_coord = predictor.transform.apply_coords(onnx_coord, image.shape[:2]).astype(np.float32)

    onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
    onnx_has_mask_input = np.zeros(1, dtype=np.float32)

    ort_inputs = {
        "image_embeddings": image_embedding,
        "point_coords": onnx_coord,
        "point_labels": onnx_label,
        "mask_input": onnx_mask_input,
        "has_mask_input": onnx_has_mask_input,
        "orig_im_size": np.array(image.shape[:2], dtype=np.float32)
    }

    masks, _, low_res_logits = ort_session.run(None, ort_inputs)
    masks = masks > predictor.model.mask_threshold

    masks = np.squeeze(np.squeeze(masks,0),0)
    masks = masks.astype(np.uint8)
    masks = np.repeat(masks[...,None], 3, axis=2)
    image = im['image'].convert("RGB")
    white = (255 * np.ones([image.size[0],image.size[1],3])).astype(np.uint8) 
    image = np.array(image) * masks + white * (1 - masks)
    
    return Image.fromarray(image)

def add_tab():
    with gr.Blocks(analytics_enabled=False) as ui:
        with gr.Tabs() as tabs:
            with gr.Blocks():
                with gr.Row():
                    im = gr.Image(type="pil", tool="sketch", source='upload', brush_radius=5, label="input")
                    output_im = gr.Image(type="pil", label="output")
            with gr.Tab("Segment Anything", id="input_image"):
                segment_anything_btn = gr.Button(value="Segment", variant="primary")

        segment_anything_btn.click(
            fn=segment_anything,
            inputs=[im],
            outputs=[output_im],
        )

    return [(ui, "Segment Anything", "segment_anything")]

script_callbacks.on_ui_tabs(add_tab)
