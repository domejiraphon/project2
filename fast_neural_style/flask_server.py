from __future__ import print_function
import datetime
from flask import Flask, render_template, request
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np
import os 
import matplotlib.pyplot as plt
from neural_style.transformer_net import TransformerNet
from loguru import logger 
from neural_style import utils
import re

app = Flask(__name__)
port = int(os.getenv("PORT"))

@app.route('/')
def hello():
  return render_template('index.html', utc_dt=datetime.datetime.utcnow())

@app.route('/predict')
def query():
  model_type = request.args.get('model', type=str)
  content_image = os.path.join("static", request.args.get('content', type=str))
  img_path = "static/test.jpg"
  stylize(model_type, content_image, img_path)
  
  return render_template("img.html", user_image=img_path)

def stylize(model_type, content_image, img_path):
    device = torch.device("cpu")

    content_image = utils.load_image(content_image, scale=None)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)
    model_path = {"candy": "saved_models/candy.pth",
                  "mosaic": "saved_models/mosaic.pth",
                  "rain_princess": "saved_models/rain_princess/pth",
                  "udnie": "saved_models/udnie.pth"}
    logger.info("Start inference")
    with torch.no_grad():
        style_model = TransformerNet()
        
        state_dict = torch.load(model_path[model_type])
        # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
        for k in list(state_dict.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del state_dict[k]
        logger.info("Load pretrained model")
        style_model.load_state_dict(state_dict)
        style_model.to(device)
        style_model.eval()
        output = style_model(content_image).cpu()
    
    logger.info(f"The output image has been saved to {img_path}")
    utils.save_image(img_path, output[0])
    
  
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=port)
    