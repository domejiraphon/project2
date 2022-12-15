from typing import NamedTuple

import kfp
from kfp.components import func_to_container_op, InputPath, OutputPath, create_component_from_func
import argparse
import os
import sys
import time
import re
import torch


def train(dataset_path: str,
          epochs: int, 
          style_image: str,
          trained_model_path: OutputPath(torch.nn.Module),
          image_size: int = 224,
          batch_size: int = 4,
         ):
    import numpy as np
    import torch
    from torch.optim import Adam
    from torch.utils.data import DataLoader
    from torchvision import datasets
    from torchvision import transforms
    import torch.onnx
    from PIL import Image
    #import utils
    #from transformer_net import TransformerNet
    #from vgg import Vgg16
    from loguru import logger 
    from collections import namedtuple
    from torchvision import models
    import time, sys, os
    
    class TransformerNet(torch.nn.Module):
        def __init__(self):
            super(TransformerNet, self).__init__()
            # Initial convolution layers
            self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
            self.in1 = torch.nn.InstanceNorm2d(32, affine=True)
            self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
            self.in2 = torch.nn.InstanceNorm2d(64, affine=True)
            self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
            self.in3 = torch.nn.InstanceNorm2d(128, affine=True)
            # Residual layers
            self.res1 = ResidualBlock(128)
            self.res2 = ResidualBlock(128)
            self.res3 = ResidualBlock(128)
            self.res4 = ResidualBlock(128)
            self.res5 = ResidualBlock(128)
            # Upsampling Layers
            self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
            self.in4 = torch.nn.InstanceNorm2d(64, affine=True)
            self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
            self.in5 = torch.nn.InstanceNorm2d(32, affine=True)
            self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)
            # Non-linearities
            self.relu = torch.nn.ReLU()

        def forward(self, X):
            y = self.relu(self.in1(self.conv1(X)))
            y = self.relu(self.in2(self.conv2(y)))
            y = self.relu(self.in3(self.conv3(y)))
            y = self.res1(y)
            y = self.res2(y)
            y = self.res3(y)
            y = self.res4(y)
            y = self.res5(y)
            y = self.relu(self.in4(self.deconv1(y)))
            y = self.relu(self.in5(self.deconv2(y)))
            y = self.deconv3(y)
            return y

    class ConvLayer(torch.nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, stride):
            super(ConvLayer, self).__init__()
            reflection_padding = kernel_size // 2
            self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
            self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

        def forward(self, x):
            out = self.reflection_pad(x)
            out = self.conv2d(out)
            return out

    class ResidualBlock(torch.nn.Module):
        """ResidualBlock
        introduced in: https://arxiv.org/abs/1512.03385
        recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
        """

        def __init__(self, channels):
            super(ResidualBlock, self).__init__()
            self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
            self.in1 = torch.nn.InstanceNorm2d(channels, affine=True)
            self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
            self.in2 = torch.nn.InstanceNorm2d(channels, affine=True)
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            residual = x
            out = self.relu(self.in1(self.conv1(x)))
            out = self.in2(self.conv2(out))
            out = out + residual
            return out

    class UpsampleConvLayer(torch.nn.Module):
        """UpsampleConvLayer
        Upsamples the input and then does a convolution. This method gives better results
        compared to ConvTranspose2d.
        ref: http://distill.pub/2016/deconv-checkerboard/
        """

        def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
            super(UpsampleConvLayer, self).__init__()
            self.upsample = upsample
            reflection_padding = kernel_size // 2
            self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
            self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

        def forward(self, x):
            x_in = x
            if self.upsample:
                x_in = torch.nn.functional.interpolate(x_in, mode='nearest', scale_factor=self.upsample)
            out = self.reflection_pad(x_in)
            out = self.conv2d(out)
            return out
        
    class Vgg16(torch.nn.Module):
        def __init__(self, requires_grad=False):
            super(Vgg16, self).__init__()
            vgg_pretrained_features = models.vgg16(pretrained=True).features
            self.slice1 = torch.nn.Sequential()
            self.slice2 = torch.nn.Sequential()
            self.slice3 = torch.nn.Sequential()
            self.slice4 = torch.nn.Sequential()
            for x in range(4):
                self.slice1.add_module(str(x), vgg_pretrained_features[x])
            for x in range(4, 9):
                self.slice2.add_module(str(x), vgg_pretrained_features[x])
            for x in range(9, 16):
                self.slice3.add_module(str(x), vgg_pretrained_features[x])
            for x in range(16, 23):
                self.slice4.add_module(str(x), vgg_pretrained_features[x])
            if not requires_grad:
                for param in self.parameters():
                    param.requires_grad = False

        def forward(self, X):
            h = self.slice1(X)
            h_relu1_2 = h
            h = self.slice2(h)
            h_relu2_2 = h
            h = self.slice3(h)
            h_relu3_3 = h
            h = self.slice4(h)
            h_relu4_3 = h
            vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
            out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
            return out

    def load_image(filename, size=None, scale=None):
        img = Image.open(filename).convert('RGB')
        if size is not None:
            img = img.resize((size, size), Image.ANTIALIAS)
        elif scale is not None:
            img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
        return img
    
    def gram_matrix(y):
        (b, ch, h, w) = y.size()
        features = y.view(b, ch, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (ch * h * w)
        return gram
    
    def normalize_batch(batch):
        # normalize using imagenet mean and std
        mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        batch = batch.div_(255.0)
        return (batch - mean) / std
    
    """
    Load dataset from dataset_path
    """
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    dummy = False
    if dummy:
        logger.info("=> Dummy data is used!")
        train_dataset = datasets.FakeData(10, (3, image_size, image_size), 10, transforms.ToTensor())
    else:
        train_dataset = datasets.ImageFolder(dataset_path, 
                                             transform)
    train_loader = DataLoader(train_dataset, 
                              batch_size=batch_size)
    np.random.seed(0)
    torch.manual_seed(0)
    device = torch.device('cpu')
    
    """
    Load model
    """
    content_weight = 1e5
    style_weight = 1e10
    lr = 1e-3
    transformer = TransformerNet().to(device)
    optimizer = Adam(transformer.parameters(), lr)
    mse_loss = torch.nn.MSELoss()
    
    vgg = Vgg16(requires_grad=False).to(device)
    
    style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    if dummy:
        numpy_image = np.zeros((image_size, image_size, 3))
        style = Image.fromarray(np.uint8(numpy_image)).convert('RGB')
    else:
        style = load_image(style_image)
    style = style_transform(style)
    style = style.repeat(batch_size, 1, 1, 1).to(device)
    
    
    features_style = vgg(normalize_batch(style))
    gram_style = [gram_matrix(y) for y in features_style]
    
    """
    Start training
    
    """
    
    logger.info("Start training")
    
    for e in range(epochs):
        transformer.train()
        agg_content_loss = 0.
        agg_style_loss = 0.
        count = 0
        for batch_id, (x, _) in enumerate(train_loader):
            #logger.info(f"Step: {batch_id}")
            n_batch = len(x)
            count += n_batch
            optimizer.zero_grad()

            x = x.to(device)
            y = transformer(x)

            y = normalize_batch(y)
            x = normalize_batch(x)

            features_y = vgg(y)
            features_x = vgg(x)

            content_loss = content_weight * mse_loss(features_y.relu2_2, features_x.relu2_2)

            style_loss = 0.
            for ft_y, gm_s in zip(features_y, gram_style):
                gm_y = gram_matrix(ft_y)
                style_loss += mse_loss(gm_y, gm_s[:n_batch, :, :])
            style_loss *=  style_weight

            total_loss = content_loss + style_loss
            total_loss.backward()
            optimizer.step()

            agg_content_loss += content_loss.item()
            agg_style_loss += style_loss.item()

            if (batch_id + 1) % 1 == 0:
                mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}".format(
                    time.ctime(), e + 1, count, len(train_dataset),
                                  agg_content_loss / (batch_id + 1),
                                  agg_style_loss / (batch_id + 1),
                                  (agg_content_loss + agg_style_loss) / (batch_id + 1)
                )
                logger.info(mesg)


    """
    save model
    """
    transformer.eval().cpu()
    save_model_dir = "./"
    save_model_filename = "ckpt.pth"
    save_model_path = os.path.join(save_model_dir, save_model_filename)
    torch.save(transformer.state_dict(), save_model_path)

    logger.info("\nDone, trained model saved at", save_model_path)
    
    """
    with open(trained_model_path, 'w') as writer:
        writer.write(transformer)
    """
    torch.save(transformer.state_dict(), trained_model_path)
            
def infer(
         trained_model_path: InputPath(torch.nn.Module),
         content_img_path: str,
         output_image_path: str="./out.jpg",
         ):
    import numpy as np
    import torch
    from torch.optim import Adam
    from torch.utils.data import DataLoader
    from torchvision import datasets
    from torchvision import transforms
    import torch.onnx
    from PIL import Image
    #import utils
    #from transformer_net import TransformerNet
    #from vgg import Vgg16
    from loguru import logger 
    import time, sys, os, re
    
    class TransformerNet(torch.nn.Module):
        def __init__(self):
            super(TransformerNet, self).__init__()
            # Initial convolution layers
            self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
            self.in1 = torch.nn.InstanceNorm2d(32, affine=True)
            self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
            self.in2 = torch.nn.InstanceNorm2d(64, affine=True)
            self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
            self.in3 = torch.nn.InstanceNorm2d(128, affine=True)
            # Residual layers
            self.res1 = ResidualBlock(128)
            self.res2 = ResidualBlock(128)
            self.res3 = ResidualBlock(128)
            self.res4 = ResidualBlock(128)
            self.res5 = ResidualBlock(128)
            # Upsampling Layers
            self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
            self.in4 = torch.nn.InstanceNorm2d(64, affine=True)
            self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
            self.in5 = torch.nn.InstanceNorm2d(32, affine=True)
            self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)
            # Non-linearities
            self.relu = torch.nn.ReLU()

        def forward(self, X):
            y = self.relu(self.in1(self.conv1(X)))
            y = self.relu(self.in2(self.conv2(y)))
            y = self.relu(self.in3(self.conv3(y)))
            y = self.res1(y)
            y = self.res2(y)
            y = self.res3(y)
            y = self.res4(y)
            y = self.res5(y)
            y = self.relu(self.in4(self.deconv1(y)))
            y = self.relu(self.in5(self.deconv2(y)))
            y = self.deconv3(y)
            return y

    class ConvLayer(torch.nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, stride):
            super(ConvLayer, self).__init__()
            reflection_padding = kernel_size // 2
            self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
            self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

        def forward(self, x):
            out = self.reflection_pad(x)
            out = self.conv2d(out)
            return out

    class ResidualBlock(torch.nn.Module):
        """ResidualBlock
        introduced in: https://arxiv.org/abs/1512.03385
        recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
        """

        def __init__(self, channels):
            super(ResidualBlock, self).__init__()
            self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
            self.in1 = torch.nn.InstanceNorm2d(channels, affine=True)
            self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
            self.in2 = torch.nn.InstanceNorm2d(channels, affine=True)
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            residual = x
            out = self.relu(self.in1(self.conv1(x)))
            out = self.in2(self.conv2(out))
            out = out + residual
            return out

    class UpsampleConvLayer(torch.nn.Module):
        """UpsampleConvLayer
        Upsamples the input and then does a convolution. This method gives better results
        compared to ConvTranspose2d.
        ref: http://distill.pub/2016/deconv-checkerboard/
        """

        def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
            super(UpsampleConvLayer, self).__init__()
            self.upsample = upsample
            reflection_padding = kernel_size // 2
            self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
            self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

        def forward(self, x):
            x_in = x
            if self.upsample:
                x_in = torch.nn.functional.interpolate(x_in, mode='nearest', scale_factor=self.upsample)
            out = self.reflection_pad(x_in)
            out = self.conv2d(out)
            return out
    
    def load_image(filename, size=None, scale=None):
        img = Image.open(filename).convert('RGB')
        if size is not None:
            img = img.resize((size, size), Image.ANTIALIAS)
        elif scale is not None:
            img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
        return img
    """
    Load inference image
    """
    device = torch.device("cpu")
    dummy = False
    if dummy:
      numpy_image = np.zeros((100, 100, 3))
      content_image = Image.fromarray(np.uint8(numpy_image)).convert('RGB')
    else:
      content_image = load_image(content_img_path)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)
    logger.info("Start inference")
    with torch.no_grad():
        style_model = TransformerNet()
        """
        Load model from previously trained component
        """
       
        state_dict = torch.load(trained_model_path)
        # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
        for k in list(state_dict.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del state_dict[k]
        logger.info("Load pretrained model")
        style_model.load_state_dict(state_dict)
        
        #style_model.to(device)
        style_model.eval()
        output = style_model(content_image).cpu()
    logger.info(f"The output image has been saved to {output_image_path}")
    
    def save_image(filename, data):
        img = data.clone().clamp(0, 255).numpy()
        img = img.transpose(1, 2, 0).astype("uint8")
        img = Image.fromarray(img)
        img.save(filename)
        
    save_image(output_image_path, output[0])

train_op = create_component_from_func(
                    func = train,
                    base_image="jy3694/kubeflow_component_1:latest",
                    packages_to_install=["loguru"]
                )
infer_op = create_component_from_func(
                    func = infer,
                    base_image="jy3694/kubeflow_component_1:latest",
                    packages_to_install=["loguru"]
                )
    
def my_pipeline(dataset_path: str="/data/coco_sampled", 
                epochs: int=1,
                style_image: str="/data/images/style-images/mosaic.jpg",
                content_img_path: str="/data/images/content-images/1.png",
                output_image_path: str="./out.jpg",):
    dataset_path = "/data/coco_sampled"
    trained_model = train_op(dataset_path,
                            epochs,
                            style_image,)
    infer_op(trained_model.output, 
          content_img_path=content_img_path,
          output_image_path=output_image_path)


if __name__ == '__main__':
    # Compiling the pipeline
    kfp.compiler.Compiler().compile(my_pipeline, 'pipeline.yaml')