import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from models.networks.resnet_encoder import ResnetEncoder
from models.networks.depth_decoder import DepthDecoder

from skimage import io
class MonoDepth(nn.Module):


    def __init__(self, model_name='mono_640x192'):
        super().__init__()
        model_path = Path(__file__).parent
        model_path = model_path / 'weights/{}'.format(model_name)
        encoder_path = os.path.join(model_path, "encoder.pth")
        depth_decoder_path = os.path.join(model_path, "depth.pth")

        self.encoder = ResnetEncoder(18, False)
        loaded_dict_enc = torch.load(encoder_path)
        # extract the height and width of image that this model was trained with
        self.image_height = loaded_dict_enc['height']
        self.image_width = loaded_dict_enc['width']
        filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in self.encoder.state_dict()}
        self.encoder.load_state_dict(filtered_dict_enc)
        print('Loaded MonoDepth2 encoder')
        self.depth_decoder = DepthDecoder(num_ch_enc = self.encoder.num_ch_enc, 
                                          scales=range(4))

        loaded_dict = torch.load(depth_decoder_path)
        self.depth_decoder.load_state_dict(loaded_dict)
        print('Loaded MonoDepth2 decoder')

    def forward(self, images):
        original_shape = images.shape
        x = F.interpolate(images, (self.image_height, self.image_width), mode="bilinear", align_corners=False)
        features = self.encoder(x)
        outputs = self.depth_decoder(features)
        disp = outputs[("disp", 0)]
        disp = F.interpolate(disp, (original_shape[-2], original_shape[-1]), mode="bilinear", align_corners=False)
        return disp

