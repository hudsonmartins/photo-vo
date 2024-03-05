
from torch import nn
import gluefactory as gf

class MotionEstimator(nn.Module):
    default_config = {
    }
    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}
        #TODO

    def forward(self, data):
        #TODO
        pass


class PhotoVoModel(nn.Module):
    default_config = {
    }
    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}
        self.motion_estimator = MotionEstimator(self.config)
        self.matcher = gf.models.get_model(config.model.name)(config.model)
    
    def forward(self, data):
        #TODO
        pass

    
def get_photo_vo_model(config):
    return PhotoVoModel(config)