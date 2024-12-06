from going_modular.MagLinear import MagLinear
import torch

class FaceRecognition(torch.nn.Module):
    def __init__(self, feature_extractor, classification=MagLinear(512,440)):
        super(FaceRecognition, self).__init__()
        self.feature_extractor = feature_extractor
        self.classification = classification
    
    def forward(self, x, target):
        x = self.feature_extractor(x)
        logits, x_norm = self.classification(x)
        
        return logits, x_norm