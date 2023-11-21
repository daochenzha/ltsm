import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TimeSeriesForecastModel(nn.Module):
    # add typing to functions
    def __init__(self, load_past=None):
        pass

    def pre_process(self, data):
        # Pre-process the input data
        pass

    def encoder(self, preprocessed_data):
        # Implement the encoder component
        pass

    def decoder(self, encoded_data):
        # Implement the decoder component
        pass

    def post_process(self, decoded_data):
        # Post-process the decoded data
        pass
    
    def forward(self, data):
        # Forward function
        pass




