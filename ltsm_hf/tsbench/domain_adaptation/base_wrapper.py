class TimeSeriesDomainAdaptationWrapper:
    # add typing for functions
    def __init__(self, source_model, target_model, configs, load_past=None):
        # Possibly adding new layers to the original model
        pass

    def new_train_forward(self, source_x, target_x):
        # Possibly modify the forward pass process in training stage of the original model
        # remove "new" from the function name, find another word
        pass
    
    def new_inference_forward(self, x, domain_flag):
        # Possibly modify the forward pass process in inference stage of the original model
        # remove "new" from the function name, find another word
        pass

    def loss(self, source_pred, target_pred, source_y, target_y):
        # Implement the loss
        pass

    def train(self, data, lr, n_epoch):
        # Implement the training process
        # move batch, lr, n_epoch to config and store it in init function
        pass
