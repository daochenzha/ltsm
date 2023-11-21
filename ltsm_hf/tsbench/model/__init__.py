from tsbench.model.d_linear import DLinear

model_dict = {}

def register_model(module):
    assert module.module_id not in model_dict, f"Model {module.module_id} alreader registered"
    model_dict[module.module_id] = module

register_model(DLinear)
