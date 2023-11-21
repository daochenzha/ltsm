def get_model(config):

    if config.model == 'PatchTST':
        from ltsm.models.PatchTST import PatchTST
        model = PatchTST(config)
    elif config.model == 'DLinear':
        from ltsm.models.DLinear import DLinear
        model = DLinear(config)
    else:
        from ltsm.models.ltsm_model import LTSM
        model = LTSM(config)

    return model
