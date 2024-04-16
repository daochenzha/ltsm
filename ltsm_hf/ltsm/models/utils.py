def get_model(config):

    if config.model == 'PatchTST':
        from ltsm.models.PatchTST import PatchTST
        model = PatchTST(config)
    elif config.model == 'DLinear':
        from ltsm.models.DLinear import DLinear
        model = DLinear(config)
    elif config.model == 'TimeLLM':
        from ltsm.models.TimeLLM import TimeLLM
        model = TimeLLM(config)
    elif config.model == 'Chronos':
        from ltsm.models.Chronos import Chronos
        model = Chronos(config)
    else:
        from ltsm.models.ltsm_model import LTSM
        if config.local_pretrain == "None":
            model = LTSM(config)
        else:
            model = LTSM.from_pretrained(config.local_pretrain, config=config)

    return model
