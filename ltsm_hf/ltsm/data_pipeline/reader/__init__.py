from ltsm.data_pipeline.reader.monash_reader import MonashReader



reader_dict = {}

def register_reader(module):
    assert module.module_id not in reader_dict, f"Reader {module.module_id} alreader registered"
    reader_dict[module.module_id] = module

register_reader(MonashReader)
