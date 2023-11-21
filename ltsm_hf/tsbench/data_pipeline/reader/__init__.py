from tsbench.data_pipeline.reader.monash_reader import MonashReader
from tsbench.data_pipeline.reader.spm_reader import SPMReader
from tsbench.data_pipeline.reader.eyelink_reader import EyeLinkReader
from tsbench.data_pipeline.reader.opm_auditory_reader import OPM_AuditoryReader
from tsbench.data_pipeline.reader.somato_reader import SomatoReader
from tsbench.data_pipeline.reader.kiloword_reader import KilowordReader
from tsbench.data_pipeline.reader.opm_reader import OPMReader
from tsbench.data_pipeline.reader.erp_core_reader import ERP_CoreReader
from tsbench.data_pipeline.reader.ssvep_reader import SSVEPReader
from tsbench.data_pipeline.reader.mtrf_reader import MTRFReader
from tsbench.data_pipeline.reader.hf_sef_reader import Hf_SefReader


reader_dict = {}

def register_reader(module):
    assert module.module_id not in reader_dict, f"Reader {module.module_id} alreader registered"
    reader_dict[module.module_id] = module

register_reader(MonashReader)
register_reader(SPMReader)
register_reader(EyeLinkReader)
register_reader(OPM_AuditoryReader)
register_reader(SomatoReader)
register_reader(KilowordReader)
register_reader(OPMReader)
register_reader(ERP_CoreReader)
register_reader(SSVEPReader)
register_reader(MTRFReader)
register_reader(Hf_SefReader)
