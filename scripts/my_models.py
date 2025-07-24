
from pydantic import BaseModel
from core.models.utility_models import DpoDatasetType
from core.models.utility_models import FileFormat
from core.models.utility_models import GrpoDatasetType
from core.models.utility_models import InstructTextDatasetType
from core.models.utility_models import TaskType
from core.models.utility_models import TextDatasetType

class ConfigRequest(BaseModel):
    m_id: str | None=None
    job_id: str | None = None
    worker_id: str | None = None
    num_gpus: int
    model_id: str
    model_size: int | None = None
    remained_minutes: int
    model_conf: dict | None = None
    dataset_num_rows: int
    dataset_type: TextDatasetType