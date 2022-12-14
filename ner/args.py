from dataclasses import dataclass
from dataclasses import field
from typing import Optional

@dataclass
class DataTrainingArguments:
    max_seq_length: int = field(default=400, )
    cross_num: int = field(default=0, )
    dataset_name: str = field(default=None, )
    task_name: str = field(default='base', )
    cache_dir: str = field(default='./cache', )
    gradient_checkpointing: bool = field(default=False, )
    overwrite_cache: bool = field(default=False, )
    max_train_samples: Optional[int] = field(default=None, )
    max_eval_samples: Optional[int] = field(default=None, )
    max_predict_samples: Optional[int] = field(default=None, )


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default=None, )
    use_fast_tokenizer: bool = field(default=True, )
    model_revision: str = field(default="main", )
    use_auth_token: bool = field(default=False, )