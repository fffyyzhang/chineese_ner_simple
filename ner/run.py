
import logging
import sys
import numpy as np
from dataclasses import dataclass,field
from typing import Optional,Union
from datasets.utils.download_manager import GenerateMode
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
    AutoModelForTokenClassification
)

from datasets import load_dataset
from args import ModelArguments, DataTrainingArguments

def main():
    #model = AutoModelForTokenClassification.from_pretrained("bert-base-chinese")
    #m2 =model.bert_model
    d=1

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

if __name__ == "__main__":
    main()