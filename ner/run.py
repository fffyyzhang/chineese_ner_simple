
import logging
import sys
import numpy as np
from dataclasses import dataclass,field
from typing import Optional,Union
from datasets.utils.download_manager import GenerateMode
from torch.utils.data.dataloader import DataLoader
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
    AutoModelForTokenClassification,
    Trainer,
    DataCollatorForTokenClassification
)

from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from datasets import load_dataset, load_from_disk
from args import ModelArguments, DataTrainingArguments

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    dataset=load_from_disk('/home/liyuan/data/ner/msra/dataset.dump')

    max_seq_length=510

    #可以挪到预处理那段代码里面去
    dataset['train'] = dataset['train'].filter(lambda example: len(example['tokens']) <= 510)
    dataset['test'] = dataset['test'].filter(lambda example: len(example['tokens']) <= 510)


    cls_label = dataset['train'].features['ner_tags'].feature
    num_labels=cls_label.num_classes

    max_seq_length = data_args.max_seq_length
    config = AutoConfig.from_pretrained(model_args.model_name_or_path,
                                        num_labels=num_labels)
    config.max_length = max_seq_length


    #setattr(config, "gradient_checkpointing", data_args.gradient_checkpointing)
    #-liy 修改了一下，增加这个option之后会报option conflict的错误
    setattr(config, "gradient_checkpointing", False)
    #logger.info(config)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    model = AutoModelForTokenClassification\
        .from_pretrained(model_args.model_name_or_path,config=config)
    model.to(device)

    def preprocess_function(examples):
        #Run tokenizer on 1 batch in dataset
        result = tokenizer(examples["tokens"], padding=False, max_length=max_seq_length,
                           truncation=True,  is_split_into_words=True,
#                           return_offsets_mapping=True
                           )

        #liy:map CLS and SEP to -100
        result['labels']= [[-100]+tags+[-100] for tags in examples['ner_tags']]

        l1 = [len(l) for l in result['input_ids']]
        l2 =  [len(l) for l in result['labels']]

        if l1!=l2:
            d=1

        return result

    dataset = dataset.map(preprocess_function, batched=True,
                            remove_columns=dataset["train"].column_names,
                            load_from_cache_file=not data_args.overwrite_cache,
                            desc="Running tokenizer on dataset",
                            num_proc=1)

    # split train to train and dev
    ds_splited = dataset['train'].train_test_split(test_size=0.1, shuffle=True)
    ds_train, ds_dev, ds_test = ds_splited['train'], ds_splited['test'], dataset['test']

    @dataclass
    class DataCollatorForMSRA:
        tokenizer: PreTrainedTokenizerBase
        padding: Union[bool, str, PaddingStrategy] = True
        max_length: Optional[int] = None
        pad_to_multiple_of: Optional[int] = None

        def __call__(self, features):
            batch = self.tokenizer.pad(
                features,
                padding=True,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                # return_tensors='pt',
            )

            feed_dict={}
            for i in ['input_ids', 'attention_mask', 'token_type_ids']:
                feed_dict[i] = torch.tensor(batch[i], dtype=torch.long).to(device)

            #这个name在model里面怎么定的呢，应该是被统一叫labels了
            #labels这个东西如果在collator阶段pad的话，就需要手动填充了！！！！，
            # 在tokenize阶段pad可能就不需要
            feed_dict['labels'] = torch.tensor(batch['labels'], dtype=torch.float32).to(device)


            return feed_dict



            # sequence_length = torch.tensor(batch["input_ids"]).shape[1]
            #
            # labels = [feature['labels'] for feature in features]
            # batch["labels"] = [label + [[0.0] * num_labels * 2] * (sequence_length - len(label)) for label in labels]
            #
            # final_batch = {}
            # for i in ['input_ids', 'attention_mask', 'token_type_ids']:
            #     final_batch[i] = torch.tensor(batch[i], dtype=torch.long).to(device)
            #
            # for i in ['labels']:
            #     final_batch[i] = torch.tensor(batch[i], dtype=torch.float32).to(device)
            # return final_batch

    #data_collator = DataCollatorForMSRA(tokenizer, pad_to_multiple_of=8)
    data_collator_msra = DataCollatorForTokenClassification(tokenizer=tokenizer,pad_to_multiple_of=8)
    train_dataloader = DataLoader(ds_train, shuffle=True, collate_fn=data_collator_msra,
                                      batch_size=training_args.per_device_train_batch_size)
    eval_dataloader = DataLoader(ds_dev, collate_fn=data_collator_msra,
                                     batch_size=training_args.per_device_eval_batch_size)
    predict_dataloader = DataLoader(ds_test, collate_fn=data_collator_msra,
                                        batch_size=training_args.per_device_eval_batch_size)


    for step, batch in enumerate(train_dataloader):
        print(step)
        d = 1

    # try:
    #     for step, batch in enumerate(train_dataloader):
    #         d=1
    # except Exception as e:
    #     d=1

    # trainer = Trainer(
    #     model=model,  # the instantiated   Transformers model to be trained
    #     args=training_args,  # training arguments, defined above
    #     train_dataset=ds_train,  # training dataset
    #     eval_dataset=ds_dev,  # evaluation dataset
    #     data_collator=data_collator_msra
    #     #compute_metrics=compute_metrics
    # )

    #trainer.train()


if __name__ == "__main__":
    main()