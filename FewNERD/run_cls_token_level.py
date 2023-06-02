#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for token classification.
"""
# You can also adapt this script on your own token classification task and datasets. Pointers for this are left as
# comments.

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import evaluate
import numpy as np
from datasets import ClassLabel, load_dataset

import transformers
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    HfArgumentParser,
    PretrainedConfig,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from t5_encoder_classifier_token_level import T5ConfigForTokenClassification, T5ForTokenClassification
from t5_trainer_classifier import T5EncoderTrainer

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.30.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/token-classification/requirements.txt")

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )
    classifier_dropout: float = field(
        default=0.2,
        metadata={"help": "Dropout rate for the classifier layer."},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: Optional[str] = field(default="ner", metadata={"help": "The name of the task (ner, pos...)."})
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a csv or JSON file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate on (a csv or JSON file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to predict on (a csv or JSON file)."},
    )
    text_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of text to input in the file (a csv or JSON file)."}
    )
    label_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of label to input in the file (a csv or JSON file)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. If set, sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    label_all_tokens: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to put the label for one word on all tokens of generated by that word or just on the "
                "one (in which case the other tokens will have a padding index)."
            )
        },
    )
    return_entity_level_metrics: bool = field(
        default=False,
        metadata={"help": "Whether to return all the entity levels during evaluation or just the overall ones."},
    )
    label_list_file: Optional[str] = field(
        default=None,
        metadata={"help": "The file containing the label list."},
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        self.task_name = self.task_name.lower()

@dataclass
class MyTrainingArguments(TrainingArguments):
    ignore_label: Optional[str] = field(
        default="O",
        metadata={"help": "The label to ignore in the evaluation."},
    )
    learning_rate_proj: Optional[float] = field(
        default=1e-4,
        metadata={"help": "The learning rate for the projection layer."},
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, MyTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_ner", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
        extension = data_args.train_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir)
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
        features = raw_datasets["train"].features
    else:
        column_names = raw_datasets["validation"].column_names
        features = raw_datasets["validation"].features

    if data_args.text_column_name is not None:
        text_column_name = data_args.text_column_name
    elif "tokens" in column_names:
        text_column_name = "tokens"
    else:
        text_column_name = column_names[0]

    if data_args.label_column_name is not None:
        label_column_name = data_args.label_column_name
    elif f"{data_args.task_name}_tags" in column_names:
        label_column_name = f"{data_args.task_name}_tags"
    else:
        label_column_name = column_names[1]

    # In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the
    # unique labels.
    def get_label_list(labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        print(len(label_list), "labels:", label_list)
        # exit()
        return label_list

    # If the labels are of type ClassLabel, they are already integers and we have the map stored somewhere.
    # Otherwise, we have to get the list of labels manually.
    ''' directly read the label list from the txt file '''
    with open(data_args.label_list_file,"r") as f:
        label_list = f.readlines()
        label_list = [label.strip() for label in label_list]
        
    label2id = {label: i for i, label in enumerate(label_list)}
    label_to_id = label2id  
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = T5ConfigForTokenClassification.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    config.get_additional_args(num_labels=num_labels,classifier_dropout=model_args.classifier_dropout)

    tokenizer_name_or_path = model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path
    if config.model_type in {"bloom", "gpt2", "roberta"}:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=True,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            add_prefix_space=True,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=True,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    model = T5ForTokenClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )
    
    model.resize_token_embeddings(len(tokenizer))

    # Tokenizer check: this script requires a fast tokenizer.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models at"
            " https://huggingface.co/transformers/index.html#supported-frameworks to find the model types that meet"
            " this requirement"
        )

    # Model has labels -> use them.
    labels_are_int = False
    if model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id:
        if sorted(model.config.label2id.keys()) == sorted(label_list):
            # Reorganize `label_list` to match the ordering of the model.
            if labels_are_int:
                label_to_id = {i: int(model.config.label2id[l]) for i, l in enumerate(label_list)}
                label_list = [model.config.id2label[i] for i in range(num_labels)]
            else:
                label_list = [model.config.id2label[i] for i in range(num_labels)]
                label_to_id = {l: i for i, l in enumerate(label_list)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {sorted(model.config.label2id.keys())}, dataset labels:"
                f" {sorted(label_list)}.\nIgnoring the model labels as a result.",
            )
    # print(label_list)
    # print(label_to_id)
    # exit()

    # Set the correspondences label/ID inside the model config
    model.config.label2id = {l: i for i, l in enumerate(label_list)}
    model.config.id2label = dict(enumerate(label_list))

    # Map that sends B-Xxx label to its I-Xxx counterpart
    b_to_i_label = []
    for idx, label in enumerate(label_list):
        if label.startswith("B-") and label.replace("B-", "I-") in label_list:
            b_to_i_label.append(label_list.index(label.replace("B-", "I-")))
        else:
            b_to_i_label.append(idx)
    # print("len(b_to_i_label):", len(b_to_i_label))
    # print("b_to_i_label:", b_to_i_label)
    # exit()

    # Preprocessing the dataset
    # Padding strategy
    padding = "max_length" if data_args.pad_to_max_length else False

    # Tokenize all texts and align the labels with them.
    assert data_args.label_all_tokens, "in this experiment, to calculate the metrics, we want to have labels on all tokens instead of only the first one"
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            padding=padding,
            truncation=True,
            max_length=data_args.max_seq_length,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )
        labels = []
        # print(examples[label_column_name])
        # exit()
        for i, label in enumerate(examples[label_column_name]):
            if isinstance(label, str):
                # convert a string into list, e.g., "['a', 'b', 'c']" ==> ["a","b","c"]
                label = label.strip().replace("'", '"')
                label = json.loads(label)
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label_to_id[label[word_idx]])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    if data_args.label_all_tokens:
                        label_ids.append(b_to_i_label[label_to_id[label[word_idx]]])
                    else:
                        label_ids.append(-100)
                previous_word_idx = word_idx

            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        # print the first 2 examples, including the input text, tokenized token ids and labels
        # len(id) should >= len(text) because of the tokenization
        # len(id) == len(label) 
        print(tokenized_inputs["labels"][:2])
        print(tokenized_inputs["input_ids"][:2])
        print(examples[text_column_name][:2])
        # exit()
        return tokenized_inputs

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                tokenize_and_align_labels,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )

    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                tokenize_and_align_labels,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

    if training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                tokenize_and_align_labels,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )

    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)

    # Metrics
    def NER_metrics(preds, labels):
        '''
        preds: list[list[tuple]], e.g., [('am', '<person>'), ('Acme Corp.', '<organization>'), ('2006', '<misc>'), ('.', '<misc>')] 
        labels: list[list[tuple]], e.g., [('Jim', '<person>'), ('Acme Corp.', '<organization>'), ('2006', '<time>')]
        
        calculate 8 metrics:
        - for entity extration
            1. precision: # of correct extractions / # of extractions, e.g., 2/4
            2. recall: # of correct extractions / # of labels, e.g., 2/3
            3. F1: 2 * precision * recall / (precision + recall), e.g., 0.67
        - for entity typing
            4. accuracy: # of correct types / # of correct extractions, e.g., 1/2 (note the zero-division)
        - for type in-set rato
            5. in-set ratio: # of predictions in the predefined set / # of predictions, e.g., 2/4
        - for overall performance
            6. overall precision: # of correct predictions / # of predictions, e.g., 1/4
            7. overall recall: # of correct predictions / # of labels, e.g., 1/3
            8. overall F1: 2 * overall precision * overall recall / (overall precision + overall recall), e.g., 0.44
        '''
        number_of_correct_extract, number_of_correct_type, number_of_in_set,number_of_correct_prediction = 0, 0, 0, 0
        number_of_predictions = sum([len(pred) for pred in preds])
        number_of_labels = sum([len(label) for label in labels])
        assert len(preds) == len(labels), "The number of predictions and labels should be the same, but got {} and {}.".format(len(preds), len(labels))
        predefined_set = set()
        for pred, label in zip(preds, labels):
            for pd_entity in pred:
                assert isinstance(pd_entity, tuple)
                if pd_entity in label:
                    number_of_correct_prediction += 1
                for lb_entity in label:
                    assert isinstance(lb_entity, tuple)
                    predefined_set.add(lb_entity[1])
                    if pd_entity[0] == lb_entity[0]:
                        number_of_correct_extract += 1
                        if pd_entity[1] == lb_entity[1]:
                            number_of_correct_type += 1
        for pred in preds:
            for pd_entity in pred:
                if pd_entity[1] in predefined_set:
                    number_of_in_set += 1
                    
        results = {}
        results["extraction_precision"] = number_of_correct_extract / number_of_predictions if number_of_predictions != 0 else -1
        results["extraction_recall"] = number_of_correct_extract / number_of_labels if number_of_labels != 0 else -1
        results["extraction_F1"] = 2 * results["extraction_precision"] * results["extraction_recall"] / (results["extraction_precision"] + results["extraction_recall"]) if results["extraction_precision"] + results["extraction_recall"] != 0 else -1
        results["typing_accuracy"] = number_of_correct_type / number_of_correct_extract if number_of_correct_extract != 0 else -1
        results["in_set_ratio"] = number_of_in_set / number_of_predictions if number_of_predictions != 0 else -1
        results["overall_precision"] = number_of_correct_prediction / number_of_predictions if number_of_predictions != 0 else -1
        results["overall_recall"] = number_of_correct_prediction / number_of_labels if number_of_labels != 0 else -1
        results["overall_F1"] = 2 * results["overall_precision"] * results["overall_recall"] / (results["overall_precision"] + results["overall_recall"]) if results["overall_precision"] + results["overall_recall"] != 0 else -1
        results = {k: round(v * 100, 4) for k, v in results.items()} # convert to percentage
        results["number_of_predictions"] = number_of_predictions
        results["number_of_labels"] = number_of_labels
        return results
    
    def recover_entity_types(input_ids, label_ids, tokenizer, id2label, other_label="O"):
        '''
        find the input ids of named entities corresponding to the entity labels
        let tokenizer decode the input ids back to the named entities
        let id2label convert the entity labels back to the entity types
        
        return a list[list[tuple]], e.g., [[('Jim', '<person>'), ('Acme Corp.', '<organization>'), ('2006', '<time>')], ...]
        '''
        assert len(input_ids) == len(label_ids)
        entity_to_type_list = []  # list of list of tuples
        for input_id, label_id in zip(input_ids, label_ids):
            assert len(input_id) == len(label_id)
            label_name = [id2label[id] for id in label_id]
            entity_and_type = [] # lsit of tuples, like [('Jim', '<person>'), ('Acme Corp.', '<organization>'), ('2006', '<time>')]
            i = 0
            while i < len(label_name):
                label = label_name[i]
                if label != other_label:
                    start = i
                    end = i
                    while end + 1 < len(label_name) and label_name[end + 1] == label_name[start]:
                        end += 1
                    entity_ids = input_id[start: end + 1]
                    entity = tokenizer.decode(entity_ids, skip_special_tokens=True)
                    entity_and_type.append((entity, label))
                    i = end + 1
                else:
                    i += 1
            entity_to_type_list.append(entity_and_type)
        return entity_to_type_list
            
    metric = evaluate.load("seqeval")
    def compute_metrics(predictions, labels, input_ids):
        # convert the input_ids to list
        input_ids = input_ids if isinstance(input_ids, list) else input_ids.detach().cpu().tolist()
        # check the length of the input_ids
        assert len(input_ids) == len(predictions) == len(labels), "input_ids, predictions, and labels should have the same length, but got input_ids: {}, predictions: {}, labels: {}".format(len(input_ids), len(predictions), len(labels))
        # check the length of each input_id
        for i in range(len(input_ids)):
            assert len(input_ids[i]) == len(predictions[i]) == len(labels[i]), "input_ids, predictions, and labels should have the same length, but got input_ids[{}]: {}, predictions[{}]: {}, labels[{}]: {}".format(i, len(input_ids[i]), i, len(predictions[i]), i, len(labels[i]))
        
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens), such as [CLS], [SEP], [PAD]
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_input_ids = [
            [id for (p, l, id) in zip(prediction, label, input_id) if l != -100]
            for prediction, label, input_id in zip(predictions, labels, input_ids)
        ]
        
        processed_labels = recover_entity_types(true_input_ids, true_labels, tokenizer, model.config.id2label, other_label=training_args.ignore_label)
        processed_preds = recover_entity_types(true_input_ids, true_predictions, tokenizer, model.config.id2label, other_label=training_args.ignore_label)
        print(processed_preds[:2]) 
        print(processed_labels[:10])
        # exit()

        # self-defined metrics, we only care the quality of the extracted entities and their types
        results = NER_metrics(processed_preds, processed_labels)
        # also add the token-level label evaluation results (seqeval)
        seqeval_results = metric.compute(predictions=true_predictions, references=true_labels)
        if data_args.return_entity_level_metrics:
            # Unpack nested dictionaries
            final_seqeval_results = {}
            for key, value in seqeval_results.items():
                if isinstance(value, dict):
                    for n, v in value.items():
                        final_seqeval_results[f"{key}_{n}"] = v
                else:
                    final_seqeval_results[key] = value
        else:
            final_seqeval_results = {
                "precision": seqeval_results["overall_precision"],
                "recall": seqeval_results["overall_recall"],
                "f1": seqeval_results["overall_f1"],
                "accuracy": seqeval_results["overall_accuracy"],
            }
        # add prefix to the seqeval results and convert the scores to percentage
        final_seqeval_results = {f"seqeval_{k}" : round(v * 100, 4) for k, v in final_seqeval_results.items()}
        # add the seqeval results to the self-defined results
        results.update(final_seqeval_results)
        
        return results
            

    # Initialize our Trainer
    trainer = T5EncoderTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.init_hyper(lr=training_args.learning_rate, lr_cls=training_args.learning_rate_proj)

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model()  # Saves the tokenizer too for easy upload

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        logger.info("*** Predict ***")

        predictions, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix="predict")
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        # Save predictions
        output_predictions_file = os.path.join(training_args.output_dir, "predictions.txt")
        if trainer.is_world_process_zero():
            with open(output_predictions_file, "w") as writer:
                for prediction in true_predictions:
                    writer.write(" ".join(prediction) + "\n")

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "token-classification"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
