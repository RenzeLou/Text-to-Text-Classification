'''
DataCollator is basically a function used for processing the data before feeding it to the model.
Specificaly, in `Trainer` class, it paly a role as `collate_fn` in `torch.utils.data.DataLoader`, which defines how to batch the input data.
therefore, any format of input features can be customized (very flexible and customizable).
that's why the input feature type is List[Dict[str, Any]], and the output is Dict[str, Any]
'''
import random
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from random import randint
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

import numpy as np

from transformers import PreTrainedTokenizerBase, is_tf_available, is_torch_available
from transformers.tokenization_utils_base import PaddingStrategy
from transformers import DataCollatorWithPadding

@dataclass
class DataCollatorBinaryClassification(DataCollatorWithPadding):
    """
    inherits from DataCollatorWithPadding, the only difference is adding "id" and "candidate" to the features, 
    which are fuether used for grouped candidates prediction
    ================================================
    Data collator that will dynamically pad the inputs received.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # note that, each element in features is a dict, which contains the following keys:
        # "input_ids", "attention_mask", "label", "id", "candidate", 'text', 'category'
        # first, pop out "id" and "candidate" from features
        id_list = [ins.pop("id") for ins in features]
        candidate_list = [ins.pop("candidate") for ins in features]
        # then delete "text" and "category" from features
        features = [{k:v for k,v in ins.items() if k not in ["text","category"]} for ins in features]
        
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
        # add "id" and "candidate" back to batch
        batch["ids"] = id_list
        batch["candidates"] = candidate_list
        # print("batch:",batch)
        # exit()
        
        return batch