'''
customize a T5 encoder for token classification task
basically a pretrained T5-encoder + a linear layer on the top of each token-level hidden state
'''

import copy
from typing import Optional, Tuple, Union
import torch
from torch import nn

from transformers.models.t5.modeling_t5 import (
    T5PreTrainedModel,
    T5Config,
    T5Stack,
    T5_START_DOCSTRING,
    T5_ENCODER_INPUTS_DOCSTRING,
    PARALLELIZE_DOCSTRING,
    DEPARALLELIZE_DOCSTRING,
    _CONFIG_FOR_DOC
)
from transformers.modeling_outputs import (
    SequenceClassifierOutput,
    TokenClassifierOutput
)
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings
)
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from transformers.utils.logging import get_logger
logger = get_logger("transformers")


class T5EncoderClassificationHead(nn.Module):
    """Head for token-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        classifier_dropout = config.classifier_dropout 
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_projector = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, hidden_states, **kwargs):
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_projector(hidden_states)
        return hidden_states

class T5ConfigForTokenClassification(T5Config):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def get_additional_args(self, num_labels, classifier_dropout=0.2):
        self.num_labels = num_labels
        self.classifier_dropout = classifier_dropout


@add_start_docstrings(
    """T5 Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """, 
    T5_START_DOCSTRING
)
class T5ForTokenClassification(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"encoder.embed_tokens.weight"]

    def __init__(self, config: T5ConfigForTokenClassification):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.is_encoder_decoder = False
        encoder_config.use_cache = False
        self.encoder = T5Stack(encoder_config, self.shared)

        config.classifier_dropout = (
            config.classifier_dropout if hasattr(config, 'classifier_dropout') else config.dropout_rate
        )
        self.classifier = T5EncoderClassificationHead(config)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.classifier.to(self.encoder.first_device)
        self.model_parallel = True

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        self.encoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.classifier = self.classifier.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)

    def get_encoder(self):
        return self.encoder

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.block[layer].layer[0].SelfAttention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(T5_ENCODER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TokenClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.

        Returns:

        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # for token-level classification tasks, such as NER, we need further return back the input_ids
        # to decode the original span to calculate the metrics
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        ), input_ids.detach().cpu().numpy().tolist()

