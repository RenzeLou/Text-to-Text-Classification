'''
customize a T5 encoder for sequence classification task
basically a pretrained T5-encoder + a linear layer on the top of <s> token's hidden state
'''

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.t5.modeling_t5 import T5EncoderModel, T5Config

# class T5EncoderClassificationHead(nn.Module):
#     """Head for sentence-level classification tasks."""

#     def __init__(self, config):
#         super().__init__()
#         self.dense_projector = nn.Linear(config.hidden_size, config.hidden_size)
#         classifier_dropout = config.classifier_dropout 
#         self.dropout = nn.Dropout(classifier_dropout)
#         self.out_projector = nn.Linear(config.hidden_size, config.num_labels)

#     def forward(self, hidden_states, **kwargs):
#         hidden_states = hidden_states[:, 0, :]  # take <s> token (equiv. to [CLS]) # 90.1948
#         hidden_states = self.dropout(hidden_states)
#         hidden_states = self.dense_projector(hidden_states)
#         hidden_states = torch.tanh(hidden_states)
#         hidden_states = self.dropout(hidden_states)
#         hidden_states = self.out_projector(hidden_states)
#         return hidden_states

class T5EncoderClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        classifier_dropout = config.classifier_dropout 
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_projector = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, hidden_states, **kwargs):
        hidden_states = hidden_states[:, 0, :]  # take <s> token (equiv. to [CLS])
        # hidden_states = torch.mean(hidden_states, dim=1)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_projector(hidden_states)
        return hidden_states


# class T5EncoderClassificationHead(nn.Module):
#     """Head for sentence-level classification tasks."""

#     def __init__(self, config):
#         super().__init__()
#         '''
#         pooler: maxpooling
#         dense_1: (hidden_size, hidden_size)
#         activation: tanh
#         dense_2: (hidden_size, hidden_size)
#         dropout
#         output: (hidden_size, num_labels)
#         '''
#         self.dense_projector = nn.Linear(config.hidden_size, config.hidden_size)
#         self.dense2_projector = nn.Linear(config.hidden_size, config.hidden_size)
#         classifier_dropout = config.classifier_dropout 
#         self.dropout = nn.Dropout(classifier_dropout)
#         self.out_projector = nn.Linear(config.hidden_size, config.num_labels)

#     def forward(self, hidden_states, **kwargs):
#         hidden_states = hidden_states[:, 0, :]  # take <s> token (equiv. to [CLS]) # 89.2857
#         # hidden_states = torch.max(hidden_states, dim=1)[0] # 22, so bad 
#         # hidden_states = torch.mean(hidden_states, dim=1)  # 88.7338
#         # hidden_states = self.dropout(hidden_states)
#         hidden_states = self.dense_projector(hidden_states)
#         hidden_states = torch.tanh(hidden_states)
#         hidden_states = self.dense2_projector(hidden_states)
#         hidden_states = self.dropout(hidden_states)
#         hidden_states = self.out_projector(hidden_states)
#         return hidden_states


class T5ConfigForSequenceClassification(T5Config):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def get_additional_args(self, num_labels, problem_type="single_label_classification", classifier_dropout=0.2):
        self.num_labels = num_labels
        self.problem_type = problem_type
        self.classifier_dropout = classifier_dropout
        

class T5EncoderForSequenceClassification(T5EncoderModel):
    """
    Use an in-memory T5Encoder to do sequence classification"""
    def __init__(self, config: T5ConfigForSequenceClassification):
        super().__init__(config)
        
        self.num_labels = config.num_labels
        self.classifier = T5EncoderClassificationHead(config)
        assert self.encoder is not None, "T5EncoderForSequenceClassification requires an encoder"
        
    def freeze_encoder(self, signal:bool):
        for param in self.encoder.parameters():
            param.requires_grad = signal
            
    def freeze_projector(self, signal:bool):
        for param in self.classifier.parameters():
            param.requires_grad = signal

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_hidden_states=None,
        output_attentions=None,
        return_dict=None,
        ids=None, # list of str
        candidates=None, # list of str
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        id: used for grouping all candidates of the same instance together
        candidates: the candidate outputs of current batch of instances
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        # if not return_dict:
        #     output = (logits,) + encoder_outputs[2:]
        #     return ((loss,) + output) if loss is not None else output
        assert return_dict, "return_dict must be True"

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        ), ids, candidates