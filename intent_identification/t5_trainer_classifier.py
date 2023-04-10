import torch.nn as nn

from transformers import Seq2SeqTrainer, Trainer
from transformers.trainer import *


class T5EncoderTrainer(Trainer):
    def init_hyper(self, lr, lr_cls):
        self.lr = lr
        self.lr_proj = lr_cls
        
    
    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if self.optimizer is None:
            decay_parameters = get_parameter_names(self.model, [nn.LayerNorm])
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if n in decay_parameters and "_projector" not in n],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters and "_projector" not in n],
                    "weight_decay": 0.0,
                },
            ]
            optimizer_grouped_parameters += [
                {
                    "params": [p for n, p in self.model.named_parameters() if n in decay_parameters and "_projector" in n],
                    "lr": self.lr_proj,
                    "weight_decay": self.args.weight_decay
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters and "_projector" in n],
                    "lr": self.lr_proj,
                    "weight_decay": 0.0
                }
            ]

            print([n for n, p in self.model.named_parameters() if "projector" in n])  # for debug
            # print(optimizer_grouped_parameters)
            # exit()
            
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer