# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
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

# Lint as: python3
"""Natural Instruction V2 Dataset."""


import json
import os
import random
import datasets

from compute_metrics import normalize_answer

logger = datasets.logging.get_logger(__name__)

_CITATION = """
@article{wang2022benchmarking,
  title={Benchmarking Generalization via In-Context Instructions on 1,600+ Language Tasks},
  author={Wang, Yizhong and Mishra, Swaroop and Alipoormolabashi, Pegah and Kordi, Yeganeh and others},
  journal={arXiv preprint arXiv:2204.07705},
  year={2022}
}
"""

_DESCRIPTION = """
Natural-Instructions v2 is a benchmark of 1,600+ diverse language tasks and their expert-written instructions. 
It covers 70+ distinct task types, such as tagging, in-filling, and rewriting. 
These tasks are collected with contributions of NLP practitioners in the community and 
through an iterative peer review process to ensure their quality. 
"""

_URL = "https://instructions.apps.allenai.org/"

class NIConfig(datasets.BuilderConfig):
    def __init__(self, *args, task_dir=None, max_num_instances_per_task=None, max_num_instances_per_eval_task=None, train_mix_gen=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_dir: str = task_dir
        self.max_num_instances_per_task: int = max_num_instances_per_task
        self.max_num_instances_per_eval_task: int = max_num_instances_per_eval_task
        self.train_mix_gen: bool = train_mix_gen


class NaturalInstructions(datasets.GeneratorBasedBuilder):
    """NaturalInstructions Dataset."""

    VERSION = datasets.Version("2.0.0")
    BUILDER_CONFIG_CLASS = NIConfig
    BUILDER_CONFIGS = [
        NIConfig(name="default", description="Default config for NaturalInstructions")
    ]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "Task": datasets.Value("string"),
                    "Contributors": datasets.Value("string"),
                    "Source": [datasets.Value("string")],
                    "URL": [datasets.Value("string")],
                    "Categories": [datasets.Value("string")],
                    "Reasoning": [datasets.Value("string")],
                    "Definition": [datasets.Value("string")],
                    "Positive Examples": [{
                        "input": datasets.Value("string"),
                        "output": datasets.Value("string"),
                        "explanation": datasets.Value("string")
                    }],
                    "Negative Examples": [{
                        "input": datasets.Value("string"),
                        "output": datasets.Value("string"),
                        "explanation": datasets.Value("string")
                    }],
                    "Input_language": [datasets.Value("string")],
                    "Output_language": [datasets.Value("string")],
                    "Instruction_language": [datasets.Value("string")],
                    "Domains": [datasets.Value("string")],
                    # "Instances": [{
                    #     "input": datasets.Value("string"),
                    #     "output": [datasets.Value("string")]
                    # }],
                    "Instance": {
                        "id": datasets.Value("string"),
                        "input": datasets.Value("string"),
                        # "output": [datasets.Value("string")]
                        "candidate_output": datasets.Value("string"), ## one candidate from the output space
                        "label": datasets.Value("string"),  ## 0: negative, 1: positive
                    },
                    "Instance License": [datasets.Value("string")]
                }
            ),
            supervised_keys=None,
            homepage="https://github.com/allenai/natural-instructions",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        if self.config.data_dir is None or self.config.task_dir is None:
            dl_path = dl_manager.download_and_extract(_URL)
            self.config.data_dir = self.config.data_dir or os.path.join(dl_path, "splits")
            self.config.task_dir = self.config.task_dir or os.path.join(dl_path, "tasks")

        split_dir = self.config.data_dir
        task_dir = self.config.task_dir
        train_mix_gen = self.config.train_mix_gen

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "path": os.path.join(split_dir, "train_tasks.txt"), 
                    "task_dir": task_dir, 
                    "max_num_instances_per_task": self.config.max_num_instances_per_task,
                    "subset": "train",
                    "mix_gen": train_mix_gen
                }),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "path": os.path.join(split_dir, "dev_tasks.txt"), 
                    "task_dir": task_dir,
                    "max_num_instances_per_task": self.config.max_num_instances_per_eval_task,
                    "subset": "dev",
                    "mix_gen": False
                }),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "path": os.path.join(split_dir, "test_tasks.txt"), 
                    "task_dir": task_dir, 
                    "max_num_instances_per_task": self.config.max_num_instances_per_eval_task,
                    "subset": "test",
                    "mix_gen": False
                }),
        ]
        
    # def is_gen_task(self, category):
    #     '''
    #     category: str, e.g., "Question Answering"
    #     '''
    #     ## metric list
    #     EM_list = ["TE","CEC","CR","DAR","AC","WA"]
    #     Rouge_list = ["OE","KT","QR","TG","DT","GEC"]
    #     ## abv dic
    #     name_abv = {"answerability_classification":"AC"		
    #     ,"cause_effect_classification": "CEC"		
    #     ,"coreference_resolution": "CR"		
    #     ,"data_to_text": "DT"		
    #     ,"dialogue_act_recognition": "DAR"		
    #     ,"grammar_error_correction": "GEC"		
    #     ,"keyword_tagging": "KT"		
    #     ,"overlap_extraction": "OE"		
    #     ,"question_rewriting": "QR"		
    #     ,"textual_entailment": "TE"		
    #     ,"title_generation": "TG"		
    #     ,"word_analogy": "WA"}
        
    #     cate_name = "_".join(category.lower().split())
    #     if name_abv[cate_name] in EM_list:
    #         return False
    #     elif name_abv[cate_name] in Rouge_list:
    #         return True

    def is_gen_task(self, task_data):
        # output_space_size == -1 means there are numerous output spaces, we regard it as a generation tasks
        if task_data["output_space_size"] == -1:
            return True
        else:
            return False

    def _generate_examples(self, path=None, task_dir=None, max_num_instances_per_task=None, subset=None, mix_gen=False):
        """Yields examples."""
        logger.info(f"Generating tasks from = {path}")
        pos_ins_num, neg_ins_num = 0, 0
        with open(path, encoding="utf-8") as split_f:
            for line in split_f:
                task_name = line.strip()
                task_path = os.path.join(task_dir, task_name + ".json")
                with open(task_path, encoding="utf-8") as task_f:
                    s = task_f.read()
                    task_data = json.loads(s)
                    task_data["Task"] = task_name
                    if "Instruction Source" in task_data:
                        task_data.pop("Instruction Source")
                    all_instances = task_data.pop("Instances")
                    if subset == "test":
                        # for testing tasks, 100 instances are selected for efficient evaluation and they are label-balanced.
                        # we put them in the first for reproducibility.
                        # so, we use them here
                        instances = all_instances[:100]
                    else:
                        instances = all_instances
                    if max_num_instances_per_task is not None and max_num_instances_per_task >= 0:
                        random.shuffle(instances)
                        instances = instances[:max_num_instances_per_task]
                    
                    # we only evaluate on the cls tasks
                    # so, if it is a generation task, skip
                    if not mix_gen and self.is_gen_task(task_data):
                        continue
                    
                    if "output_space" in task_data:
                        output_space = task_data.pop("output_space")
                        output_space_size = task_data.pop("output_space_size")
                    
                    for idx, instance in enumerate(instances):
                        if output_space_size == -1:
                            # for gen tasks, we directly regard the groud truth as the only candidate output, so the labels are all 1
                            ground_truth_outputs = instance.pop("output")
                            # random select one ground truth output
                            ground_truth_output = random.choice(ground_truth_outputs)
                            new_instance = instance.copy()
                            new_instance["candidate_output"] = ground_truth_output
                            new_instance["label"] = "1"
                            new_instance["id"] = instance["id"] + "_gen_ground_truth_0"
                            
                            example = task_data.copy()
                            example["id"] = new_instance["id"]
                            example["Instance"] = new_instance
                            
                            pos_ins_num += 1
                            
                            yield f"{task_name}_{idx}", example
                        else:
                            # for cls tasks, we combine each candidate output with the instance, the labels are 0 or 1
                            ground_truth_outputs = instance.pop("output")
                            assert isinstance(ground_truth_outputs, list)
                            for candidate_idx, candidate in enumerate(output_space):
                                new_instance = instance.copy()
                                assert isinstance(candidate, str)
                                new_instance["candidate_output"] = candidate
                                # if this candidate is the ground truth, label is 1 (positive example), otherwise 0 (negative example)
                                # note that ground_truth_outputs is case sensitive
                                if normalize_answer(candidate) in [normalize_answer(x) for x in ground_truth_outputs]:
                                    new_instance["label"] = "1"
                                else:
                                    new_instance["label"] = "0"
                                # new_instance["label"] = "1" if candidate in ground_truth_outputs else "0"  
                                new_instance["id"] = instance["id"] + f"_candidate_{candidate_idx}"
                                
                                example = task_data.copy()
                                example["id"] = new_instance["id"]
                                example["Instance"] = new_instance
                                
                                if new_instance["label"] == "1":
                                    pos_ins_num += 1
                                else:
                                    neg_ins_num += 1
                        
                                yield f"{task_name}_{idx}_{candidate_idx}", example
                                
        print(f"==> For {subset}: pos_ins_num = {pos_ins_num}, neg_ins_num = {neg_ins_num}")
        
