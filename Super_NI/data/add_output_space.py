import argparse
import copy
import json
import shutil
import jsonlines
import os
import random
import numpy as np
from tqdm import tqdm

def FindAllSuffix(task_path,sufix="json"):
    all_path = os.listdir(task_path)
    result = []
    for p in all_path:
        if not os.path.isdir(p) and sufix in p:
            result.append(os.path.join(task_path,p))
            
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_data_path",type=str,default='./tasks')
    parser.add_argument("--target_save_path",type=str,default="./tasks/add_output_space")
    parser.add_argument("--split_path",type=str,default="./splits/default")
    parser.add_argument("--category_path",type=str,default="./splits/categories_cls_gen.json")
    parser.add_argument("--emperical_max_output_space",type=int,default=100, help="The max output space of a classificaiton task, empirically decided.")
    parser.add_argument("--seed",type=int,default=42)

    args, unparsed = parser.parse_known_args()
    if unparsed:
        raise ValueError(unparsed)
    
    training_data_path = args.training_data_path
    target_save_path = args.target_save_path
    split_path = args.split_path
    category_path = args.category_path
    emperical_max_output_space = args.emperical_max_output_space
    seed = args.seed
    
    random.seed(seed)
    np.random.seed(seed)
    
    os.makedirs(target_save_path,exist_ok=True)

    del_ent = lambda x:x[:-1]
    PREFIX = len(training_data_path) + 1
    all_tasks_num = 0
    max_output_space, max_output_space_tk_name = 0,None
    avg_output_space_num = []
    cls_num, gen_num = 0,0
    gen_num_train, gen_num_test = 0,0
    cls_num_train, cls_num_test = 0,0
    
    with open(category_path,"r") as cp:
        task2cates = json.load(cp)

    with open(split_path+"/train_tasks.txt","r") as sp:
        all_tr_tasks = sp.readlines()
        all_tr_tasks = list(map(del_ent,all_tr_tasks))
        all_tr_tasks_key = dict([(t,1) for t in all_tr_tasks])
        
    with open(split_path+"/test_tasks.txt","r") as sp:
        all_te_tasks = sp.readlines()
        all_te_tasks = list(map(del_ent,all_te_tasks))
        all_te_tasks_key = dict([(t,1) for t in all_te_tasks])
        
    all_tasks_pt = FindAllSuffix(training_data_path,"json")
    for _,tk_pt in enumerate(tqdm(all_tasks_pt)):
        tk_name = tk_pt[PREFIX:len(tk_pt)-5]
        tk_id = tk_name.split("_")[0]
        # only consider the training and testing tasks (except the excluded tasks)
        if all_tr_tasks_key.get(tk_name,0) == 1 or all_te_tasks_key.get(tk_name,0) == 1:
            all_tasks_num += 1
            with open(tk_pt,"r",encoding="utf-8") as tk:
                tk_info = json.load(tk)
                catename = tk_info["Categories"][0] 
                if task2cates[catename] == "GEN":
                    # for general tasks, there is no output space
                    tk_info["output_space"] = []
                    tk_info["output_space_size"] = -1
                    gen_num += 1
                    if all_tr_tasks_key.get(tk_name,0) == 1:
                        gen_num_train += 1
                    else:
                        gen_num_test += 1
                elif task2cates[catename] == "CLS":
                    # for classification tasks, counting all the labels of the instances as the output space
                    all_labels = set()
                    for ins in tk_info["Instances"]:
                        # all_labels = all_labels.union(set(ins["output"]))
                        # ins["output"] can be case sensitive, so we need to convert it to lower case
                        all_labels.update([x.lower() for x in ins["output"]])
                    # double check the output space size
                    if len(all_labels) <= emperical_max_output_space:
                        tk_info["output_space"] = list(all_labels)
                        tk_info["output_space_size"] = len(all_labels)
                        avg_output_space_num.append(len(all_labels))
                        cls_num += 1
                        if all_tr_tasks_key.get(tk_name,0) == 1:
                            cls_num_train += 1
                        else:
                            cls_num_test += 1
                        if len(all_labels) > max_output_space:
                            max_output_space = len(all_labels)
                            max_output_space_tk_name = tk_name
                    else:
                        # if the output space size is larger than the emperical_max_output_space, we regard it as a generation task
                        print("Warning: {} has {} output label space, it has been regarded as a generation task.".format(tk_name, len(all_labels)))
                        tk_info["output_space"] = []
                        tk_info["output_space_size"] = -1
                        gen_num += 1
                        if all_tr_tasks_key.get(tk_name,0) == 1:
                            gen_num_train += 1
                        else:
                            gen_num_test += 1
                else:
                    raise ValueError("Unknown task category: {}".format(task2cates[tk_name]))
                
                with open(os.path.join(target_save_path,tk_name+".json"),"w",encoding="utf-8") as tk_out:
                    json.dump(tk_info,tk_out,ensure_ascii=False, indent=4)
        else:
            # directly copy the excluded tasks
            shutil.copy(tk_pt,os.path.join(target_save_path,tk_name+".json"))
        
    print("Total tasks: {}".format(all_tasks_num))  
    print("Max output space: {} (task: {})".format(max_output_space,max_output_space_tk_name))
    print("Avg output space: {}".format(np.mean(avg_output_space_num)))      
    assert cls_num + gen_num == all_tasks_num
    print("Classification tasks: {}; Generation tasks: {}".format(cls_num,gen_num))
    assert cls_num_train + cls_num_test == cls_num
    assert gen_num_train + gen_num_test == gen_num
    print("==> In Training: Classification tasks: {}; Generation tasks: {}".format(cls_num_train,gen_num_train))
    print("==> In Testing: Classification tasks: {}; Generation tasks: {}".format(cls_num_test,gen_num_test))
    
if __name__ == "__main__":
    main()