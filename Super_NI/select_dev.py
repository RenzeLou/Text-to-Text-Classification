'''
randomly select 100 tasks from those exlcuded tasks as the dev set
'''
import os
import random
import json
import multiprocessing
from tqdm import tqdm

def FindAllSuffix(task_path,sufix="json"):
    all_path = os.listdir(task_path)
    result = []
    for p in all_path:
        if not os.path.isdir(p) and sufix in p:
            result.append(os.path.join(task_path,p))
            
    return result

def main():
    # set random seed
    random.seed(42)

    dev_task_num = 100


    task_path = "./data/tasks"
    PREFIX = len(task_path) + 1
    split_path = "./data/splits/add_dev"
    ori_split_path = "./data/splits/default"

    os.makedirs(split_path,exist_ok=True)

    del_ent = lambda x:x[:-1]
    add_ent = lambda x:x+"\n"

    with open(ori_split_path+"/excluded_tasks.txt","r") as sp:
        all_te_tasks = sp.readlines()
        all_te_tasks = list(map(del_ent,all_te_tasks))
        all_te_tasks_key = dict([(t,1) for t in all_te_tasks])

    train_num = 0
    candidate = []
    all_tasks_pt = FindAllSuffix(task_path,"json")
    en_num, non_en_num = 0,0
    for _,tk_pt in tqdm(enumerate(all_tasks_pt)):
        tk_name = tk_pt[PREFIX:len(tk_pt)-5]
        if all_te_tasks_key.get(tk_name,0) == 1:
            train_num += 1
            with open(tk_pt,"r",encoding="utf-8") as tk:
                tk_info = json.load(tk) 
                all_lan = set([tk_info["Input_language"][0], tk_info["Output_language"][0], tk_info["Instruction_language"][0]])
                # we only use those English tasks
                if len(all_lan) == 1 and "English" in all_lan:
                    candidate.append(tk_name)
                    en_num += 1
                else:
                    non_en_num += 1
            
    assert len(all_te_tasks) == train_num    


    dev_tasks = random.sample(candidate,dev_task_num)
    dev_tasks = list(map(add_ent,dev_tasks))
        
    # save the dev tasks
    with open(split_path+"/dev_tasks.txt", "w") as f:
        f.writelines(dev_tasks)
        f.write("")
        
    # copy the training and test tasks from ori_split_path to split_path
    with open(ori_split_path+"/train_tasks.txt","r") as sp:
        train_tasks = sp.readlines()
    with open(split_path+"/train_tasks.txt", "w") as f:
        f.writelines(train_tasks)
        f.write("")
    with open(ori_split_path+"/test_tasks.txt","r") as sp:
        test_tasks = sp.readlines()
    with open(split_path+"/test_tasks.txt", "w") as f:
        f.writelines(test_tasks)
        f.write("")
    # copy the excluded tasks from ori_split_path to split_path
    with open(ori_split_path+"/excluded_tasks.txt","r") as sp:
        excluded_tasks = sp.readlines()
    with open(split_path+"/excluded_tasks.txt", "w") as f:
        f.writelines(excluded_tasks)
        f.write("")

    print("\n" + "="*40 + "\n")
    print("all excluded tasks: ",len(all_te_tasks))
    print(f"en num: {en_num}",f"non_en_num: {non_en_num}",sep="\t")


if __name__ == "__main__":
    # main()
    pool = multiprocessing.Pool(2)
    pool.apply_async(func=main)
    pool.close()
    pool.join()