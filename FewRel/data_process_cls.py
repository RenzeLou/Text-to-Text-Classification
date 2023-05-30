'''
used for processing fewrel data (classification paradigm)
'''

import argparse
import json
import pickle
import os
import csv
import random
import numpy as np
from tqdm import tqdm

def save_nli_data_csv(ins_list:list,path:str):
    assert path.endswith(".csv"), "should be a csv file."
    
    with open(path,"w") as csvfile: 
        writer = csv.writer(csvfile)

        # columns name
        writer.writerow(["sentence1","sentence2","label"])
        # all instances
        writer.writerows(ins_list)

def save_t2t_data_csv(ins_list:list,path:str):
    assert path.endswith(".csv"), "should be a csv file."
    
    with open(path,"w") as csvfile: 
        writer = csv.writer(csvfile)

        # columns name
        writer.writerow(["text","category","id","candidate"])
        # all instances
        writer.writerows(ins_list)

def process_fewrel_classification(ins_list:list, id2label:dict=None, is_train:bool=False):
    ins_template = "Sentence: '{sentence}'\n\nThe head entity is '{e_1}' and the tail entity is '{e_2}'.\n\nThe relation between two entities is '{category}' ?"
    
    label_set = set()
    for ins in tqdm(ins_list):
        relation = ins["relation"] if id2label is None else id2label[ins["relation"]][0]
        category = relation.lower().replace(" ","_")  # e.g., "Statement Describes" => "statement_describes"
        label_set.add(category)
    
    processed_list = []
    for ins_id, ins in tqdm(enumerate(ins_list)):
        sentence = ins["sentence"] # token list
        sentence = " ".join(sentence)
        relation = ins["relation"] if id2label is None else id2label[ins["relation"]][0]
        e_1 = ins["head"]["word"]
        e_2 = ins["tail"]["word"]
        category = relation.lower().replace(" ","_")  # e.g., "Statement Describes" => "statement_describes"
        text = ins_template.format(sentence=sentence,e_1=e_1,e_2=e_2,category=category)
        # positive instance
        processed_list.append((text,1,"instance_"+str(ins_id)+"_candidate_0",category))
        # negative instance
        if is_train:
            # for training, to balance the positive and negative instances, there only one randomly selected negative instance for each positive instance
            neg_category = random.choice(list(label_set - set([category])))
            neg_text = ins_template.format(sentence=sentence,e_1=e_1,e_2=e_2,category=neg_category)
            processed_list.append((neg_text,0, "instance_"+str(ins_id)+"_candidate_1",neg_category))
        else:
            # for evaluation, all negative instances are included
            neg_categories = list(label_set - set([category]))
            for cate_id, neg_category in enumerate(neg_categories):
                neg_text = ins_template.format(sentence=sentence,e_1=e_1,e_2=e_2,category=neg_category)
                processed_list.append((neg_text,0, "instance_"+str(ins_id)+"_candidate_"+str(cate_id+1),neg_category))
    
    return processed_list, label_set
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path",type=str,default='./data/fewrel_ori')
    parser.add_argument("--target_path",type=str,default="./data")
    parser.add_argument("--id2name_path",type=str,default="./data/pid2name.json")
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--shuffle",type=bool,default=True)

    args, unparsed = parser.parse_known_args()
    if unparsed:
        raise ValueError(unparsed)

    target_path = args.target_path
    source_path = args.source_path
    seed = args.seed
    
    random.seed(seed)
    np.random.seed(seed)
    
    cls_path = os.path.join(target_path,"cls")
    os.makedirs(cls_path,exist_ok=True)

    # read ori data
    with open(os.path.join(source_path,"fewrel80_train.json"),"r") as f:
        train_data = json.load(f)
    with open(os.path.join(source_path,"fewrel80_test_test.json"),"r") as f:
        test_data = json.load(f)
    with open(os.path.join(source_path,"fewrel80_test_train.json"),"r") as f:
        dev_data = json.load(f)
    
    # read id2name
    with open(args.id2name_path,"r") as f:
        id2name = json.load(f)
        
    # process and save data
    print("processing training data...")
    train_data_processed, train_label_set = process_fewrel_classification(train_data,id2name,is_train=True)
    print("processing dev data...")
    dev_data_processed, dev_label_set = process_fewrel_classification(dev_data,id2name,is_train=False)
    print("processing test data...")
    test_data_processed, test_label_set = process_fewrel_classification(test_data,id2name,is_train=False)
    
    if args.shuffle:
        random.shuffle(train_data_processed)
    
    save_t2t_data_csv(train_data_processed,os.path.join(cls_path,"train.csv"))
    save_t2t_data_csv(dev_data_processed,os.path.join(cls_path,"eval.csv"))
    save_t2t_data_csv(test_data_processed,os.path.join(cls_path,"test.csv"))
    
    # save the label set
    with open(os.path.join(cls_path,"train_label_set.json"),"w") as f:
        json.dump(list(train_label_set),f,indent=4)
    with open(os.path.join(cls_path,"eval_label_set.json"),"w") as f:
        json.dump(list(dev_label_set),f,indent=4)
    with open(os.path.join(cls_path,"test_label_set.json"),"w") as f:
        json.dump(list(test_label_set),f,indent=4)
        
    
    print("==> {} training instances, {} dev instances, {} test instances saved.".format(len(train_data_processed),len(dev_data_processed),len(test_data_processed)))
    print("==> {} training labels, {} dev labels, {} test labels saved.".format(len(train_label_set),len(dev_label_set),len(test_label_set)))
    
    # get the overlapped label set between training and test set
    overlap_set_1 = train_label_set.intersection(test_label_set)
    print("{} overlappped categories between training and test set.".format(len(overlap_set_1)))
    # get the overlapped label set between training and dev set
    overlap_set_2 = train_label_set.intersection(dev_label_set)
    print("{} overlappped categories between training and dev set.".format(len(overlap_set_2)))
    # get the overlapped label set between dev and test set
    overlap_set_3 = dev_label_set.intersection(test_label_set)
    print("{} overlappped categories between dev and test set.".format(len(overlap_set_3)))
    
    
if __name__ == "__main__":
    main()