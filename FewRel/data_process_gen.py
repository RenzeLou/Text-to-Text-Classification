'''
used for processing fewrel data (generation paradigm)
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
        writer.writerow(["text","category"])
        # all instances
        writer.writerows(ins_list)

def process_fewrel_generation(ins_list:list, id2label:dict=None):
    ins_template = "Sentence: '{sentence}'\n\nThe head entity is '{e_1}' and the tail entity is '{e_2}'.\n\nThe relation between two entities is: "
    processed_list = []
    label_set = set()
    for ins in tqdm(ins_list):
        sentence = ins["sentence"] # token list
        sentence = " ".join(sentence)
        relation = ins["relation"] if id2label is None else id2label[ins["relation"]][0]
        e_1 = ins["head"]["word"]
        e_2 = ins["tail"]["word"]
        text = ins_template.format(sentence=sentence,e_1=e_1,e_2=e_2)
        category = relation.lower().replace(" ","_")  # e.g., "Statement Describes" => "statement_describes"
        processed_list.append((text,category))
        label_set.add(category)
    
    # append instruction for each instance
    instruction = "Given a sentence containing two entities, classify the relation between entities into one of the following categories: {label_set}\n\n"
    instruction = instruction.format(label_set= "[" + ", ".join(label_set) + "]")
    processed_list = [(instruction+text,category) for text,category in processed_list]
    
    return processed_list, label_set
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path",type=str,default='./data/fewrel_ori')
    parser.add_argument("--target_path",type=str,default="./data")
    parser.add_argument("--id2name_path",type=str,default="./data/pid2name.json")
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--label_semantic",action="store_true",default=False, help="whether to use semantic-relavant labels.")
    parser.add_argument("--shuffle",type=bool,default=True)

    args, unparsed = parser.parse_known_args()
    if unparsed:
        raise ValueError(unparsed)

    target_path = args.target_path
    source_path = args.source_path
    seed = args.seed
    
    random.seed(seed)
    np.random.seed(seed)
    
    gen_path = os.path.join(target_path,"gen_id") if not args.label_semantic else os.path.join(target_path,"gen_label")
    os.makedirs(gen_path,exist_ok=True)

    # read ori data
    with open(os.path.join(source_path,"fewrel80_train.json"),"r") as f:
        train_data = json.load(f)
    with open(os.path.join(source_path,"fewrel80_test_test.json"),"r") as f:
        test_data = json.load(f)
    with open(os.path.join(source_path,"fewrel80_test_train.json"),"r") as f:
        dev_data = json.load(f)
    
    # read id2name
    if args.label_semantic:
        with open(args.id2name_path,"r") as f:
            id2name = json.load(f)
    else:
        id2name = None
        
    # process and save data
    print("processing training data...")
    train_data_processed, train_label_set = process_fewrel_generation(train_data,id2name)
    print("processing dev data...")
    dev_data_processed, dev_label_set = process_fewrel_generation(dev_data,id2name)
    print("processing test data...")
    test_data_processed, test_label_set = process_fewrel_generation(test_data,id2name)
    
    if args.shuffle:
        random.shuffle(train_data_processed)
    
    save_t2t_data_csv(train_data_processed,os.path.join(gen_path,"train.csv"))
    save_t2t_data_csv(dev_data_processed,os.path.join(gen_path,"eval.csv"))
    save_t2t_data_csv(test_data_processed,os.path.join(gen_path,"test.csv"))
    
    # save the label set
    with open(os.path.join(gen_path,"train_label_set.json"),"w") as f:
        json.dump(list(train_label_set),f,indent=4)
    with open(os.path.join(gen_path,"eval_label_set.json"),"w") as f:
        json.dump(list(dev_label_set),f,indent=4)
    with open(os.path.join(gen_path,"test_label_set.json"),"w") as f:
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
    
    # save label2id
    train_label2id = {label:idx for idx,label in enumerate(train_label_set)}
    test_label2id = {label:idx for idx,label in enumerate(test_label_set)}
    with open(os.path.join(args.target_path,"train_label2id.json"),"w") as f:
        json.dump(train_label2id,f,indent=4)
    with open(os.path.join(args.target_path,"test_label2id.json"),"w") as f:
        json.dump(test_label2id,f,indent=4)
    
if __name__ == "__main__":
    main()