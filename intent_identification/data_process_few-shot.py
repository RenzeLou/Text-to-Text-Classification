'''
randomly select few-shot examples for each class
making the intent classification task more challenging
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
        
def group_by_label(train_samples:list):
    '''
    group instances by label
    '''
    # categorize training samples according to the labels
    grouped_train_samples = {}
    for ins in train_samples:
        text, label = ins[0], ins[1]
        if grouped_train_samples.get(label,None) is None:
            grouped_train_samples[label] = [text]
        else:
            grouped_train_samples[label].append(text)
            
    return grouped_train_samples

def merge_results(grouped_train_samples:dict):
    train_samples_res = []
    for label, samples in grouped_train_samples.items():
        for sample in samples:
            train_samples_res.append((sample,label))
    return train_samples_res

def split_few_shot(train_samples:list,test_samples:list,shot:int):
    '''
    randomly select few-shot examples for each class
    move the remaining from training set to test set
    '''
    
    # randomly select few-shot examples for each class, move the remaining from training set to test set
    grouped_train_samples = group_by_label(train_samples)
    grouped_test_samples = group_by_label(test_samples)
    assert len(grouped_train_samples) == len(grouped_test_samples), "assert the number of classes in training set and test set should be the same."
    for label, samples in grouped_train_samples.items():
        random.shuffle(samples)
        grouped_train_samples[label] = samples[:shot]
        grouped_test_samples[label] = grouped_test_samples[label] + samples[shot:]
    # merge (label,sample) pairs
    train_samples_res = merge_results(grouped_train_samples)
    test_samples_res = merge_results(grouped_test_samples)
    
    return train_samples_res, test_samples_res

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path",type=str,default='./data/banking_data')
    parser.add_argument("--target_path",type=str,default="./data/banking_data")
    parser.add_argument("--shot", type=int, default=3)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--split",action="store_true",default=False)
    parser.add_argument("--shuffle",type=bool,default=True)

    args, unparsed = parser.parse_known_args()
    if unparsed:
        raise ValueError(unparsed)

    label2id = {} # used for cls indices

    target_path = os.path.join(args.target_path, f"{args.shot}-shot")
    source_path = args.source_path
    seed = args.seed
    
    random.seed(seed)
    np.random.seed(seed)
    
    gen_path = os.path.join(target_path,"gen")
    cls_path = os.path.join(target_path,"cls")
    os.makedirs(gen_path,exist_ok=True)
    os.makedirs(cls_path,exist_ok=True)

    # open csv file
    with open(os.path.join(source_path,"train.csv"),"r") as f:
        reader = csv.reader(f)
        train_data = list(reader)
    with open(os.path.join(source_path,"test.csv"),"r") as f:
        reader = csv.reader(f)
        test_data = list(reader)
    
    # remove head
    train_data = train_data[1:]
    test_data = test_data[1:]
    
    # for training/eval data
    train_samples = []
    for i, ins in tqdm(enumerate(train_data)):
        text, label = ins[0], ins[1]
        if label2id.get(label,None) is None:
            label2id[label] = len(label2id)
        train_samples.append((text,label2id[label]))
    
    if args.shuffle:
        combined = list(zip(train_samples, train_data))
        # Shuffle the combined list
        random.shuffle(combined)
        # Unzip the result back into separate lists
        train_samples, train_data = zip(*combined)
    
    # split 0.1 for eval
    if args.split:
        train_samples, eval_samples = train_samples[:-int(len(train_samples)*0.1)], train_samples[-int(len(train_samples)*0.1):]
        train_data, eval_data = train_data[:-int(len(train_data)*0.1)], train_data[-int(len(train_data)*0.1):]
    else:
        train_samples, eval_samples = train_samples, []
        train_data, eval_data = train_data, []
    
    # for test data
    test_samples = []
    for i, ins in tqdm(enumerate(test_data)):
        text, label = ins[0], ins[1]
        assert label2id.get(label,None) is not None, "assert labels in testing file should all be in training file."
        test_samples.append((text,label2id[label]))
    
    train_samples, test_samples = split_few_shot(train_samples,test_samples,args.shot)
    train_data, test_data = split_few_shot(train_data,test_data,args.shot)
    
    # count the unique labels number in training and testing set (for double check)
    # print("==> cls train labels: {}\t test labels: {}".format(len(set(dict(train_samples).values())),len(set(dict(test_samples).values()))))
    # print("==> gen train labels: {}\t test labels: {}".format(len(set(dict(train_data).values())),len(set(dict(test_data).values()))))
    # exit()
    
    print("==> for generation")
    print("train samples: {}, eval samples: {}, test samples: {}".format(len(train_data),len(eval_data),len(test_data)))
    save_t2t_data_csv(train_data,os.path.join(gen_path,"train.csv"))
    save_t2t_data_csv(eval_data,os.path.join(gen_path,"eval.csv"))
    save_t2t_data_csv(test_data,os.path.join(gen_path,"test.csv"))
    
    print("==> for classification")
    print("train samples: {}, eval samples: {}, test samples: {}".format(len(train_samples),len(eval_samples),len(test_samples)))
    save_t2t_data_csv(train_samples,os.path.join(cls_path,"train.csv"))
    save_t2t_data_csv(eval_samples,os.path.join(cls_path,"eval.csv"))
    save_t2t_data_csv(test_samples,os.path.join(cls_path,"test.csv"))
    
    with open(os.path.join(cls_path,"categories2id.json"),"w") as f:
        json.dump(label2id,f)

if __name__ == "__main__":
    main()