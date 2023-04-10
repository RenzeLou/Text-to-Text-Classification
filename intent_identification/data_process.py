'''
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path",type=str,default='./data/banking_data')
    parser.add_argument("--target_path",type=str,default="./data/banking_data")
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--split",action="store_true",default=False)
    parser.add_argument("--shuffle",type=bool,default=True)

    args, unparsed = parser.parse_known_args()
    if unparsed:
        raise ValueError(unparsed)

    label2id = {} # used for cls indices

    target_path = args.target_path
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