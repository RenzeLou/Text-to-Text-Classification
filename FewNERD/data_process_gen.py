'''
process the conll2003 NER data into the generation format

============================================================
there are two types of data format for NER task:

(1). a popular NER prompt (vanilla), see https://arxiv.org/pdf/2106.01223.pdf
## input: 
Jim bought 300 shares of Acme Corp. in 2006.
## output: 
Jim \t <person> \n 
Acme Corp. \t <organization> \n
2006 \t <time> \n 

(2). a straightforward NER prompt
## input:
Jim bought 300 shares of Acme Corp. in 2006.
## output: 
Jim <person> bought 300 shares of Acme Corp. <organization> in 2006. <time>

currently use (1) because this prediction result is easier to evaluate and is more difficult; while (2) doesn't require the entity span to be exactly the same as the ground truth

============================================================
by training text-to-text model, the model will be optimized to 
(1) extract entities
(2) classify entities
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

def read_txt_file(file_path:str):
    ''' return a list of (text,label) '''
    with open(file_path,"r") as f:
        lines = f.readlines()
    
    all_instances = []
    per_ins = []
    for i, line in enumerate(lines):
        # if i == 0:
        #     continue
        if line != "\n":
            line_data = line.strip().split("\t")
            per_ins.append((line_data[0],line_data[-1]))
        elif len(per_ins) > 0:
            all_instances.append(per_ins)
            per_ins = []
    
    return all_instances

def process_ner_data(ins_list:list):
    target_ins_list = []
    for ins in tqdm(ins_list):
        # combine all tokens into a sentence
        source_sequence = " ".join([token[0] for token in ins])
        target_sequence = ""
        # find all named entities (continuous spans) in this instance
        # e.g., [("am",O), ("lou",PER), ("ren",PER), ("ze",PER), ("at",O), ("Penn",LOC), ("State",LOC), ("!",O)] ==> lou ren ze \t PER \n Penn State \t LOC \n
        i = 0
        while i < len(ins):
            token = ins[i]
            if token[1] != "O":
                current_label = token[1]
                start = i
                end = i
                while end < len(ins) - 1 and ins[end+1][1] == current_label:
                    end += 1
                entity = "[" + current_label + "]"  # do not forget to add the brackets, e.g., "[PER]"
                target_sequence += " ".join([token[0] for token in ins[start:end+1]]) + "\t" + entity + "\n"
                i = end + 1
            else:
                i += 1
        target_sequence = target_sequence.strip()
        target_ins_list.append((source_sequence,target_sequence))
    
    # print("data: ", target_ins_list[:5])
    # exit()
    return target_ins_list
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path",type=str,default='./data')
    parser.add_argument("--setting",type=str,default="supervised", choices=["supervised","inter","intra"])
    parser.add_argument("--target_path",type=str,default="./data")
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--shuffle",type=bool,default=True)

    args, unparsed = parser.parse_known_args()
    if unparsed:
        raise ValueError(unparsed)

    target_path = os.path.join(args.target_path, args.setting, "gen")
    source_path = os.path.join(args.source_path, args.setting)
    seed = args.seed
    
    random.seed(seed)
    np.random.seed(seed)
    
    os.makedirs(target_path,exist_ok=True)

    # read txt files
    train_data = read_txt_file(os.path.join(source_path,"train.txt"))
    # print("train data: ",train_data[:2])
    # print(len(train_data))
    # exit()
    dev_data = read_txt_file(os.path.join(source_path,"dev.txt"))
    test_data = read_txt_file(os.path.join(source_path,"test.txt"))
    
    # process and save the ner data
    train_data_processed = process_ner_data(train_data)
    dev_data_processed = process_ner_data(dev_data)
    test_data_processed = process_ner_data(test_data)
    
    if args.shuffle:
        random.shuffle(train_data_processed)
    
    save_t2t_data_csv(train_data_processed,os.path.join(target_path,"train.csv"))
    save_t2t_data_csv(dev_data_processed,os.path.join(target_path,"eval.csv"))
    save_t2t_data_csv(test_data_processed,os.path.join(target_path,"test.csv"))
    
    print("==> Training: {}, Validation: {}, Test: {}".format(len(train_data_processed),len(dev_data_processed),len(test_data_processed)))

if __name__ == "__main__":
    main()