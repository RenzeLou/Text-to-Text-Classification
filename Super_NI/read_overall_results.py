from copy import copy, deepcopy
import os
import json
from turtle import color
from unittest import result
import numpy as np
import matplotlib.pyplot as plt
from regex import R


def read_exp_results(file_name:str):
    with open(file_name,"r",encoding="utf-8") as f:
        data = json.load(f)
    # em = data['eval_exact_match']
    # rg = data['eval_rougeL']
    em = data['CLS_exact_match']
    rg = data['GEN_rougeL']
    all_rg = data["ALL_rougeL"]
    
    return em,rg,all_rg

def FindAllSuffix(task_path,sufix="json"):
    all_path = os.listdir(task_path)
    all_path = [os.path.join(task_path,p) for p in all_path]
    result = []
    for p in all_path:
        file_name = os.path.join(p,sufix)
        if os.path.isdir(p) and os.path.isfile(file_name):
            result.append(file_name)
            
    return result

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--path",type=str,default="./output/",help="the output path.")
    parser.add_argument("--results_file_name", type=str, default="overall_predict_results.json", help="the output file name.")
    
    args, unparsed = parser.parse_known_args()
    if unparsed:
        raise ValueError(unparsed)
    
    results = dict()
    # all_results_file = FindAllSuffix(args.path,"eval_results.json")
    all_results_file = FindAllSuffix(args.path,args.results_file_name)

    for file_name in all_results_file:
        try:
            em,rg,all_rg = read_exp_results(file_name) 
            # em_rg_avg = np.mean([em,rg])
            exp_name = file_name.rsplit("/")[-2]
            # results[exp_name] = [em,rg,float(f"{em_rg_avg:.4f}")]
            results[exp_name] = [em,rg,all_rg]
        except FileNotFoundError:
            print("warn: the exp '{}' is not ready!".format(exp_name))
            continue
    key = lambda x:x[1][1]  # based on the RougeL
    results = dict(sorted(results.items(),key=key,reverse=True))
    
    print("\nTotally {} EXP results under '{}':".format(len(results),args.path))
    print("EXP Name\t[EM (CLS), RougeL (GEN), RougeL (ALL)]")
    for k,v in results.items():
        print("{}:\t{}".format(k,v))
    
if __name__ == "__main__":
    main()