from copy import copy, deepcopy
import os
import json
import sys

# append "./src/" to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__),"./src/"))

from src.compute_metrics import compute_metrics


def read_jsonl(file_name):
    result = []
    with open(file_name,"r", encoding="utf-8") as f:    
        for line in f:
            result.append(json.loads(line))
    return result

def read_ref_and_pred(file_name):
    references, predictions = [], []
    with open(file_name,"r", encoding="utf-8") as f:
        for line in f:
            result = json.loads(line)
            pred = result["Prediction"]
            if result.get("GroundTruth",None) is None:
                ref = result["Instance"]["output"]
            else:
                ref = result["GroundTruth"]
            predictions.append(pred)
            references.append(ref)
    assert len(references) == len(predictions), f"len(references) = {len(references)}, len(predictions) = {len(predictions)}"  # impossible
    return references, predictions

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
    parser.add_argument("--cls_eval_path",type=str,default="./output_classifier/",help="The path to the eval results on the classification test tasks.")
    parser.add_argument("--gen_eval_path",type=str,default="./output_generator/",help="The path to the eval results on the generation test tasks.")
    parser.add_argument("--gen_result_folder_name", type=str, default="eval_on_gen", help="The subfolder name of the generation eval results.")
    parser.add_argument("--save_file", type=str, default="overall_predict_results.json")
    
    args, unparsed = parser.parse_known_args()
    if unparsed:
        raise ValueError(unparsed)
    
    # get each folder under cls_eval_path
    all_exps = os.listdir(args.cls_eval_path)
    print(f"all_exps = {all_exps}\n")
    for exp_name in all_exps:
        cls_result_file = os.path.join(args.cls_eval_path,exp_name,"predict_eval_predictions.jsonl")
        gen_result_file = os.path.join(args.gen_eval_path,exp_name,args.gen_result_folder_name,"predict_eval_predictions.jsonl")
        # if both files exist, we can calculate the overall metric
        if os.path.isfile(cls_result_file) and os.path.isfile(gen_result_file):
            # read these two jsonl files
            cls_refs, cls_preds = read_ref_and_pred(cls_result_file)
            gen_refs, gen_preds = read_ref_and_pred(gen_result_file)
            # cls, gen metrics seperately (double check)
            cls_metrics = compute_metrics(cls_preds,cls_refs)
            cls_metrics = {"CLS_"+k:v for k,v in cls_metrics.items()}
            gen_metrics = compute_metrics(gen_preds,gen_refs)
            gen_metrics = {"GEN_"+k:v for k,v in gen_metrics.items()}
            # overall metrics
            all_metrics = compute_metrics(cls_preds+gen_preds,cls_refs+gen_refs)
            all_metrics = {"ALL_"+k:v for k,v in all_metrics.items()}
            # save these metrics
            metrics = {**cls_metrics,**gen_metrics,**all_metrics}
            metrics["test_instance_num"] = len(cls_refs+gen_refs)
            metrics["test_instance_num_cls"] = len(cls_refs)
            metrics["test_instance_num_gen"] = len(gen_refs)
            with open(os.path.join(args.cls_eval_path,exp_name,args.save_file),"w",encoding="utf-8") as f:
                json.dump(metrics,f,indent=2)
            
            print("save metrics to ",os.path.join(args.cls_eval_path,exp_name,args.save_file))

if __name__ == "__main__":
    main()