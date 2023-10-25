'''
This script is used to delete the large model files under a given path.
'''
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path",type=str,default="./",help="the target path.")
parser.add_argument("--file_names",type=str,default=None,help="additional files you want to del, split with ','.")

args, unparsed = parser.parse_known_args()
if unparsed:
    raise ValueError(unparsed)

path = args.path  ## modify the path according to your own need

del_file=['pytorch_model-00001-of-00002.bin','pytorch_model-00002-of-00002.bin','training_args.bin','config.json', 'pytorch_model.bin', 'special_tokens_map.json', 'tokenizer_config.json', 'vocab.txt'] ## modify it

if args.file_names is not None:
    addition = args.file_names.split(",")
    print("==> Note you have added the following files: ",addition)
    del_file += addition

for dir_path,dir_name,file_name in os.walk(path):
    for file in file_name:
        if file in del_file:
            os.remove(os.path.join(dir_path,file))