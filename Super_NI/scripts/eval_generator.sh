#!/bin/bash
gpu=$1
batch=$2
model=$3  # model name, like t5-base
train_mix_gen=$4

set -x

echo "export CUDA_VISIBLE_DEVICES=$gpu"

export CUDA_VISIBLE_DEVICES=${gpu}
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=/scratch/rml6079/.cache/huggingface
export CUDA_LAUNCH_BLOCKING="1"

port=$(shuf -i25000-30000 -n1)

# convert train_mix_gen to boolean
if [ "$train_mix_gen" -eq 1 ]; then
  train_mix_gen=true
elif [ "$train_mix_gen" -eq 0 ]; then
  train_mix_gen=false
else
  echo "Invalid train_mix_gen, set to False as default"
  train_mix_gen=false
fi

data_dir=data/splits/default
task_dir=data/tasks/add_output_space
model_dir=output_generator/${model}-mix_gen_${train_mix_gen}
output_dir=output_generator/${model}-mix_gen_${train_mix_gen}/eval_on_gen
Tk_instruct_cache_dir=/scratch/rml6079/project/Tk-Instruct/cache/

python src/run_s2s_eval_on_gen.py \
    --do_predict \
    --predict_with_generate \
    --model_name_or_path ${model_dir} \
    --max_source_length 1024 \
    --max_target_length 128 \
    --generation_max_length 128 \
    --max_num_instances_per_task 100 \
    --max_num_instances_per_eval_task 100 \
    --add_task_name False \
    --add_task_definition True \
    --num_pos_examples 0 \
    --num_neg_examples 0 \
    --add_explanation False \
    --tk_instruct False \
    --data_dir ${data_dir} \
    --task_dir ${task_dir} \
    --output_dir ${output_dir} \
    --overwrite_output_dir \
    --cache_dir ${Tk_instruct_cache_dir} \
    --overwrite_cache \
    --per_device_eval_batch_size ${batch} \
    --run_name train_generator-mix_gen_${train_mix_gen}_eval \
    --seed 42 
