
gpu=$1
batch=$2
model=$3

export out_dir="out/gen"
export data_dir="/scratch/rml6079/project/GEN_CLS/FewRel/data/gen_label"
export CUDA_VISIBLE_DEVICES=$gpu
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export cache_dir="/scratch/rml6079/.cache"
export TRANSFORMERS_CACHE=${cache_dir}/huggingface
export CUDA_LAUNCH_BLOCKING="0"

# note to add --overwrite_cache \ when doing the final running

export lr=5e-4
export epoch=3

export out_dir="out/gen/${model}"
python run_sen_gen.py \
    --model_name_or_path ${model} \
    --do_train \
    --do_eval \
    --do_predict \
    --train_file ${data_dir}/train.csv \
    --validation_file ${data_dir}/eval.csv \
    --test_file ${data_dir}/test.csv \
    --per_device_train_batch_size ${batch} \
    --per_device_eval_batch_size ${batch} \
    --cache_dir ${cache_dir} \
    --output_dir ./${out_dir}/ \
    --overwrite_output_dir \
    --overwrite_cache \
    --learning_rate ${lr} \
    --num_train_epochs ${epoch} \
    --save_strategy no \
    --evaluation_strategy epoch \
    --seed 42 \
    --source_prefix "" \
    --text_column text \
    --target_column category \
    --predict_with_generate \
    --max_source_length 1024 \
    --max_target_length 128 \
    --train_label2ids /scratch/rml6079/project/GEN_CLS/FewRel/data/train_label2id.json \
    --test_label2ids /scratch/rml6079/project/GEN_CLS/FewRel/data/test_label2id.json \
    --run_name gen_${model}_lr-${lr}_epoch-${epoch}
