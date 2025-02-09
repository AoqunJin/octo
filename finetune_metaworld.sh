export CUDA_VISIBLE_DEVICES=0

# abs path only
python examples/02_finetune_metaworld.py \
    --pretrained_path="/path/to/pretrained/model" \
    --data_dir="/path/to/tensorflow_datasets" \
    --save_dir="/path/to/models/octo-small-1.5" \
    --batch_size=32
