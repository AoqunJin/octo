export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl

python3 examples/03_eval_metaworld.py --finetuned_path=/path/to/finetuned/model
