TIME=$(date +%s%3N)
DATA_PATH='/home/h_haoy/Documents/GitHub/test_time_training_mae/data'
OUTPUT_DIR='/home/h_haoy/Documents/GitHub/test_time_training_mae'
python -m torch.distributed.launch --nproc_per_node=1 main_pretrain.py \
        --model mae_vit_large_patch16 \
        --input_size 224 \
        --batch_size 8 \
        --mask_ratio 0.75 \
        --warmup_epochs 40 \
        --epochs 400 \
        --blr 1e-3 \
        --save_ckpt_freq 100 \
        --output_dir ${OUTPUT_DIR}  \
        --dist_url "file://$OUTPUT_DIR/$TIME"
