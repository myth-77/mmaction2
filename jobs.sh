
# # train with original lr for 8GPUs on 2GPUs
# ./tools/dist_train.sh configs/recognition/tanet/tanet_r50_1x1x16_50e_hmdb51_rgb.py 2 \
# --validate --seed 0 --deterministic \
# --work-dir ./exp/tanet_r50_1x1x16_50e_hmdb51s1_rgb


# # # train with larger lr for 8GPUs on 2GPUs
# ./tools/dist_train.sh configs/recognition/timesformer/timesformer_divST_8x32x1_15e_hmdb51s1_rgb_L.py 2 \
# --validate --seed 0 --deterministic \
# --work-dir ./exp/timesformer_divST_8x32x1_15e_hmdb51s1_rgb_L

# train with smaller lr for 8GPUs on 2GPUs
# CUDA_VISIBLE_DEVICES=1 ./tools/dist_train.sh configs/recognition/timesformer/timesformer_divST_8x32x1_15e_hmdb51s1_rgb_finetune.py 1 \
# --validate --seed 0 --deterministic \
# --work-dir ./exp/timesformer_divST_8x32x1_15e_hmdb51s1_rgb_1e4_finetunek400_adaw

# ./tools/dist_train.sh configs/recognition/timesformer/timesformer_divST_8x32x1_15e_hmdb51s1_rgb_finetune2.py 2 \
# --validate --seed 0 --deterministic \
# --work-dir ./exp/timesformer_divST_8x32x1_15e_hmdb51s1_rgb_1e4_finetunek400


# CUDA_VISIBLE_DEVICES=1 ./tools/dist_test.sh configs/recognition/timesformer/timesformer_divST_8x32x1_15e_hmdb51s1_rgb_finetune.py \
# /home/myth/workplace/mmaction2/exp/timesformer_divST_8x32x1_15e_hmdb51s1_rgb_1e3_finetunek400/best_top1_acc_epoch_20.pth \
# 1 --eval top_k_accuracy

# train with smaller lr for 8GPUs on 2GPUs
CUDA_VISIBLE_DEVICES=1 ./tools/dist_train.sh \
/home/myth/workplace/mmaction2/configs/compressedvideo/i3d/i3d_r50_16x4x1_100e_hmdb51_residual.py 1 \
--validate --seed 0 --deterministic \
--work-dir ./exp/i3d_r50_16x4x1_100e_hmdb51_residual

# CUDA_VISIBLE_DEVICES=0,1 ./tools/dist_train.sh \
# configs/recognition/timesformer/timesformer_divST_16x4x1_15e_hmdb51s2_rgb_finetune.py 2 \
# --validate --seed 0 --deterministic \
# --work-dir ./exp/timesformer_divST_16x4x1_15e_hmdb51s2_rgb_SGB1e4_finetunek400

# CUDA_VISIBLE_DEVICES=0,1 ./tools/dist_train.sh \
# configs/recognition/timesformer/timesformer_divST_16x4x1_15e_hmdb51s3_rgb_finetune.py 2 \
# --validate --seed 0 --deterministic \
# --work-dir ./exp/timesformer_divST_16x4x1_15e_hmdb51s3_rgb_SGB1e4_finetunek400

# ./jobs2.sh