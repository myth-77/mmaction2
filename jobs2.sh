# # CUDA_VISIBLE_DEVICES=1 ./tools/train.sh \
# # /home/myth/workplace/mmaction2/configs/compressedvideo/tsn/tsn_r50_1x1x8_50e_hmdb51s1_kinetics400_res.py 1 \
# # --validate --seed 0 --deterministic \
# # --work-dir ./exp/tsn_r50_1x1x8_50e_hmdb51s1_k400pre_res \
# # --gpus 1

# CUDA_VISIBLE_DEVICES=1 ./tools/train.sh \
# /home/myth/workplace/mmaction2/configs/compressedvideo/tsn/tsn_r50_1x1x8_50e_hmdb51s2_kinetics400_res.py 1 \
# --validate --seed 0 --deterministic \
# --work-dir ./exp/tsn_r50_1x1x8_50e_hmdb51s2_k400pre_res \
# --gpus 1

# CUDA_VISIBLE_DEVICES=1 ./tools/train.sh \
# /home/myth/workplace/mmaction2/configs/compressedvideo/tsn/tsn_r50_1x1x8_50e_hmdb51s3_kinetics400_res.py 1 \
# --validate --seed 0 --deterministic \
# --work-dir ./exp/tsn_r50_1x1x8_50e_hmdb51s3_k400pre_res \
# --gpus 1


# ./jobs3.sh

CUDA_VISIBLE_DEVICES=0 ./tools/train.sh \
/home/myth/workplace/mmaction2/configs/compressedvideo/i3d/i3d_r50_16x4x1_hmdb51s1_timsformer_kd_feat_residual_1.py 1 \
--validate --seed 0 --deterministic \
--work-dir ./exp/i3d_r50_16x4x1_hmdb51s1_timsformer_kd_feat_residual_cls1_kd9
