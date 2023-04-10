CUDA_VISIBLE_DEVICES=1 ./tools/train.sh \
/home/myth/workplace/mmaction2/configs/compressedvideo/i3d/i3d_r50_16x4x1_hmdb51s1_timsformer_kd_residual.py 1 \
--validate --seed 0 --deterministic \
--work-dir ./exp/i3d_r50_16x4x1_hmdb51s1_timsformer_kd_residual_kd0.1-cls0.9-fix \
--gpus 1

./jobs3.sh

