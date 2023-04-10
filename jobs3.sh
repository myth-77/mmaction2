CUDA_VISIBLE_DEVICES=1 ./tools/train.sh \
/home/myth/workplace/mmaction2/configs/compressedvideo/timesformer/timesformer_divST_16x4x1_15e_hmdb51s1_residual_kd.py 1 \
--validate --seed 0 --deterministic \
--work-dir ./exp/timesformer_divST_16x4x1_30e_hmdb51s1_residual_kd0.1-cls0.9_lr1e4 \
--gpus 1

./jobs2.sh