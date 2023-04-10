CUDA_VISIBLE_DEVICES=1 ./tools/train.sh \
/home/myth/workplace/mmaction2/configs/recognition/uniformer/uniformer-base_imagenet1k-pre_16x4x1_hmdb51s1-rgb.py 1 \
--validate --seed 0 --deterministic \
--work-dir ./exp/uniformer-base_imagenet1k-pre_16x4x1_hmdb51s1-rgb \
--gpus 1
