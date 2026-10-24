python scripts/7.1.gen.CQ500.pt.images/gen_eval_indices.py \
    --test-root datasets/rendered_images/CQ500_merged_split/test \
    -o methods/TokenGS/assets/10.21.benchmark.test.volumes.first10.tfn/cq500_eval_6v_first10tfn_my_pick_for_cq500.json \
    --context 0 12 15 19 22 47 --max-tfn 10

export VOLUME_EVAL_ROOT=/home/kctung/Projects/FFGS-benchmark/datasets/rendered_images/CQ500_merged_split/test

cd methods/TokenGS
conda activate tokengs
accelerate launch --config_file acc_configs/gpu1.yaml \
    -m tokengs.evaluate eval_volume_latent_6view_ssim \
    --workspace results/10.21.benchmark.test.volumes.first10.tfn/CQ500_6v_my_pick_context \
    --resume workspace/10.13.new.mix.dataset.train.set.only/latent_ssim_model_lambert_even_view/model.safetensors \
    --use_ttt_for_eval \
    --eval_n_media_dumps 20 \
    --dataset_kwargs evaluation_json assets/10.21.benchmark.test.volumes.first10.tfn/cq500_eval_6v_first10tfn_my_pick_for_cq500.json \
    --bg_color black

accelerate launch --config_file acc_configs/gpu1.yaml \
    -m tokengs.evaluate eval_volume_latent_6view_ssim \
    --workspace results/10.21.benchmark.test.volumes.first10.tfn/CQ500_6v_my_pick_context_no_ttt \
    --resume workspace/10.13.new.mix.dataset.train.set.only/latent_ssim_model_lambert_even_view/model.safetensors \
    --eval_n_media_dumps 20 \
    --dataset_kwargs evaluation_json assets/10.21.benchmark.test.volumes.first10.tfn/cq500_eval_6v_first10tfn_my_pick_for_cq500.json \
    --bg_color black