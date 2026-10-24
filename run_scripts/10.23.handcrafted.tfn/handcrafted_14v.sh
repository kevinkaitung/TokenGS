#!/bin/bash
# Handcrafted paper TFs (configs/handcrafted_tfs/): 3 CQ500 + 3 Nyx scenes, plus 3 Nyx scenes
# tuned for premultiplied (rgb*alpha) output. Scenes come from merge_eval_views.py
# (frames 0-49 train, 50-69 eval), so the 14v context (6 axis + 8 corner views) / 20 eval targets apply.
set -e
source "$(conda info --base)/etc/profile.d/conda.sh"
REPO=/home/kctung/Projects/FFGS-benchmark
ASSETS=assets/10.23.handcrafted.tfn
RESULTS=results/10.23.handcrafted.tfn
CKPT=workspace/10.13.new.mix.dataset.train.set.only/latent_ssim_model_lambert_even_view/model.safetensors

cd $REPO
conda activate ovr_redbull
python scripts/7.1.gen.CQ500.pt.images/gen_eval_indices.py --force \
    --test-root datasets/rendered_images/handcrafted_tfs_eval_merged \
    -o methods/TokenGS/$ASSETS/handcrafted_eval_14v.json --views 14v
python scripts/7.1.gen.CQ500.pt.images/gen_eval_indices.py --force \
    --test-root datasets/rendered_images/handcrafted_tfs_premul_eval_merged \
    -o methods/TokenGS/$ASSETS/handcrafted_premul_eval_14v.json --views 14v

# run from methods/TokenGS so its own tokengs package is imported
cd $REPO/methods/TokenGS
conda activate tokengs
for name in handcrafted handcrafted_premul; do
    export VOLUME_EVAL_ROOT=$REPO/datasets/rendered_images/${name/handcrafted/handcrafted_tfs}_eval_merged

    accelerate launch --config_file acc_configs/gpu1.yaml \
        -m tokengs.evaluate eval_volume_latent_6view_ssim --num_input_views 14 \
        --workspace $RESULTS/${name}_14v \
        --resume $CKPT \
        --use_ttt_for_eval \
        --eval_n_media_dumps 20 \
        --dataset_kwargs evaluation_json $ASSETS/${name}_eval_14v.json \
        --bg_color black

    accelerate launch --config_file acc_configs/gpu1.yaml \
        -m tokengs.evaluate eval_volume_latent_6view_ssim --num_input_views 14 \
        --workspace $RESULTS/${name}_14v_no_ttt \
        --resume $CKPT \
        --eval_n_media_dumps 20 \
        --dataset_kwargs evaluation_json $ASSETS/${name}_eval_14v.json \
        --bg_color black
done
