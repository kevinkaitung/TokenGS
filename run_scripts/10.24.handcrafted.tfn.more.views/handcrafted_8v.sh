#!/bin/bash
# Handcrafted paper TFs (configs/handcrafted_tfs/): 3 CQ500 + 3 Nyx scenes, plus Nyx scenes
# tuned for premultiplied (rgb*alpha) output. Scenes come from merge_eval_views.py
# (frames 0-49 train, 50-69 eval), so the 8v context (evenly spread views, see gen_eval_indices.py) / 20 eval targets apply.
set -e
source "$(conda info --base)/etc/profile.d/conda.sh"
REPO=/home/kctung/Projects/FFGS-benchmark
ASSETS=assets/10.24.handcrafted.tfn.more.views
RESULTS=results/10.24.handcrafted.tfn.more.views
# SETS limits which eval sets run, e.g. SETS=handcrafted_premul bash <this script>
SETS=${SETS:-"handcrafted handcrafted_premul"}
CKPT=workspace/10.13.new.mix.dataset.train.set.only/latent_ssim_model_lambert_even_view/model.safetensors

cd $REPO
conda activate ovr_redbull
python scripts/7.1.gen.CQ500.pt.images/gen_eval_indices.py --force \
    --test-root datasets/rendered_images/handcrafted_tfs_eval_merged \
    -o methods/TokenGS/$ASSETS/handcrafted_eval_8v.json --views 8v
python scripts/7.1.gen.CQ500.pt.images/gen_eval_indices.py --force \
    --test-root datasets/rendered_images/handcrafted_tfs_premul_eval_merged \
    -o methods/TokenGS/$ASSETS/handcrafted_premul_eval_8v.json --views 8v

# run from methods/TokenGS so its own tokengs package is imported
cd $REPO/methods/TokenGS
conda activate tokengs
for name in $SETS; do
    export VOLUME_EVAL_ROOT=$REPO/datasets/rendered_images/${name/handcrafted/handcrafted_tfs}_eval_merged

    accelerate launch --config_file acc_configs/gpu1.yaml \
        -m tokengs.evaluate eval_volume_latent_6view_ssim --num_input_views 8 \
        --workspace $RESULTS/${name}_8v \
        --resume $CKPT \
        --use_ttt_for_eval \
        --eval_n_media_dumps 20 \
        --dataset_kwargs evaluation_json $ASSETS/${name}_eval_8v.json \
        --bg_color black

    accelerate launch --config_file acc_configs/gpu1.yaml \
        -m tokengs.evaluate eval_volume_latent_6view_ssim --num_input_views 8 \
        --workspace $RESULTS/${name}_8v_no_ttt \
        --resume $CKPT \
        --eval_n_media_dumps 20 \
        --dataset_kwargs evaluation_json $ASSETS/${name}_eval_8v.json \
        --bg_color black
done
