#!/bin/bash
# Handcrafted premultiplied-alpha TFs (configs/handcrafted_tfs/{nyx,cq500}_premul/), 33 scenes:
# 13 Nyx (nyx_premul_*), 10 TFs on one CQ500 volume (cq500_premul_*), and the same 10 TFs on 10 CQ500 test volumes
# (CQ500CT<N>_premul_*). Scenes come from merge_eval_views.py (frames 0-49 train, 50-69 eval), so the 20 eval
# targets apply. Revised from ../10.23.handcrafted.tfn/handcrafted_{6v,14v}.sh:
#   - 6v uses the paper benchmark's context: hand-picked cq500_6v for CQ500 scenes, generic 6v for Nyx scenes
#     (their cameras match the CQ500 / Nyx benchmark layouts); 14v uses the generic 14v set for both.
#   - --eval_n_media_dumps 300 so every scene gets views/<scene>/target_XX_{gt,pred}.png for figures.
#   - new asset / result dirs; existing JSONs are never overwritten.
set -e
source "$(conda info --base)/etc/profile.d/conda.sh"
REPO=/home/kctung/Projects/FFGS-benchmark
ROOT=$REPO/datasets/rendered_images/handcrafted_tfs_premul_eval_merged
ASSETS=assets/10.28.handcrafted.tfn.premul
RESULTS=results/10.28.handcrafted.tfn.premul
# VIEWS limits which context sets run, e.g. VIEWS=6 bash <this script>
VIEWS=${VIEWS:-"6 14"}
CKPT=workspace/10.13.new.mix.dataset.train.set.only/latent_ssim_model_lambert_even_view/model.safetensors

cd $REPO
conda activate tokengs
A=methods/TokenGS/$ASSETS
mkdir -p $A
gen() {  # gen <output> <args...>: write an eval JSON unless it already exists
    [ -f "$1" ] || python scripts/7.1.gen.CQ500.pt.images/gen_eval_indices.py --test-root $ROOT -o "$@"
}
gen $A/handcrafted_premul_eval_14v.json --views 14v
if [ ! -f $A/handcrafted_premul_eval_6v.json ]; then
    gen $A/_all_6v.json --views 6v
    gen $A/_all_cq500_6v.json --views cq500_6v
    # Nyx scenes take the generic 6v context, CQ500 scenes the hand-picked one
    python - $A <<'EOF'
import json, sys
a = sys.argv[1]
g, c = json.load(open(f"{a}/_all_6v.json")), json.load(open(f"{a}/_all_cq500_6v.json"))
out = {k: (c[k] if k.lower().startswith("cq500") else g[k]) for k in g}
json.dump(out, open(f"{a}/handcrafted_premul_eval_6v.json", "w"), indent=2)
print(f"handcrafted_premul_eval_6v.json: {sum(k.lower().startswith('cq500') for k in out)} CQ500 (cq500_6v), "
      f"{sum(not k.lower().startswith('cq500') for k in out)} Nyx (6v)")
EOF
    rm $A/_all_6v.json $A/_all_cq500_6v.json
fi

# run from methods/TokenGS so its own tokengs package is imported
cd $REPO/methods/TokenGS
export VOLUME_EVAL_ROOT=$ROOT
for v in $VIEWS; do
    accelerate launch --config_file acc_configs/gpu1.yaml \
        -m tokengs.evaluate eval_volume_latent_6view_ssim --num_input_views $v \
        --workspace $RESULTS/handcrafted_premul_${v}v \
        --resume $CKPT \
        --use_ttt_for_eval \
        --eval_n_media_dumps 300 \
        --dataset_kwargs evaluation_json $ASSETS/handcrafted_premul_eval_${v}v.json \
        --bg_color black

    accelerate launch --config_file acc_configs/gpu1.yaml \
        -m tokengs.evaluate eval_volume_latent_6view_ssim --num_input_views $v \
        --workspace $RESULTS/handcrafted_premul_${v}v_no_ttt \
        --resume $CKPT \
        --eval_n_media_dumps 300 \
        --dataset_kwargs evaluation_json $ASSETS/handcrafted_premul_eval_${v}v.json \
        --bg_color black
done
