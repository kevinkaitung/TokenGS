# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from pathlib import Path

from tokengs.data.dynamic.kubric import Kubric
from tokengs.data.static.dl3dv import DL3DV10K, DL3DVEval

# Repository root (parent of the `tokengs` package)
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DL3DV_ROOT = _REPO_ROOT / "data" / "dl3dv"
_DEFAULT_DL3DV_EVAL_ROOT = _REPO_ROOT / "data" / "dl3dv_eval"
_DEFAULT_KUBRIC_ROOT = _REPO_ROOT / "data" / "kubric"

# add default dataset root for volume rendered images
# the default path assumes this tokengs repo is under FFGS-benchmark repo
# if not, set corresponding env var (VOLUME_ROOT or VOLUME_EVAL_ROOT) to default dataset root
_DEFAULT_VOLUME_ROOT = Path(
    os.environ.get("VOLUME_ROOT", _REPO_ROOT.parents[1] / "datasets" / "rendered_images" / "CQ500_split" / "train" )
)
_DEFAULT_VOLUME_EVAL_ROOT = Path(
    os.environ.get("VOLUME_EVAL_ROOT", _REPO_ROOT.parents[1] / "datasets" / "rendered_images" / "CQ500_split" / "test" )
)
print(f"Train Datatset directory: {_DEFAULT_VOLUME_ROOT}")
print(f"Eval Datatset directory: {_DEFAULT_VOLUME_EVAL_ROOT}")

dataset_registry = {}

# DL3DV: point `data/dl3dv` at your DL3DV-ALL (or compatible) tree, e.g. `ln -snf /path/to/DL3DV-ALL-960P-undistorted data/dl3dv`
dataset_registry["dl3dv"] = {
    "cls": DL3DV10K,
    "kwargs": {
        "root_path": str(_DEFAULT_DL3DV_ROOT),
        "resolution": "960p_images",
    },
    "scene_scale": 1.0,
    "max_gap": 40,
    "min_gap": 10,
}

# Eval benchmark: point `data/dl3dv_eval` at DL3DV-10K-Benchmark (or compatible)
dataset_registry["dl3dv_eval"] = {
    "cls": DL3DVEval,
    "kwargs": {
        "root_path": str(_DEFAULT_DL3DV_EVAL_ROOT),
        "subset": ["140"],
    },
    "scene_scale": 1.0,
    "max_gap": 1e6,
    "min_gap": 0,
}

# add dataset registry for my own volume rendered images
# (my dataset format follows DL3DV dataset, so use directly use the class of DL3DV10K)
dataset_registry["volume"] = {
    "cls": DL3DV10K,
    "kwargs": {
        "root_path": str(_DEFAULT_VOLUME_ROOT),
        "resolution": "256x256",  # volume rendered image resolution "<H>x<W>"
        "subset": ["140"]   # my dataset's file structure follows DL3DV 140 subset, 
        # so use "140" as subset here
    },
    "scene_scale": 1.0, # default value, but might be overwritten in provider.py 
    "max_gap": 40,
    "min_gap": 10,
}

dataset_registry["volume_eval"] = {
    "cls": DL3DVEval,
    "kwargs": {
        "root_path": str(_DEFAULT_VOLUME_EVAL_ROOT),
        "resolution": "256x256",  # volume rendered image resolution "<H>x<W>"
        "subset": ["140"],  # my dataset's file structure follows DL3DV 140 subset, 
        # so use "140" as subset here
    },
    "scene_scale": 1.0, # default value, but might be overwritten in provider.py 
    "max_gap": 1e6,
    "min_gap": 0,
}

# Kubric multi-view 4D dataset.
# Point `data/kubric` at the kubric_mv dump, e.g.
#   ln -snf /path/to/objaverse_4d/kubric_mv data/kubric
# Layout: <root>/{v0,v1,v2}/<scene>/output_{view:03d}.tar
# Each scene contains 9 cameras x 24 frames.
dataset_registry["kubric"] = {
    "cls": Kubric,
    "kwargs": {
        "root_path": str(_DEFAULT_KUBRIC_ROOT),
        "subset": ("v0", "v1", "v2"),
    },
    "scene_scale": 1.0,
    "max_gap": 23,
    "min_gap": 3,
}
