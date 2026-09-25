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

import json
import struct
import zipfile
from typing import Dict, List, Optional, Tuple
import os
import numpy as np
import torch
from pathlib import Path
import glob
from PIL import Image
from io import BytesIO

from tokengs.data.datafield import (
    DF_CAMERA_C2W_TRANSFORM,
    DF_CAMERA_INTRINSICS,
    DF_DEPTH,
    DF_IMAGE_RGB,
)


class _ZipSceneSource:
    """Reads scene payload paths (``"<clip>/transforms.json"``) out of a zip."""

    def __init__(self, path):
        self._zip = zipfile.ZipFile(path, "r")
        self._names = None

    def open(self, name):
        return self._zip.open(name, "r")

    def exists(self, name):
        if self._names is None:
            self._names = set(self._zip.namelist())
        return name in self._names

    def close(self):
        self._zip.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class _DirSceneSource:
    """Same interface as :class:`_ZipSceneSource` for an unpacked scene folder.

    Names are resolved against the scene folder's *parent*, so the very same
    ``f"{clip_name}/..."`` strings used for zips work unchanged.
    """

    def __init__(self, path):
        self._base = Path(path).parent

    def open(self, name):
        return open(self._base / name, "rb")

    def exists(self, name):
        return (self._base / name).exists()

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _scene_key(path) -> str:
    """Scene hash / name for either a ``<hash>.zip`` or a ``<hash>/`` folder."""
    p = Path(path)
    return p.stem if p.suffix.lower() == ".zip" else p.name


def _is_scene_dir(path) -> bool:
    p = Path(path)
    return p.is_dir() and (
        (p / "transforms.json").exists()
        or (p / "gaussian_splat" / "transforms.json").exists()
    )


def _open_scene_source(path):
    if str(path).lower().endswith(".zip"):
        return _ZipSceneSource(path)
    return _DirSceneSource(path)


class DL3DV10K:
    def __init__(
        self,
        root_path,
        subset=['1K', '2K', '3K', '4K', '5K', '6K', '7K', '8K', '9K', '10K', '11K'],
        resolution='960p',
        load_depth=False,
        scene_format: str = 'auto',
        **kwargs,
    ):
        """
        data_format: support different formats for what different code base expect

        ``scene_format`` selects how each scene is stored under ``root_path``:
        ``'zip'`` for the packed DL3DV dumps (``<hash>.zip``), ``'folder'`` for
        unpacked scene directories (``<hash>/transforms.json`` or
        ``<hash>/gaussian_splat/transforms.json``, e.g. the CQ500 renders), or
        ``'auto'`` (default) to pick up both. Both layouts are read through the
        same code path, so everything below applies to either one.
        """
        super().__init__(**kwargs)
        if scene_format not in ('auto', 'zip', 'folder'):
            raise ValueError(
                f"scene_format must be 'auto', 'zip' or 'folder', got {scene_format!r}"
            )
        self.root_path = root_path
        self.subset = subset
        self.load_depth = load_depth
        self.scene_format = scene_format

        self.sample_list = []
        for sub in self.subset:
            search_dir = root_path if sub == '140' else f"{root_path}/{sub}"
            self.sample_list.extend(self._discover_scenes(search_dir))

        if resolution == '960p':
            # self.resolution = [540, 960]
            # self.resolution = [600, 600]
            self.resolution = [256, 256]
            self.image_folder = 'images_4'
        elif resolution == '960p_images':
            # For training dataset that uses 'images' folder instead of 'images_4'
            # self.resolution = [540, 960]
            # self.resolution = [600, 600]
            self.resolution = [256, 256]
            self.image_folder = 'images'
        else:
            raise NotImplementedError(f"Resolution {resolution} not supported")
        
        self.is_static = True

    def _discover_scenes(self, search_dir) -> List[str]:
        """List the scene zips and/or scene folders directly under ``search_dir``."""
        scenes: List[str] = []
        if self.scene_format in ('auto', 'zip'):
            scenes.extend(glob.glob(f"{search_dir}/*.zip"))
        if self.scene_format in ('auto', 'folder'):
            scenes.extend(p for p in glob.glob(f"{search_dir}/*") if _is_scene_dir(p))
        return sorted(scenes)

    def _open_scene(self, idx):
        """Open scene ``idx`` and return ``(clip_name, source)``.

        ``clip_name`` is the prefix every payload path is under: the scene key
        itself, or ``"<key>/gaussian_splat"`` for the DL3DV-140 / CQ500 layout.
        """
        path = self.sample_list[idx]
        source = _open_scene_source(path)
        key = _scene_key(path)
        try:
            for clip_name in (f"{key}/gaussian_splat", key):
                if source.exists(f"{clip_name}/transforms.json"):
                    return clip_name, source
        except Exception:
            source.close()
            raise
        # Nothing matched (e.g. a zip we cannot cheaply introspect): fall back
        # to the historical subset-driven rule and let the open below fail.
        # From original implementation of official TokenGS repo.
        if self.subset == ['140']:
            return f"{key}/gaussian_splat", source
        return key, source

    def __len__(self):
        return len(self.sample_list)
    
    def load_intrinsics(self, data_dict, resolution = None):
        fx = data_dict['fl_x']
        fy = data_dict['fl_y']
        cx = data_dict['cx']
        cy = data_dict['cy']

        intrinsics = np.array([fx, fy, cx, cy], dtype=np.float32)
        if resolution is not None:
            H_new, W_new = resolution
            W_old = data_dict['w']
            H_old = data_dict['h']

            # Scale the intrinsics to the new resolution
            intrinsics[0] *= W_new / W_old
            intrinsics[1] *= H_new / H_old
            intrinsics[2] *= W_new / W_old
            intrinsics[3] *= H_new / H_old

        return intrinsics

    def load_video_reader(self, idx):
        clip_name, scene_source = self._open_scene(idx)

        # load json data
        with scene_source.open(f"{clip_name}/transforms.json") as f:
            json_data = json.load(f)

        # load video length
        video_length = len(json_data['frames'])

        # load intrinsics
        intrinsics = self.load_intrinsics(json_data, resolution = self.resolution)
        transform_matrix_all = self.load_cameras(json_data)

        return clip_name, video_length, intrinsics, transform_matrix_all, scene_source, json_data

    def load_cameras(self, data_dict):
        transform_matrix_all = []
        for frame_data in data_dict['frames']:
            transform_matrix = np.array(frame_data['transform_matrix'])
            c2w = transform_matrix
            c2w[2, :] *= -1
            c2w = c2w[np.array([1, 0, 2, 3]), :]
            c2w[0:3, 1:3] *= -1
            transform_matrix_all.append(c2w)
        return np.stack(transform_matrix_all, axis=0)

    def count_cameras(self, video_idx: int) -> int:
        return 1
    
    def count_frames(self, idx):
        clip_name, scene_source = self._open_scene(idx)
        with scene_source:
            # load json data
            with scene_source.open(f"{clip_name}/transforms.json") as f:
                json_data = json.load(f)
            total_frames = len(json_data['frames'])

        return total_frames

    def get_data(
        self,
        idx,
        data_fields: List[str],
        frame_indices: Optional[List[int]] = None,
        view_indices: List[int] = None,
        camera_convention: str = "opencv",
        num_depth_frames: Optional[int] = None,
    ):
        assert camera_convention == "opencv"

        clip_name, total_frames, intrinsics, transform_matrices, scene_source, json_data = self.load_video_reader(idx)
        if frame_indices is None:
            frame_indices = range(total_frames)

        # load camera poses
        c2w = transform_matrices[frame_indices]
        
        # load img_seq
        img_seq = []
        for frame_idx in frame_indices:
            img_name = json_data['frames'][frame_idx]['file_path'].split('/')[-1]
            with scene_source.open(f"{clip_name}/{self.image_folder}/{img_name}") as f:
                img = Image.open(BytesIO(f.read()))
                img_seq.append(np.array(img))
        img_seq = np.stack(img_seq, axis=0) # n h w c
        img_seq = torch.from_numpy(img_seq).permute(0, 3, 1, 2).contiguous()
        img_seq = img_seq / 255.0  # (0,1)

        c2w = torch.from_numpy(c2w).float().contiguous()
        # prepare intrinsics
        intrinsics = torch.from_numpy(intrinsics).unsqueeze(0).repeat(len(frame_indices), 1)

        output_dict = {}
        output_dict["__key__"] = clip_name
        for data_field in data_fields:
            if data_field == DF_IMAGE_RGB:
                output_dict[data_field] = img_seq
            elif data_field == DF_CAMERA_C2W_TRANSFORM:
                output_dict[data_field] = c2w
            elif data_field == DF_CAMERA_INTRINSICS:
                output_dict[data_field] = intrinsics
            elif data_field == DF_DEPTH and self.load_depth:
                depth_seq = self._load_depth_seq(
                    clip_name, json_data, scene_source, frame_indices, num_depth_frames
                )
                output_dict[data_field] = torch.from_numpy(depth_seq).float().unsqueeze(1)


        return output_dict

    def _load_depth_seq(
        self,
        clip_name,
        json_data,
        scene_source,
        frame_indices,
        num_depth_frames: Optional[int],
    ) -> np.ndarray:
        n_load = len(frame_indices) if num_depth_frames is None else min(num_depth_frames, len(frame_indices))
        depth_seq = []
        for i, frame_idx in enumerate(frame_indices):
            if i < n_load:
                depth_path = f"{clip_name}/{json_data['frames'][frame_idx]['depth_path']}"
                with scene_source.open(depth_path) as f:
                    depth = np.load(BytesIO(f.read()))
            else:
                depth = np.zeros(self.resolution, dtype=np.float64)
            depth_seq.append(depth)
        return np.stack(depth_seq, axis=0)


class DL3DVEval(DL3DV10K):
    def __init__(self, root_path, evaluation_json, subset = ['1K', '2K', '3K', '4K', '5K', '6K', '7K', '8K', '9K', '10K', '11K'], resolution = '960p', num_input=16, load_depth=False, scene_format: str = 'auto'):
        super().__init__(root_path, subset, resolution, load_depth=load_depth, scene_format=scene_format)

        self.evaluation_indices = json.load(open(evaluation_json, "r"))
        self.sample_list = []
        for k in self.evaluation_indices:
            scene_name = k if isinstance(k, str) else k['scene_name']
            self.sample_list.append(self._resolve_scene_path(root_path, scene_name))
        
        self.is_static = True

        self.num_input = num_input
        self._colmap_points_cache: Dict[str, Tuple[np.ndarray, List[frozenset]]] = {}

    def _resolve_scene_path(self, root_path, scene_name: str) -> str:
        """Path of an eval scene, as a zip or as an unpacked folder."""
        zip_path = os.path.join(root_path, scene_name + '.zip')
        dir_path = os.path.join(root_path, scene_name)
        if self.scene_format == 'zip':
            return zip_path
        if self.scene_format == 'folder':
            return dir_path
        if not os.path.exists(zip_path) and os.path.isdir(dir_path):
            return dir_path
        # Keep the zip path when neither exists so the error names the zip.
        return zip_path

    def _read_points3d_bin(
        self, scene_source, clip_name: str
    ) -> Tuple[np.ndarray, List[frozenset]]:
        path = f"{clip_name}/sparse/0/points3D.bin"
        with scene_source.open(path) as f:
            data = f.read()

        off = 0
        (n_points,) = struct.unpack_from("<Q", data, off)
        off += 8
        xyz = np.empty((n_points, 3), dtype=np.float64)
        image_id_sets: List[frozenset] = []
        for i in range(n_points):
            off += 8
            xyz[i] = struct.unpack_from("<ddd", data, off)
            off += 24 + 3 + 8
            (track_len,) = struct.unpack_from("<Q", data, off)
            off += 8
            track = struct.unpack_from("<" + "II" * track_len, data, off)
            off += 8 * track_len
            image_id_sets.append(frozenset(track[::2]))
        return xyz, image_id_sets

    def _get_colmap_points(
        self, scene_source, clip_name: str
    ) -> Tuple[np.ndarray, List[frozenset]]:
        cached = self._colmap_points_cache.get(clip_name)
        if cached is not None:
            return cached
        cached = self._read_points3d_bin(scene_source, clip_name)
        self._colmap_points_cache[clip_name] = cached
        return cached

    @staticmethod
    def _colmap_world_to_final_world_transform(json_data: Dict) -> np.ndarray:
        transform = np.eye(4, dtype=np.float64)
        applied = json_data.get("applied_transform")
        if applied is not None:
            transform[:3, :] = np.asarray(applied, dtype=np.float64)

        z_flip = np.diag([1.0, 1.0, -1.0, 1.0])
        xy_swap = np.array(
            [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float64,
        )
        return xy_swap @ z_flip @ transform

    def _load_depth_seq(
        self,
        clip_name,
        json_data,
        scene_source,
        frame_indices,
        num_depth_frames: Optional[int],
    ) -> np.ndarray:
        xyz_all, image_id_sets = self._get_colmap_points(scene_source, clip_name)
        world_transform = self._colmap_world_to_final_world_transform(json_data)

        intrinsics = self.load_intrinsics(json_data, resolution=self.resolution)
        fx, fy, cx, cy = intrinsics
        c2ws_all = self.load_cameras(json_data)
        H, W = self.resolution

        xyz_h = np.concatenate(
            [xyz_all, np.ones((xyz_all.shape[0], 1), dtype=np.float64)], axis=1
        )
        xyz_world = (xyz_h @ world_transform.T)[:, :3]

        n_frames = len(frame_indices)
        n_load = n_frames if num_depth_frames is None else min(num_depth_frames, n_frames)
        depth_seq = np.zeros((n_frames, H, W), dtype=np.float64)
        for i in range(n_load):
            frame_idx = int(frame_indices[i])
            colmap_image_id = json_data["frames"][frame_idx].get("colmap_im_id")
            if colmap_image_id is None:
                continue

            mask = np.fromiter(
                (colmap_image_id in s for s in image_id_sets),
                dtype=bool,
                count=len(image_id_sets),
            )
            if not mask.any():
                continue

            c2w = c2ws_all[frame_idx]
            w2c = np.linalg.inv(c2w)
            pts_cam = xyz_world[mask] @ w2c[:3, :3].T + w2c[:3, 3]
            z = pts_cam[:, 2]
            in_front = z > 1e-6
            if not in_front.any():
                continue
            pts_cam = pts_cam[in_front]
            z = z[in_front]

            u = fx * pts_cam[:, 0] / z + cx
            v = fy * pts_cam[:, 1] / z + cy
            u_i = np.round(u).astype(np.int64)
            v_i = np.round(v).astype(np.int64)
            in_bounds = (u_i >= 0) & (u_i < W) & (v_i >= 0) & (v_i < H)
            if not in_bounds.any():
                continue

            u_i = u_i[in_bounds]
            v_i = v_i[in_bounds]
            z = z[in_bounds]
            order = np.argsort(-z)
            depth_seq[i, v_i[order], u_i[order]] = z[order]

        return depth_seq

    def get_context_target_frames(self, idx):
        if isinstance(self.evaluation_indices, dict):
            scene_name = _scene_key(self.sample_list[idx])
            eval_data = self.evaluation_indices[scene_name]
            context_frames = eval_data["context"]
            target_frames = eval_data["target"]
            return context_frames, target_frames
        else:
            # llrm eval
            eval_data = self.evaluation_indices[idx]
            if self.num_input == 16:
                context_frames = eval_data["fold_8_kmeans_16_input"]
            elif self.num_input == 32:
                context_frames = eval_data["fold_8_kmeans_32_input"]
            else:
                raise ValueError(f"Unsupported number of input frames: {self.num_input}")
            target_frames = [x for x in range(self.count_frames(idx))]
            return context_frames, target_frames
