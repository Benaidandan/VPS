import argparse
import collections.abc as collections
from pathlib import Path
import re
from typing import Optional, List, Dict, Union
import h5py
import numpy as np
import torch
import json
import time
from hloc import logger
from hloc.utils.io import list_h5_names
from hloc.utils.parsers import parse_image_lists
from hloc.utils.read_write_model import read_images_binary
import logging


def parse_names(prefix, names, names_all):
    if prefix is not None:
        if not isinstance(prefix, str):
            prefix = tuple(prefix)
        names = [n for n in names_all if n.startswith(prefix)]
        if len(names) == 0:
            raise ValueError(f"Could not find any image with the prefix `{prefix}`.")
    elif names is not None:
        if isinstance(names, (str, Path)):
            names = parse_image_lists(names)
        elif isinstance(names, collections.Iterable):
            names = list(names)
        else:
            raise ValueError(
                f"Unknown type of image list: {names}."
                "Provide either a list or a path to a list file."
            )
    else:
        names = names_all
    return names


def get_descriptors(names, path, name2idx=None, key="global_descriptor"):
    if name2idx is None:
        with h5py.File(str(path), "r", libver="latest") as fd:
            desc = [fd[n][key].__array__() for n in names]
    else:
        desc = []
        for n in names:
            with h5py.File(str(path[name2idx[n]]), "r", libver="latest") as fd:
                desc.append(fd[n][key].__array__())
    return torch.from_numpy(np.stack(desc, 0)).float()


def pairs_from_score_matrix(
    scores: torch.Tensor,
    invalid: np.ndarray,
    num_select: int,
    min_score: Optional[float] = None,
):
    assert scores.shape == invalid.shape
    if isinstance(scores, np.ndarray):
        scores = torch.from_numpy(scores)
    invalid = torch.from_numpy(invalid).to(scores.device)
    if min_score is not None:
        invalid |= scores < min_score
    scores.masked_fill_(invalid, float("-inf"))

    topk = torch.topk(scores, num_select, dim=1)
    indices = topk.indices.cpu().numpy()
    valid = topk.values.isfinite().cpu().numpy()

    pairs = []
    for i, j in zip(*np.where(valid)):
        pairs.append((i, indices[i, j]))
    return pairs


def spatial_filter(db_names, db_desc, last_pose, ref_poses_dir, spatial_radius):
    start_time = time.time()
    pose_list = []
    valid_indices = []
    for idx, name in enumerate(db_names):
        pose_path = Path(ref_poses_dir) / f"{Path(name).stem}.txt"
        pose = np.loadtxt(pose_path)
        pose_list.append(pose[:3, 3])
        valid_indices.append(idx)
    if len(pose_list) == 0:
        logging.info("空间过滤失效")
        return db_names, db_desc
    poses = np.stack(pose_list)  # shape: (N, 3)
    dists = np.linalg.norm(poses - last_pose, axis=1)
    keep_mask = dists <= spatial_radius
    keep_indices = np.array(valid_indices)[keep_mask]
    db_names_filtered = [db_names[i] for i in keep_indices]
    db_desc_filtered = db_desc[keep_indices]
    logging.info(f"空间过滤后剩余: {len(db_names_filtered)} 张参考图像")
    end_time = time.time()
    logging.info(f"空间过滤时间: {end_time - start_time:.4f} 秒")
    return db_names_filtered, db_desc_filtered


def find_similar(
    query_descriptors,
    db_descriptors,
    db_names,
    db_desc,
    output,
    num_matched,
    query_prefix=None,
    query_list=None,
    similarity_threshold=0.7,
    last_pose=None,
    spatial_radius=None,
    use_spatial_filtering=False,
    ref_poses_dir=None,
):

    start_time = time.time()
    query_names_h5 = list_h5_names(query_descriptors)
    if len(db_names) == 0:
        logging.error("Could not find any database image.")
        raise ValueError("Could not find any database image.")
    end_time = time.time()
    start_time = time.time()
    query_names = parse_names(query_prefix, query_list, query_names_h5)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    query_desc = get_descriptors(query_names, query_descriptors)
    if use_spatial_filtering \
    and last_pose is not None and len(last_pose) > 0 \
    and spatial_radius is not None \
    and ref_poses_dir is not None:
        db_names, db_desc = spatial_filter(db_names, db_desc, last_pose, ref_poses_dir, spatial_radius)
    sim = torch.einsum("id,jd->ij", query_desc.to(device), db_desc.to(device))
    # Avoid self-matching
    self = np.array(query_names)[:, None] == np.array(db_names)[None]
    pairs = pairs_from_score_matrix(sim, self, num_matched, min_score=similarity_threshold)  
    pairs = [(query_names[i], db_names[j]) for i, j in pairs]
    with open(output, "w") as f:
        f.write("\n".join(" ".join([i, j]) for i, j in pairs))


