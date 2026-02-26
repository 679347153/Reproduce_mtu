import cmath
import dataclasses
import json
import math
from typing import Any, Dict, List, Optional
import attr
import numpy as np
import quaternion  # noqa: F401
from omegaconf import OmegaConf
import magnum as mn
import numpy as np
import quaternion as qt
from scipy.spatial.transform import Rotation as R
import os
import time
import socket
import cv2
from sklearn.metrics.pairwise import cosine_similarity


def load_episode_objects(sim, episode, transformations):
    paths = episode.additional_obj_config_paths #potential paths for object config json file
    objects = episode.rigid_objs
    
    for object in objects:
        idx = object[1]
        array = np.vstack([transformations[idx], [0, 0, 0, 1]])
        transformation = mn.Matrix4(
                [[array[j][i] for j in range(4)] for i in range(4)]
            )
        name = object[0]
        object_config_path = None
        for path in paths:
            obj_path = os.path.join(path, name)
            if os.path.exists(obj_path):
                object_config_path = obj_path
                break
        sim.load_object(object_config_path, transformation=transformation, motion="DYNAMIC")


def quaternion_to_list(q: quaternion.quaternion):
    return q.imag.tolist() + [q.real]


def quat_from_magnum(quat: mn.Quaternion) -> qt.quaternion:
    a = qt.quaternion(1, 0, 0, 0)
    a.real = quat.scalar
    a.imag = quat.vector
    return a


def not_none_validator(self: Any, attribute: attr.Attribute, value: Optional[Any]) -> None:
    if value is None:
        raise ValueError(f"Argument '{attribute.name}' must be set")


def rotate_vector_along_axis(vector, axis, radian):
    v = np.asarray(vector)
    axis = np.asarray(axis)
    axis = axis / np.linalg.norm(axis)
    r = R.from_rotvec(radian * axis)
    m = r.as_matrix()
    a = np.matmul(m, v.T)
    return a.T


class DatasetJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, quaternion.quaternion):
            return quaternion_to_list(obj)
        if OmegaConf.is_config(obj):
            return OmegaConf.to_container(obj)
        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)

        return obj.__getstate__() if hasattr(obj, "__getstate__") else obj.__dict__


def run_utility_module_in_parallel(component, cmd):
    print("Connecting to " + component + " module...")
    os.system(cmd)


def connect_with_retry(socket_file, max_retries=1000, retry_interval=2):
    retries = 0
    connected = False
    while retries < max_retries and not connected:
        try:
            client_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            client_socket.connect(socket_file)
            connected = True
        except FileNotFoundError or ConnectionRefusedError:
            retries += 1
            print(f"Waiting for the connection. Retrying in {retry_interval} seconds...")
            time.sleep(retry_interval)

    if connected:
        print("Connection successful!")
        return client_socket
    else:
        print("Connection failed after max retries.")
        return None


def transform_rgb_bgr(image, crosshair=False):
    bgr = image[:, :, [2, 1, 0]].copy()
    if crosshair:
        w, h = bgr.shape[:2]
        cx, cy = w // 2, h // 2
        l = max(w * h // 100000, 1)
        thickness = max(w * h // 300000, 1)
        cv2.line(bgr, (cy, cx - l), (cy, cx + l), color=(0, 255, 0), thickness=thickness)
        cv2.line(bgr, (cy - l, cx), (cy + l, cx), color=(0, 255, 0), thickness=thickness)
    return bgr


def hash_color(semantic):
    r = ((semantic*7+30) % 256).astype(np.uint8)
    g = ((semantic*14+60) % 256).astype(np.uint8)
    b = ((semantic*21+90) % 256).astype(np.uint8)
    return np.stack((r, g, b), axis=-1)


def save_image_frames(frame, path):
    if not os.path.exists(path):
        os.makedirs(path)
    idx = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
    frame_name = f"frame_{idx}"
    np.save(os.path.join(path, frame_name), frame)


def load_image_frames(path):
    num = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
    frames = []
    for i in range(num):
        frame = np.load(os.path.join(path, f"frame_{i}.npy"))
        frames.append(frame)
    return frames


def compute_cosine_similarity(target_embedding, embedding_list):
    target_embedding_tensor = target_embedding.reshape(1, -1)
    # Compute cosine similarity
    similarity_scores = cosine_similarity(target_embedding_tensor, embedding_list)
    return similarity_scores.reshape(-1)


def top_k_indices(scores, k):
    max_len = scores.shape[0]
    k = min(max_len, k)
    indices = np.argsort(scores)[-k:][::-1]
    return list(indices)


if __name__ == "__main__":
    res = rotate_vector_along_axis([3, 5, 0], [4, 4, 1], 1.2)
    print(np.linalg.norm(res) ** 2)
