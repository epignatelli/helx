#!python
# Copyright [2023] The Helx Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import logging
import os
import platform
import subprocess
import tarfile

import requests

from .logging import get_logger

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REQUIREMENTS_PATH = "requirements.txt"
MUJOCO_ROOT = os.path.join(os.path.expanduser("~"), ".mujoco")
ATARI_ROOT = os.path.join(os.path.expanduser("~"), ".atari")


logging = get_logger()


def _download_url(url, out_path, chunk_size=128):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    r = requests.get(url, stream=True)
    logging.info(f"Downloading {url} into {out_path}")
    with open(out_path, "wb") as fd:
        chunks = 0
        for chunk in r.iter_content(chunk_size=chunk_size):
            chunks += 1
            fd.write(chunk)


def download_mujoco210():
    # example_url: "https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz"
    base_url = "https://mujoco.org/download"
    mujoco_version = "mujoco210"

    # get system version
    mujoco_systems = {
        "Windows": "win64",
        "Linux": "linux",
        "Darwin": "macos",
    }
    system = platform.system()
    if system not in mujoco_systems:
        raise ValueError(f"Unsupported system: {system}")
    mujoco_system = mujoco_systems[system]

    # get architecture version
    machine = platform.machine().lower()
    if machine != "x86_64":
        raise ValueError(f"Unsupported architecture: {machine}")

    # download mujoco
    url = os.path.join(base_url, f"{mujoco_version}-{mujoco_system}-{machine}.tar.gz")
    out_filename = os.path.basename(url)
    out_path = os.path.join(MUJOCO_ROOT, out_filename)
    out_dir = os.path.splitext(out_path)[0]
    if os.path.exists(out_dir) and len(os.listdir(out_dir)) > 0:
        return

    _download_url(url, out_path)

    # unzip mujoco
    logging.info(f"Extracting {out_path} into {MUJOCO_ROOT}")
    with tarfile.open(out_path, "r:gz") as tar_ref:
        tar_ref.extractall(MUJOCO_ROOT)
    os.remove(out_path)

    # install key
    key_url = "https://www.roboti.us/file/mjkey.txt"
    key_filepath = os.path.join(MUJOCO_ROOT, "mjkey.txt")
    _download_url(key_url, key_filepath)


def download_mujoco_dm_control():
    # example url: https://github.com/deepmind/mujoco/releases/download/2.3.1/mujoco-2.3.1-linux-x86_64.tar.gz

    # get operating system
    systems = {
        "Windows": "windows",
        "Linux": "linux",
        "Darwin": "macos-universal2",
    }
    system = platform.system()
    if system not in systems:
        raise ValueError(f"Unsupported system: {system}")

    # get architecture
    architectures = {
        "x86_64": "x86_64",
        "AMD64": "x86_64",
        "i386": "x86_32",
        "i686": "x86_32",
        "aarch64": "aarch64",
    }
    arch = platform.machine().lower()
    if arch not in architectures:
        raise ValueError(f"Unsupported architecture: {arch}")

    # download mujoco
    url = "https://github.com/deepmind/mujoco/releases/download/2.3.1/mujoco-2.3.1-{}-{}.tar.gz"
    url = url.format(systems[system], arch)
    out_filename = os.path.basename(url)
    out_path = os.path.join(MUJOCO_ROOT, out_filename)
    out_dir = os.path.join(MUJOCO_ROOT, os.path.splitext(out_filename)[0])
    if os.path.exists(out_dir) and len(os.listdir(out_dir)) > 0:
        return
    _download_url(url, out_path)

    # untar mujoco
    logging.info(f"Extracting {out_path} into {MUJOCO_ROOT}")
    with tarfile.open(out_path, "r:gz") as tar_ref:
        tar_ref.extractall(MUJOCO_ROOT)
    os.remove(out_path)


def download_atari_roms():
    out_path = os.path.join(ATARI_ROOT, "roms.rar")
    out_dir = os.path.dirname(out_path)
    if os.path.exists(out_dir) and len(os.listdir(out_dir)) > 0:
        return

    # downlaod the file
    url = "https://roms8.s3.us-east-2.amazonaws.com/Roms.tar.gz"
    _download_url(url, out_path)

    # extract the file
    logging.info(f"Extracting {out_path} into {out_dir}")
    with tarfile.open(out_path, "r:gz") as tar_ref:
        tar_ref.extractall(out_dir)

    # try import the roms
    try:
        subprocess.call(f"ale-import-roms {out_dir}", shell=True)
    except Exception as e:
        msg = f"ALE ROMs have been downloaded and extracted but there \
        was an error with ale-py while importing roms: {e}. \
        Please install helx first and try download the extra requirements again"
        logging.error(msg)


def download_all():
    download_atari_roms()
    download_mujoco210()
    download_mujoco_dm_control()
