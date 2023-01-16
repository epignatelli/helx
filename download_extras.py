import logging
import os
import platform
import tarfile
import rarfile

import requests


logging.basicConfig(level=logging.INFO)


MUJOCO_ROOT = os.path.join(os.path.expanduser("~"), ".mujoco")


def _download_url(url, out_path, chunk_size=128):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    r = requests.get(url, stream=True)
    logging.info("Downloading {} into {}".format(url, out_path))
    with open(out_path, "wb") as fd:
        chunks = 0
        for chunk in r.iter_content(chunk_size=chunk_size):
            chunks += 1
            fd.write(chunk)


def _download_mujoco210():
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
        raise ValueError("Unsupported system: {}".format(system))
    mujoco_system = mujoco_systems[system]

    # get architecture version
    machine = platform.machine().lower()
    if machine != "x86_64":
        raise ValueError("Unsupported architecture: {}".format(machine))

    # download mujoco
    url = os.path.join(
        base_url, "{}-{}-{}.tar.gz".format(mujoco_version, mujoco_system, machine)
    )
    tar_path = os.path.join(MUJOCO_ROOT, os.path.basename(url))
    _download_url(url, tar_path)

    # unzip mujoco
    with tarfile.open(tar_path, "r:gz") as tar_ref:
        tar_ref.extractall(MUJOCO_ROOT)
    os.remove(tar_path)

    # install key
    key_url = "https://www.roboti.us/file/mjkey.txt"
    key_filepath = os.path.join(MUJOCO_ROOT, "mjkey.txt")
    _download_url(key_url, key_filepath)


def _download_mujoco_dm_control():
    if os.path.exists(MUJOCO_ROOT) and os.path.exists(os.path.join(MUJOCO_ROOT, "bin")):
        return
    systems = {
        "Windows": "windows",
        "Linux": "linux",
        "Darwin": "macos-universal2",
    }
    url = "https://github.com/deepmind/mujoco/releases/download/2.3.1/mujoco-2.3.1-{}-{}.tar.gz"
    system = platform.system()
    if system not in systems:
        raise ValueError("Unsupported system: {}".format(system))
    url = url.format(systems[system])
    if system != "Darwin":
        url = url.format(platform.machine().lower())
    _download_url(url, MUJOCO_ROOT)


def _download_atari_roms():
    if os.path.exists(os.path.join(os.path.expanduser("~"), ".atari", "roms")):
        return
    url = "http://www.atarimania.com/roms/Roms.rar"
    out_path = os.path.join(os.path.expanduser("~"), ".atari", "roms.rar")
    # downlaod the file
    _download_url(url, out_path)

    # extract the file
    with rarfile.RarFile(out_path) as rar_ref:
        rar_ref.extractall(os.path.dirname(out_path))
    os.remove(out_path)


if __name__ == "__main__":
    _download_mujoco210()
    # _download_mujoco_dm_control()
    _download_atari_roms()
