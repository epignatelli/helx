import logging
import os
import platform
import tarfile
import subprocess

import requests

logging.basicConfig(level=logging.INFO)


MUJOCO_ROOT = os.path.join(os.path.expanduser("~"), ".mujoco")
ATARI_ROOT = os.path.join(os.path.expanduser("~"), ".atari")


def unrar(file_path, out_dir, remove_after=True, override=False):
    command = "unrar {} x {} {}".format("-f" if override else "", file_path, out_dir)
    try:
        subprocess.call(command, shell=True)
    except Exception as e:
        logging.error("Error while extracting rar file: {}. \
        `unrar` is currently a pre-requisite".format(e))
    else:
        if remove_after:
            os.remove(file_path)
    return


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
    out_filename = os.path.basename(url)
    out_path = os.path.join(MUJOCO_ROOT, out_filename)
    out_dir = os.path.splitext(out_path)[0]
    if os.path.exists(out_dir) and len(os.listdir(out_dir)) > 0:
        return

    _download_url(url, out_path)

    # unzip mujoco
    with tarfile.open(out_path, "r:gz") as tar_ref:
        tar_ref.extractall(MUJOCO_ROOT)
    os.remove(out_path)

    # install key
    key_url = "https://www.roboti.us/file/mjkey.txt"
    key_filepath = os.path.join(MUJOCO_ROOT, "mjkey.txt")
    _download_url(key_url, key_filepath)


def _download_mujoco_dm_control():
    # example url: https://github.com/deepmind/mujoco/releases/download/2.3.1/mujoco-2.3.1-linux-x86_64.tar.gz

    # get operating system
    systems = {
        "Windows": "windows",
        "Linux": "linux",
        "Darwin": "macos-universal2",
    }
    system = platform.system()
    if system not in systems:
        raise ValueError("Unsupported system: {}".format(system))

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
        raise ValueError("Unsupported architecture: {}".format(arch))

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
    with tarfile.open(out_path, "r:gz") as tar_ref:
        tar_ref.extractall(MUJOCO_ROOT)
    os.remove(out_path)



def _download_atari_roms():
    out_path = os.path.join(ATARI_ROOT, "roms.rar")
    out_dir = os.path.dirname(out_path)
    if os.path.exists(out_dir) and len(os.listdir(out_dir)) > 0:
        return

    # downlaod the file
    url = "http://www.atarimania.com/roms/Roms.rar"
    _download_url(url, out_path)

    # extract the file
    unrar(out_path, out_dir, remove_after=True)

    # try import the roms
    try:
        subprocess.call("ale-import-roms {}".format(out_dir), shell=True)
    except Exception as e:
        msg = "ALE ROMs have been downloaded and extracted but there \
        was an error with ale-py while importing roms: {}. \
        Please install helx first and try download the extra requirements again".format(e)
        logging.error(msg)


if __name__ == "__main__":
    _download_mujoco210()
    _download_mujoco_dm_control()
    _download_atari_roms()
