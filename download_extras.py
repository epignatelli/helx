import os
import platform

import requests
import zipfile
import logging


MUJOCO_ROOT = os.path.join(os.path.expanduser("~"), ".mujoco")


def _download_url(url, out_path, chunk_size=128):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    r = requests.get(url, stream=True)
    with open(out_path, "wb") as fd:
        chunks = 0
        for chunk in r.iter_content(chunk_size=chunk_size):
            print("Downloading chunk {}".format(chunks), end="\t\t\t\t\r")
            chunks += 1
            fd.write(chunk)


def _download_mujoco_openai():
    # check if mujoco is already installed
    mujoco_version = "mujoco200"
    mujoco_path = os.path.join(MUJOCO_ROOT, mujoco_version)
    if os.path.exists(os.path.join(mujoco_path, "bin")):
        return

    # get system version
    muojoco_systems = {
        "Windows": "win64",
        "Linux": "linux",
        "Darwin": "macos",
    }
    system = platform.system()
    if system not in muojoco_systems:
        raise ValueError("Unsupported system: {}".format(system))

    # download mujoco
    system = muojoco_systems[system]
    url = "https://www.roboti.us/download/{}_{}.zip".format(mujoco_version, system)
    mujoco_zip_filepath = os.path.join(MUJOCO_ROOT, os.path.basename(url))
    logging.info(
        "Downloading mujoco version {} into {}".format(url, mujoco_zip_filepath)
    )
    _download_url(url, mujoco_zip_filepath)

    # unzip mujoco
    logging.info(
        "Unzipping {} into {}".format(
            mujoco_zip_filepath, MUJOCO_ROOT
        )
    )
    with zipfile.ZipFile(mujoco_zip_filepath, "r") as zip_ref:
        zip_ref.extractall(MUJOCO_ROOT)
    path_before = os.path.join(MUJOCO_ROOT, "{}_{}".format(mujoco_version, system))
    path_after = os.path.join(MUJOCO_ROOT, "{}".format(mujoco_version))
    os.rename(path_before, path_after)
    os.remove(mujoco_zip_filepath)

    # install key
    key_url = "https://www.roboti.us/file/mjkey.txt"
    key_filepath = os.path.join(MUJOCO_ROOT, "mjkey.txt")
    _download_url(key_url, key_filepath)
    return


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


if __name__ == "__main__":
    _download_mujoco_openai()
    # _download_mujoco_dm_control()
