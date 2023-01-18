import logging
import os
import platform
import subprocess
import tarfile

import requests
from setuptools import find_packages, setup
from setuptools.command.install import install

logger = logging.getLogger("helx setup")
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
BOLD = '\033[1m'
DARK_GREY = "\033[1;30m"
END = '\033[0m'
ch.setFormatter(logging.Formatter(BOLD + DARK_GREY + '%(asctime)s | %(name)s | %(levelname)s | %(message)s' + END))
logger.addHandler(ch)


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REQUIREMENTS_PATH = "requirements.txt"
MUJOCO_ROOT = os.path.join(os.path.expanduser("~"), ".mujoco")
ATARI_ROOT = os.path.join(os.path.expanduser("~"), ".atari")


class PostInstall(install):
    def run(self):
        install.run(self)

        # download engines
        logger.info("Installing extra requirements: mujoco, atari.")
        _download_mujoco210()
        _download_mujoco_dm_control()
        # TODO(epignatelli): switch to autorom when pipeline is fixed
        _download_atari_roms()


def _download_url(url, out_path, chunk_size=128):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    r = requests.get(url, stream=True)
    logger.info("Downloading {} into {}".format(url, out_path))
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
    url = "https://roms8.s3.us-east-2.amazonaws.com/Roms.tar.gz"
    _download_url(url, out_path)

    # extract the file
    with tarfile.open(out_path, "r:gz") as tar_ref:
        tar_ref.extractall(out_dir)

    # try import the roms
    try:
        subprocess.call("ale-import-roms {}".format(out_dir), shell=True)
    except Exception as e:
        msg = "ALE ROMs have been downloaded and extracted but there \
        was an error with ale-py while importing roms: {}. \
        Please install helx first and try download the extra requirements again".format(
            e
        )
        logger.error(msg)


def _get_version():
    filepath = os.path.join(CURRENT_DIR, "helx", "__init__.py")
    with open(filepath) as f:
        for line in f:
            if line.startswith("__version__") and "=" in line:
                version = line[line.find("=") + 1 :].strip(" '\"\n")
                if version:
                    return version
    raise ValueError("`__version__` not defined in {}".format(filepath))


def _parse_requirements():
    filepath = os.path.join(CURRENT_DIR, REQUIREMENTS_PATH)
    with open(filepath) as f:
        return [
            line.rstrip() for line in f if not (line.isspace() or line.startswith("#"))
        ]


setup(
    name="Helx",
    version=_get_version(),
    description="Helx is a helper library for Reinforcement Learning for JAX",
    author="Eduardo Pignatelli",
    author_email="edu.pignatelli@gmail.com",
    url="https://github.com/epignatelli/helx",
    packages=find_packages(exclude=["experiments", "test", "examples"]),
    python_requires=">=3.9",
    install_requires=_parse_requirements(),
    cmdclass={"install": PostInstall, "develop": PostInstall},
)
