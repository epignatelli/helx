import os

MUJOCO_ROOT = os.path.join(os.path.expanduser("~"), ".mujoco")
MUJOCO_VERSION = "mujoco210"


def configure_mujoco_paths():
    # add mujoco to path
    mujoco_path =  os.path.join(MUJOCO_ROOT, MUJOCO_VERSION)
    mujoco_lib = os.path.join(mujoco_path, "lib")
    mujoco_so = os.path.join(mujoco_path, f"lib{MUJOCO_VERSION}.so")

    # add mujoco lib to LD_LIBRARY_PATH
    if mujoco_lib not in os.environ["LD_LIBRARY_PATH"]:
        os.environ["LD_LIBRARY_PATH"] += ":" + mujoco_lib

    # add mujoco bin to MJLIB_PATH
    if not os.environ.get("MJLIB_PATH"):
        os.environ["MJLIB_PATH"] = mujoco_so

    # add mujoco path to MUJOCO_PY_MUJOCO_PATH
    if not os.environ.get("MUJOCO_PY_MUJOCO_PATH"):
        os.environ["MUJOCO_PY_MUJOCO_PATH"] = mujoco_path

    # set mujoco GL
    if not os.environ.get("MUJOCO_GL"):
        os.environ["MUJOCO_GL"] = "egl"

    # set mujoco PYOPENGL_PLATFORM
    if not os.environ.get("PYOPENGL_PLATFORM"):
        os.environ["PYOPENGL_PLATFORM"] = "egl"


def configure_all():
    configure_mujoco_paths()
