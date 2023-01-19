import os


def get_version():
    current_dir = os.path.dirname(__file__)
    filepath = os.path.join(current_dir, "..", "VERSION")
    with open(filepath) as f:
        return f.read().strip()
