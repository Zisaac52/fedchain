import os
import sys

from munch import DefaultMunch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from config import my_conf as global_conf


class Configurator:
    def __init__(self):
        # Create an isolated munch instance so downstream code can mutate
        # fields (e.g., override dataset) without touching the shared dict.
        self._config = DefaultMunch.fromDict(global_conf)

    def get_config(self):
        return self._config
