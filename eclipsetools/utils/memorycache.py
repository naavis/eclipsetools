import os
import sys

from joblib import Memory

_main_script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
_cache_dir = os.path.join(_main_script_dir, '.cache')
os.makedirs(_cache_dir, exist_ok=True)

memory = Memory(_cache_dir, verbose=0)
