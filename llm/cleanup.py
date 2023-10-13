"""
This module is used for cleanup. It stops Torchserve if it is running and
deletes all temporary files and directories generated during the run.

Attributes:
    dirpath (str): Stores parent directory of module
"""
import os
from utils.shell_utils import rm_dir
import utils.tsutils as ts

ts.stop_torchserve()
dirpath = os.path.dirname(__file__)
# clean up the logs folder to reset logs before the next run
rm_dir(os.path.join(dirpath, "utils", "gen", "logs"))
# clean up the entire generate folder
rm_dir(os.path.join(dirpath, "utils", "gen"))
