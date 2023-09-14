import os
from shell_utils import rm_dir
import tsutils as ts

ts.stop_torchserve()
dirpath = os.path.dirname(__file__)
# clean up the logs folder to reset logs before the next run
# TODO - To reduce logs from taking a lot of storage it is being cleared everytime it is stopped
# Understand on how this can be handled better by rolling file approach
rm_dir(os.path.join(dirpath, 'gen', 'logs'))
# clean up the entire generate folder
rm_dir(os.path.join(dirpath, 'gen'))