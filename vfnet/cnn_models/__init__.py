import os
import sys

local_path = os.path.dirname(__file__)
if local_path not in sys.path:
    sys.path.append(local_path)
local_path_up = os.path.dirname(local_path)
if local_path_up not in sys.path:
    sys.path.append(local_path_up)