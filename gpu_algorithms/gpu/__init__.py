import os
import sys
import pathlib
import coalpy.gpu as g

g_module_path = os.path.dirname(pathlib.Path(sys.modules[__name__].__file__)) + "\\"
g.add_data_path(g_module_path)
g.init()
