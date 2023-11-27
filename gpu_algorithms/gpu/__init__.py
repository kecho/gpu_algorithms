import os
import sys
import pathlib
import coalpy.gpu as g

def _checkGpu(gpuInfo, substring):
    (idx, nm) = gpuInfo
    return substring in nm.lower()

selected_gpu = next((adapter for adapter in g.get_adapters() if _checkGpu(adapter, "nvidia") or _checkGpu(adapter, "amd")), None)
if selected_gpu is not None:
    print ("Setting gpu %d" % selected_gpu[0] )
    g.get_settings().adapter_index = selected_gpu[0]

g_module_path = os.path.dirname(pathlib.Path(sys.modules[__name__].__file__)) + "\\"
g.add_data_path(g_module_path)
g.init()
print(g.get_current_adapter_info())
