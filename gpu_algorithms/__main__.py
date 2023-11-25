import argparse
import numpy as np
import coalpy.gpu
import time
from .gpu import prefix_sum

def benchmark_prefix_sum(sample_size):
    #prepare input
    rand_array = np.random.randint(0, high=sample_size, size=sample_size)

    input_buffer = coalpy.gpu.Buffer(
        name="input_buffer", 
        type = coalpy.gpu.BufferType.Standard,
        format = coalpy.gpu.Format.R32_UINT,
        element_count = sample_size,    
        stride = 4 #size of uint
        )

    prefix_sum_tmp_args = prefix_sum.allocate_args(sample_size)

    cmd_list = coalpy.gpu.CommandList()
    cmd_list.begin_marker("upload_resource")
    cmd_list.upload_resource( source=rand_array, destination=input_buffer )
    cmd_list.end_marker()

    cmd_list.begin_marker("prefix_sum")
    output_buffer = prefix_sum.run(cmd_list, input_buffer, prefix_sum_tmp_args, is_exclusive=True)
    cmd_list.end_marker()

    coalpy.gpu.begin_collect_markers()
    coalpy.gpu.schedule(cmd_list)
    marker_results = coalpy.gpu.end_collect_markers()

    #download_request = coalpy.gpu.ResourceDownloadRequest(resource = output_buffer)
    #download_request.resolve()
    #cpu_result_buffer = np.frombuffer(download_request.data_as_bytearray(), dtype='i')
    #cpu_result_buffer = np.resize(cpu_result_buffer, sample_size)

    #calculate time stamp markers
    marker_download_request = coalpy.gpu.ResourceDownloadRequest(resource = marker_results.timestamp_buffer)
    marker_download_request.resolve()
    marker_data = np.frombuffer(marker_download_request.data_as_bytearray(), dtype=np.uint64)
    marker_benchmarks = [
        (name, (marker_data[ei]/marker_results.timestamp_frequency -  marker_data[bi]/marker_results.timestamp_frequency) * 1000) for (name, pid, bi, ei) in marker_results.markers]

    print(marker_benchmarks)
    #print(cpu_result_buffer)

    cpu_start_time = time.time()
    prefix_cpu_result = np.cumsum(rand_array)
    ellapsed_seconds = time.time() - cpu_start_time

    print(ellapsed_seconds * 1000)
    
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="python -m gpu_algorithms",
        description = "::gpu_algorithms:: - benchmark tool for GPU algorithms")
    parser.add_argument("-s", "--size", default=1600, required=False, help="size of input")
    args = parser.parse_args()
    benchmark_prefix_sum(int(args.size))
