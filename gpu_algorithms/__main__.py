import argparse
import numpy as np
import coalpy.gpu
import time
from .gpu import prefix_sum
from .gpu import radix_sort

from . import native

def benchmark_prefix_sum(args):
    sample_size = int(args.size)

    #prepare input
    rand_array = np.random.randint(0, high=sample_size, size=sample_size)

    print (":: Prefix Sum ::")

    if args.printresults:
        print("Input: " + str(rand_array))

    benchmark_prefix_sum_numpy(sample_size, rand_array, args)
    benchmark_prefix_sum_cpu(sample_size, rand_array, args)
    benchmark_prefix_sum_gpu(sample_size, rand_array, args)

def benchmark_prefix_sum_gpu(sample_size, sample_array, args):
    print("\t ::: GPU Prefix Sum :::")

    input_buffer = coalpy.gpu.Buffer(
        name="input_buffer", 
        type = coalpy.gpu.BufferType.Standard,
        format = coalpy.gpu.Format.R32_UINT,
        element_count = sample_size,    
        stride = 4 #size of uint
        )

    prefix_sum_args = prefix_sum.allocate_args(sample_size)

    cmd_list = coalpy.gpu.CommandList()
    cmd_list.begin_marker("upload_resource")
    cmd_list.upload_resource( source=sample_array, destination=input_buffer )
    cmd_list.end_marker()

    cmd_list.begin_marker("prefix_sum")
    output_buffer = prefix_sum.run(cmd_list, input_buffer, prefix_sum_args, is_exclusive=False)
    cmd_list.end_marker()

    coalpy.gpu.begin_collect_markers()
    coalpy.gpu.schedule(cmd_list)
    marker_results = coalpy.gpu.end_collect_markers()

    if args.printresults:
        download_request = coalpy.gpu.ResourceDownloadRequest(resource = output_buffer)
        download_request.resolve()
        cpu_result_buffer = np.frombuffer(download_request.data_as_bytearray(), dtype='i')
        cpu_result_buffer = np.resize(cpu_result_buffer, sample_size)
        print("\t Results: " + str(cpu_result_buffer))

    #calculate time stamp markers
    marker_download_request = coalpy.gpu.ResourceDownloadRequest(resource = marker_results.timestamp_buffer)
    marker_download_request.resolve()
    marker_data = np.frombuffer(marker_download_request.data_as_bytearray(), dtype=np.uint64)
    marker_benchmarks = [
        (name, (marker_data[ei]/marker_results.timestamp_frequency -  marker_data[bi]/marker_results.timestamp_frequency) * 1000) for (name, pid, bi, ei) in marker_results.markers]

    (_, ellapsed_time) = marker_benchmarks[1]

    print("\t Elapsed time: " + str(ellapsed_time) + " ms.")
    print();
    return

def benchmark_prefix_sum_cpu(sample_size, sample_array, args):
    print ("\t ::: CPU (C) Prefix Sum :::")
    (time, result) = native.prefix_sum(sample_array)

    if args.printresults:
        array_value = np.frombuffer(result, dtype='i')
        print ("\t Results: " + str(array_value))
    
    print ("\t Elapsed time: " + str(time) + " ms.")
    print();
    return

def benchmark_prefix_sum_numpy(sample_size, sample_array, args):
    print("\t ::: Numpy Prefix Sum :::")
    cpu_start_time = time.time()
    prefix_cpu_result = np.cumsum(sample_array)
    ellapsed_seconds = time.time() - cpu_start_time
    if args.printresults:
        print("\t Result: " + str(prefix_cpu_result))
    print("\t Elapsed time: " + str(ellapsed_seconds * 1000) + " ms.")
    print()
    return


def benchmark_quicksort_numpy(sample_size, rand_array, args):
    print ("\t ::: Numpy Quicksort :::")
    cpu_start_time = time.time()
    sort_result = np.sort(rand_array, axis=-1, kind='quicksort')
    ellapsed_seconds = time.time() - cpu_start_time
    if args.printresults:
        print("\t Results: " + str(sort_result))
    print("\t Elapsed time: " + str(ellapsed_seconds * 1000) + " ms.")
    print()
    return

def benchmark_radixsort_cpu(sample_size, rand_array, args):
    print ("\t ::: CPU (C) Radix Sort :::")
    (time, result) = native.radix_sort(rand_array)

    if args.printresults:
        array_value = np.frombuffer(result, dtype='i')
        print("\t Results: " + str(array_value))

    print("\t Elapsed time: " + str(time) + " ms.")
    print()
    return

def benchmark_radix_sort_gpu(sample_size, sample_array, args):
    print("\t ::: GPU Radix Sort :::")

    input_buffer = coalpy.gpu.Buffer(
        name="input_buffer", 
        type = coalpy.gpu.BufferType.Standard,
        format = coalpy.gpu.Format.R32_UINT,
        element_count = sample_size,    
        stride = 4 #size of uint
        )

    radix_sort_args = radix_sort.allocate_args(sample_size, args.sort_output_ordering)

    cmd_list = coalpy.gpu.CommandList()
    cmd_list.begin_marker("upload_resource")
    cmd_list.upload_resource( source=sample_array, destination=input_buffer )
    cmd_list.end_marker()

    cmd_list.begin_marker("radix_sort")
    (output_buffer, count_table_prefix) = radix_sort.run(cmd_list, input_buffer, radix_sort_args)
    cmd_list.end_marker()

    coalpy.gpu.begin_collect_markers()
    coalpy.gpu.schedule(cmd_list)
    marker_results = coalpy.gpu.end_collect_markers()

    if args.printresults:
        output_download_request = coalpy.gpu.ResourceDownloadRequest(resource = output_buffer)
        output_download_request.resolve()
        cpu_result_buffer = np.frombuffer(output_download_request.data_as_bytearray(), dtype='i')
        cpu_result_buffer = np.resize(cpu_result_buffer, sample_size)
        if args.sort_output_ordering:
            for i in range(0, sample_size):
                cpu_result_buffer[i] = sample_array[cpu_result_buffer[i]]

        print("\t Results: " + str(cpu_result_buffer))

        # uncomment to verify sort
        #for i in range(1, len(cpu_result_buffer)):
        #    if (cpu_result_buffer[i - 1 ] > cpu_result_buffer[i]):
        #        print("ERROR " + str(i))

    #calculate time stamp markers
    marker_download_request = coalpy.gpu.ResourceDownloadRequest(resource = marker_results.timestamp_buffer)
    marker_download_request.resolve()
    marker_data = np.frombuffer(marker_download_request.data_as_bytearray(), dtype=np.uint64)
    marker_benchmarks = [
        (name, (marker_data[ei]/marker_results.timestamp_frequency -  marker_data[bi]/marker_results.timestamp_frequency) * 1000) for (name, pid, bi, ei) in marker_results.markers]

    #print (marker_benchmarks)

    (_, ellapsed_time) = marker_benchmarks[1]

    print("\t Elapsed time: " + str(ellapsed_time) + " ms.")
    print();
    return


def benchmark_sort(args):
    sample_size = int(args.size)

    #prepare input
    rand_array = np.random.randint(0, high=sample_size, size=sample_size)

    print (":: Sort ::")

    if args.printresults:
        print("Input: " + str(rand_array))

    #benchmark_quicksort_numpy(sample_size, rand_array, args)
    #benchmark_radixsort_cpu(sample_size, rand_array, args)
    benchmark_radix_sort_gpu(sample_size, rand_array, args)
    

RAND_SEED_DEFAULT = 1999

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="python -m gpu_algorithms",
        description = "::gpu_algorithms:: - benchmark tool for GPU algorithms")
    parser.add_argument("-s", "--size", default=1600, required=False, help="size of input")
    parser.add_argument("-r", "--randseed", default=RAND_SEED_DEFAULT, required=False, help="random seed")
    parser.add_argument("-p", "--printresults", action='store_true', help="print inputs/outputs")
    parser.add_argument("-g", "--printgpu", action='store_true', help="print the used GPU")
    parser.add_argument("-o", "--sort_output_ordering", default=False, action='store_true', help="sort using extra buffer for keys / indices. Adds sampling cost.")
    args = parser.parse_args()

    if args.printgpu:
        print("Available gpus: " + str(coalpy.gpu.get_adapters()))
        print("Current gpu info: " + str(coalpy.gpu.get_current_adapter_info()))

    rand_seed = int(args.randseed)
    if rand_seed != RAND_SEED_DEFAULT:
        np.random.seed(int(args.randseed))

    #benchmark_prefix_sum(args)
    benchmark_sort(args)
