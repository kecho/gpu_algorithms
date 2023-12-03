import coalpy.gpu as g
from . import utilities as utils

g_group_size = 64
g_batch_size = 128
g_bits_per_radix = 8
g_bytes_per_radix = int(g_bits_per_radix/8)
g_radix_counts = int(1 << g_bits_per_radix)
g_radix_iterations = int(32/g_bits_per_radix)

g_count_scatter_shader = g.Shader(file = "radix_sort.hlsl", main_function = "csCountScatterBuckets")
g_prefix_count_table_shader = g.Shader(file = "radix_sort.hlsl", main_function = "csPrefixCountTable", defines = ["GROUP_SIZE=256"])
g_prefix_global_table_shader = g.Shader(file = "radix_sort.hlsl", main_function = "csPrefixGlobalTable", defines = ["GROUP_SIZE=RADIX_COUNTS"])
g_scatter_output_shader = g.Shader(file = "radix_sort.hlsl", main_function = "csScatterOutput", defines=["GROUP_SIZE="+str(g_batch_size)])

def allocate_args(input_counts):
    aligned_batch_count = utils.alignup(input_counts, g_batch_size)
    count_table_count = aligned_batch_count * g_radix_counts
    return (
        g.Buffer(name="localOffsets", element_count = input_counts, format = g.Format.R32_UINT),
        g.Buffer(name="outputBuffer", element_count = input_counts, format = g.Format.R32_UINT),
        g.Buffer(name="countTableBatchPrefixBuffer", element_count = count_table_count, format = g.Format.R32_UINT),
        g.Buffer(name="radixTotalCounts", element_count = g_radix_counts, format = g.Format.R32_UINT),
        g.Buffer(name="countTableBuffer", element_count = count_table_count, format = g.Format.R32_UINT),
        input_counts)

def run (cmd_list, input_buffer, sort_args):
    (local_offsets, output_buffer, count_table_prefix, radix_total_counts, count_table, input_counts) = sort_args
    batch_counts = utils.divup(input_counts, g_batch_size)

    radix_i = 0
    radix_mask = int((1 << g_bits_per_radix) - 1)
    radix_shift = g_bits_per_radix * radix_i
    
    cmd_list.dispatch(
        x = batch_counts, y = 1, z = 1,
        shader = g_count_scatter_shader,
        inputs = input_buffer,
        outputs = [ local_offsets, count_table ],
        constants = [
            int(input_counts), batch_counts, radix_mask, radix_shift,
            g_batch_size, int(0), int(0), int(0) ]
    )

    cmd_list.dispatch(
        x = int(g_radix_counts), y = 1, z = 1,
        shader = g_prefix_count_table_shader,
        inputs = count_table,
        outputs = [count_table_prefix, radix_total_counts],
        constants = [
            int(batch_counts), 0, 0, 0,
            0, 0, 0, 0 ]
    )

    cmd_list.dispatch(
        x = 1, y = 1, z = 1,
        shader = g_prefix_global_table_shader,
        inputs = radix_total_counts,
        outputs = count_table
    )

    cmd_list.dispatch(
        x = batch_counts, y = 1, z = 1,
        shader = g_scatter_output_shader,
        inputs = [input_buffer, local_offsets, count_table_prefix, count_table ],
        outputs = output_buffer,
        constants = [
            int(input_counts), batch_counts, radix_mask, radix_shift,
            g_batch_size, int(0), int(0), int(0) ]
    )

    return (local_offsets, output_buffer, count_table, count_table_prefix)
