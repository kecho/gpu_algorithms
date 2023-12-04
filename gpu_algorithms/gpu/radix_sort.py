import coalpy.gpu as g
from . import utilities as utils

g_group_size = 64
g_batch_size = 1024
g_bits_per_radix = 8
g_bytes_per_radix = int(g_bits_per_radix/8)
g_radix_counts = int(1 << g_bits_per_radix)
g_radix_iterations = int(32/g_bits_per_radix)

g_count_scatter_shader = g.Shader(file = "radix_sort.hlsl", main_function = "csCountScatterBuckets")
g_prefix_count_table_shader = g.Shader(file = "radix_sort.hlsl", main_function = "csPrefixCountTable", defines = ["GROUP_SIZE=256"])
g_prefix_global_table_shader = g.Shader(file = "radix_sort.hlsl", main_function = "csPrefixGlobalTable", defines = ["GROUP_SIZE=RADIX_COUNTS"])
g_scatter_output_shader = g.Shader(file = "radix_sort.hlsl", main_function = "csScatterOutput", defines=["GROUP_SIZE="+str(g_batch_size)])

def allocate_args(input_counts):
    aligned_batch_count = utils.divup(input_counts, g_batch_size)
    count_table_count = aligned_batch_count * g_radix_counts
    return (
        g.Buffer(name="localOffsetsBuffer", element_count = input_counts, format = g.Format.R32_UINT),
        g.Buffer(name="pingBuffer", element_count = input_counts, format = g.Format.R32_UINT),
        g.Buffer(name="pongBuffer", element_count = input_counts, format = g.Format.R32_UINT),
        g.Buffer(name="countTableBatchPrefixBuffer", element_count = count_table_count, format = g.Format.R32_UINT),
        g.Buffer(name="radixTotalCounts", element_count = g_radix_counts, format = g.Format.R32_UINT),
        g.Buffer(name="countTableBuffer", element_count = count_table_count, format = g.Format.R32_UINT),
        input_counts)

def run (cmd_list, input_buffer, sort_args):
    (local_offsets, ping_buffer, pong_buffer, count_table_prefix, radix_total_counts, count_table, input_counts) = sort_args
    batch_counts = utils.divup(input_counts, g_batch_size)

    radix_mask = int((1 << g_bits_per_radix) - 1)

    tmp_input_buffer = ping_buffer
    tmp_output_buffer = pong_buffer

    for radix_i in range(0, g_radix_iterations):
        radix_shift = g_bits_per_radix * radix_i

        (tmp_input_buffer, tmp_output_buffer) = (tmp_output_buffer, tmp_input_buffer)
        unsorted_buffer = input_buffer if radix_i == 0 else tmp_input_buffer

        cmd_list.dispatch(
            x = batch_counts, y = 1, z = 1,
            shader = g_count_scatter_shader,
            inputs = unsorted_buffer,
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
            inputs = [unsorted_buffer, local_offsets, count_table_prefix, count_table ],
            outputs = tmp_output_buffer,
            constants = [
                int(input_counts), batch_counts, radix_mask, radix_shift,
                g_batch_size, int(0), int(0), int(0) ]
        )

    return tmp_output_buffer
