import coalpy.gpu as g
from . import utilities as utils

g_group_size = 64
g_batch_size = 128
g_bits_per_radix = 8
g_bytes_per_radix = g_bits_per_radix/8
g_radix_counts = 32 / g_bits_per_radix

def allocate_args(input_counts):
    aligned_batch_count = utils.alignup(input_counts, g_batch_size)
    perform_reduction = True
    
    c = aligned_batch_count
    count_table_count = 0

    while perform_reduction:
        count_table_count += c
        c = utils.divup(c, g_group_size)
        perform_reduction = c > 1

    count_table_count = count_table_count * g_radix_counts
    return (
        g.Buffer(name="localOffsets", element_count = input_counts, format = g.Format.R32_UINT),
        g.Buffer(name="pingBuffer", element_count = input_counts, format = g.Format.R32_UINT),
        g.Buffer(name="pongBuffer", element_count = input_counts, format = g.Format.R32_UINT),
        g.Buffer(name="countTableBuffer", element_count = count_table_count, format = g.Format.R32_UINT),
        input_counts)

def run (cmd_list, input_buffer, sort_args):
    (local_offsets, ping_buffer, pong_buffer, count_table) = sort_args

    for r in range(0, g_radix_counts):
        
    
