import coalpy.gpu as g
from . import utilities as utils

g_group_size = 128
g_prefix_sum_group = g.Shader(file = "prefix_sum.hlsl", main_function = "csPrefixSumOnGroup")
g_prefix_sum_group_exclusive = g.Shader(file = "prefix_sum.hlsl", main_function = "csPrefixSumOnGroup", defines = ["EXCLUSIVE_PREFIX"])
g_prefix_sum_next_input = g.Shader(file = "prefix_sum.hlsl", main_function = "csPrefixSumNextInput")
g_prefix_sum_resolve_parent = g.Shader(file = "prefix_sum.hlsl", main_function = "csPrefixSumResolveParent")
g_prefix_sum_resolve_parent_exclusive = g.Shader(file = "prefix_sum.hlsl", main_function = "csPrefixSumResolveParent", defines = ["EXCLUSIVE_PREFIX"])

def allocate_args(input_counts):
    aligned_bin_count = utils.alignup(input_counts, g_group_size)
    reduction_count = 0
    c = input_counts
    perform_reduction = True
    while perform_reduction:
        reduction_count += utils.alignup(c, g_group_size)
        c = utils.divup(c, g_group_size)
        perform_reduction = c > 1

    return (g.Buffer(name = "reductionBufferInput", element_count = aligned_bin_count, format = g.Format.R32_UINT),
            g.Buffer(name = "reductionBufferOutput", element_count = reduction_count, format = g.Format.R32_UINT),
            input_counts)

def run(cmd_list, input_buffer, prefix_sum_args, is_exclusive = False, input_counts = -1):
    reduction_buffer_in = prefix_sum_args[0]
    reduction_buffer_out = prefix_sum_args[1]
    if (input_counts == -1):
        input_counts = prefix_sum_args[2]
    group_count = input_counts
    perform_reduction = input_counts > 0 
    iteration = 0
    input_count = 0
    input_offset = 0
    output_offset = 0
    pass_list = []
    while perform_reduction:
        input_count = group_count
        print (input_count)
        group_count = utils.divup(group_count, g_group_size)
        pass_list.append((input_count, output_offset))

        cmd_list.dispatch(
            x = group_count, y = 1, z = 1,
            shader = g_prefix_sum_group_exclusive if is_exclusive and iteration == 0 and group_count == 1 else g_prefix_sum_group,           
            inputs = input_buffer if iteration == 0 else reduction_buffer_in,
            outputs = reduction_buffer_out,
            constants = [input_count, 0, output_offset, 0])

        perform_reduction = group_count > 1
        if perform_reduction:
            next_group_count = utils.divup(group_count, g_group_size)
            cmd_list.dispatch(
                x = next_group_count, y = 1, z = 1,
                shader = g_prefix_sum_next_input,
                inputs = reduction_buffer_out,
                outputs = reduction_buffer_in,
                constants = [0, output_offset, 0, 0])

        iteration += 1
        output_offset += utils.alignup(input_count, g_group_size)

    for i in range(1, len(pass_list)):
        idx = len(pass_list) - 1 - i
        (parent_count, parent_offset) = pass_list[idx + 1]
        (count, offset) = pass_list[idx]
        const = [0, 0, offset, parent_offset]
        if i == len(pass_list) - 1 and is_exclusive:
            cmd_list.dispatch(
                x = utils.divup(count, g_group_size), y = 1, z = 1,
                shader = g_prefix_sum_resolve_parent_exclusive,
                inputs = input_buffer,
                outputs = reduction_buffer_out,
                constants = const)
        else:
            cmd_list.dispatch(
                x = utils.divup(count, g_group_size), y = 1, z = 1,
                shader = g_prefix_sum_resolve_parent,
                outputs = reduction_buffer_out,
                constants = const)
    return reduction_buffer_out
