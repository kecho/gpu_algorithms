import coalpy.gpu as g
import math

g_clear_target_shader = g.Shader(file = "clear_target.hlsl", name = "clear", main_function = "csMainClear" )
g_clear_uint_buffer_shader = g.Shader(file = "clear_target.hlsl", name = "clear", main_function = "csMainClearUintBuffer" )

def clear_texture(cmd_list, color, texture, w, h):
    cmd_list.dispatch(
        shader = g_clear_target_shader,
        constants = color,
        x = math.ceil(w / 8), 
        y = math.ceil(h / 8),  
        z = 1,
        outputs = texture)

def clear_uint_buffer(cmd_list, clear_val, buff, el_offset, el_count):
    cmd_list.dispatch(
        shader = g_clear_uint_buffer_shader,
        constants = [int(clear_val), int(el_offset), int(el_count)],
        outputs = buff,
        x = math.ceil(el_count / 64),
        y = 1,
        z = 1)


def divup(a, b):
    return int((a + b - 1)/b)

def alignup(a, b):
    return divup(a, b) * b
