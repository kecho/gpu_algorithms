#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

PyObject* prefix_sum_python(PyObject* self, PyObject* vargs, PyObject* kwds)
{
    const char* arguments[] = { "list_values", nullptr };
    PyObject* list_values = nullptr;
    if (!PyArg_ParseTupleAndKeywords(vargs, kwds, "O", const_cast<char**>(arguments), &list_values))
        return nullptr;

    if (!PyObject_CheckBuffer(list_values))
        return nullptr;

    Py_buffer buffer_view = {};
    if (PyObject_GetBuffer(list_values, &buffer_view, 0) != 0 || buffer_view.buf == nullptr)
        return nullptr;

    if (buffer_view.itemsize != 4)
        return 0;

    int* input_vals = reinterpret_cast<int*>(buffer_view.buf);
    int len = buffer_view.len/buffer_view.itemsize;
    
    clock_t begin_time = clock();
    int* results = (int*)malloc(buffer_view.len);
    int accumulator = 0;
    for (int i = 0; i < len; ++i)
    {
        accumulator += input_vals[i];
        results[i] = accumulator;
    }
    
    PyBuffer_Release(&buffer_view);

    PyObject* outputObj = PyBytes_FromStringAndSize((const char*)results, buffer_view.len);
    Py_INCREF(outputObj);
    free(results);
    clock_t end_time = clock();
    double timeMilli = ((double)(end_time-begin_time))/(double)(CLOCKS_PER_SEC) * 1000.0;
    return Py_BuildValue("(fO)", timeMilli, outputObj);
}

PyObject* radix_sort_python(PyObject* self, PyObject* vargs, PyObject* kwds)
{
    const char* arguments[] = { "list_values", nullptr };
    PyObject* list_values = nullptr;
    if (!PyArg_ParseTupleAndKeywords(vargs, kwds, "O", const_cast<char**>(arguments), &list_values))
        return nullptr;

    if (!PyObject_CheckBuffer(list_values))
        return nullptr;

    Py_buffer buffer_view = {};
    if (PyObject_GetBuffer(list_values, &buffer_view, 0) != 0 || buffer_view.buf == nullptr)
        return nullptr;

    if (buffer_view.itemsize != 4)
        return 0;

    int* input_vals = reinterpret_cast<int*>(buffer_view.buf);
    int len = buffer_view.len/buffer_view.itemsize;
    
    clock_t begin_time = clock();

    const int bits_per_radix = 8;
    const int bytes_per_radix = bits_per_radix / 8;
    const int count_table_len = (1 << bits_per_radix);

    void* dynamic_memory = malloc(buffer_view.len * 3 + count_table_len * sizeof(int));
    int* ping = (int*)dynamic_memory;
    int* pong = &ping[len];
    int* offset_table = pong + len;
    int* count_table = offset_table + len;

    int* results = nullptr;
    int pass_count = (int)sizeof(int)/bytes_per_radix;
    for (int k = 0; k < pass_count; ++k)
    {
        int* pass_input = k == 0 ? (input_vals) : ((k & 1) == 0 ? ping : pong);
        int* pass_output = results = (k & 1) == 0 ? pong : ping;

        for (int i = 0; i < count_table_len; ++i)
            count_table[i] = 0;

        //counts
        for (int i = 0; i < len; ++i)
        {
            int bucket = (pass_input[i] >> (k * bits_per_radix)) & (count_table_len - 1);
            offset_table[i] = count_table[bucket]++;
        }

        ////prefix_sum
        int accumulator = 0;
        for (int i = 0; i < count_table_len; ++i)
        {
            int v = count_table[i];
            count_table[i] = accumulator;
            accumulator += v;
        }

        //writes
        for (int i = 0; i < len; ++i)
        {
            int value = pass_input[i];
            int bucket = (value >> (k * bits_per_radix)) & (count_table_len - 1);
            pass_output[count_table[bucket] + offset_table[i]] = value;
        }
    }
    
    PyBuffer_Release(&buffer_view);

    PyObject* outputObj = PyBytes_FromStringAndSize((const char*)results, buffer_view.len);
    Py_INCREF(outputObj);
    free(dynamic_memory);
    clock_t end_time = clock();
    double timeMilli = ((double)(end_time-begin_time))/(double)(CLOCKS_PER_SEC) * 1000.0;
    return Py_BuildValue("(fO)", timeMilli, outputObj);
}

static PyMethodDef methods[] = {
    {"prefix_sum", (PyCFunction)prefix_sum_python, METH_VARARGS | METH_KEYWORDS, NULL},
    {"radix_sort", (PyCFunction)radix_sort_python, METH_VARARGS | METH_KEYWORDS, NULL},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "native",
    NULL,
    -1,
    methods,
};

PyMODINIT_FUNC PyInit_native(void)
{
    return PyModule_Create(&module);
}
