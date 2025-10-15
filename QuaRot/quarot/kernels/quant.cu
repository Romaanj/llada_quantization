#include <quant.h>


template<typename T>
__device__ __half int_to_half(T value) // __device__ indicates that this function can only be called from device code
{
    return __int2half_rn(static_cast<int>(value));
}


__global__ // __global__ indicates that this function can be called from host code and runs on the device
void sym_quantize_f16_i4_kernel(
        const half *__restrict__ x, // __restrict__: indicates that the pointer does not alias with any other pointer
        const half *__restrict__ scale,
        uint32_t rows, // 矩阵的行数。
        uint32_t colsSrc, // 输入矩阵 x 的列数。
        uint32_t colsDst, // 量化后矩阵 q 的列数（注意：由于是 int4，一个存储单位表示多个元素，所以 colsDst < colsSrc）。
        Int4Storage *__restrict__ q 
)
{
    uint32_t row = threadIdx.y + blockIdx.y * blockDim.y;
    uint32_t colDst = threadIdx.x + blockIdx.x * blockDim.x;
    if (row >= rows || colDst * kElementsPerVector >= colsSrc)
    {
        return;
    }
    Int4Storage storage;
    memset(&storage, 0, sizeof(storage));
    uint32_t id = colDst * kElementsPerVector + row * colsSrc;
#pragma unroll
    for (int i = 0; i < kElementsPerVector; ++i)
    {
        bool safe = (colDst * kElementsPerVector + i) < colsSrc;
        if (safe)
        {
            half data = __hdiv(x[id + i], scale[row]);

            int qval = clamp(__half2int_rn(data), qmin, qmax);
            Int4Subbyte{reinterpret_cast<cutlass::int4b_t *>(&storage), i}.set(
                    qval);
                    // T* new_ptr = reinterpret_cast<T*>(old_ptr);

                    // MySubbyteReference(Element *ptr,   /// pointer to memory 
                    // int64_t offset  /// logical offset in units of Element
                    // )

                    // MySubbyteReference &set(Element const &x)
        }
    }

    q[colDst + row * colsDst] = storage;
}


void sym_quant_host(
        const half *x,
        const half *scale,
        uint32_t rows,
        uint32_t colsSrc,
        uint32_t colsDst,
        Int4Storage *q
)
{

    dim3 block{std::min<uint32_t>(colsDst, 32), std::min<uint32_t>(rows, 16)};
    dim3 grid{cdiv(colsDst, block.x), cdiv(rows, block.y)};
    sym_quantize_f16_i4_kernel<<<grid, block>>>(x, scale, rows, colsSrc, colsDst, q);
}


__global__ void sym_dequantize_i32_f16_kernel(
        const int32_t *__restrict__ q,
        const half *__restrict__ scale_row,
        const half *__restrict__ scale_col,
        uint32_t rows, uint32_t cols,
        half *__restrict__ x)
{
    uint32_t row = threadIdx.y + blockIdx.y * blockDim.y;
    uint32_t col = threadIdx.x + blockIdx.x * blockDim.x;

    if (col >= cols || row >= rows)
    {
        return;
    }

    half xElement = int_to_half(q[col + row * cols]);
    x[col + row * cols] = scale_row[row] * scale_col[col] * xElement;
}

void sym_dequant_host(const int32_t *q,
                                 const half *scale_row,
                                 const half *scale_col,
                                 uint32_t rows,
                                 uint32_t cols,
                                 half *x
)
{
    dim3 block{std::min<uint32_t>(cols, 16), std::min<uint32_t>(rows, 16)};
    dim3 grid{cdiv(cols, block.x), cdiv(rows, block.y)};
    sym_dequantize_i32_f16_kernel<<<grid, block>>>(
            q,
            scale_row, scale_col,
            rows, cols, x);
}
