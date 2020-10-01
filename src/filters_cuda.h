#include <cuda_runtime.h>

__device__ float tp_apply_kernel
(Matrix3D *subvolume, Matrix3D *kernel, DIM islc, DIM irow, DIM icol, enum operation op);

char tp_apply_kernel_cuda
(Matrix3D *working_buffer, Matrix3D *kernel, Matrix2D *pore_slice_buffer, enum operation op, DIM islc);
