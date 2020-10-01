#include <stdio.h>

#include "matrix.h"

// Take a volume and return a flattened index for a given slice, row and column
DDIM tp_indices(Matrix3D *vol, DIM islice, DIM irow, DIM icolumn) {
  return islice * vol->nrows * vol->ncolumns + irow * vol->ncolumns + icolumn;
}


// Take a slice and return a flattened index for a given row and column
__host__ __device__ DDIM tp_indices2d(Matrix2D *vol, DIM irow, DIM icolumn) {
  return irow * vol->ncolumns + icolumn;
}

Matrix3D *tp_matrixmalloc(DIM n_slices, DIM n_rows, DIM n_columns) {
  // Allocated memory for the 3D matrix
  // Matrix3D *new_matrix = (Matrix3D *) cudaMallocManaged(sizeof(Matrix3D) + n_slices * n_rows * n_columns * sizeof(DTYPE));
  Matrix3D *new_matrix;
  cudaMallocManaged(&new_matrix, sizeof(Matrix3D) + n_slices * n_rows * n_columns * sizeof(DTYPE));
  if (new_matrix == NULL) {
    fprintf(stderr, "Unable to allocate memory for (%lu, %lu, %lu) array.", n_slices, n_rows, n_columns);
  } 
  // Store the size of the array
  new_matrix->nslices = n_slices;
  new_matrix->nrows = n_rows;
  new_matrix->ncolumns = n_columns;
  return new_matrix;
}


Matrix2D *tp_matrixmalloc2d(DIM n_rows, DIM n_columns) {
  // Allocated memory for the 2D matrix
  Matrix2D *new_matrix;
  cudaMallocManaged(&new_matrix, sizeof(Matrix2D) + n_rows * n_columns * sizeof(DTYPE));
  if (new_matrix == NULL) {
    fprintf(stderr, "Unable to allocate memory for (%lu, %lu) array.", n_rows, n_columns);
  } 
  // Store the size of the array
  new_matrix->nrows = n_rows;
  new_matrix->ncolumns = n_columns;
  return new_matrix;
}
