#include <cstdlib>
#include <cstdio>
#include <iostream>

#include "matrix.h"


Matrix2D::Matrix2D(DIM rows, DIM columns)
  : _nrows{rows}, _ncolumns{columns}
{
  // Allocate memory for the actual array
  _nelements = _nrows * _ncolumns;
  arr = new DTYPE[_nelements];
}

Matrix2D::~Matrix2D(void)
{
  // Free up the memory previously allocated for holding the array
  delete[] arr;
}

DTYPE Matrix2D::getElement(DIM row, DIM column)
// Retrieve a data element based on structured indices
{
  return arr[flatIndex(row, column)];
}

void Matrix2D::setElement(DIM row, DIM column, DTYPE value)
// Set a data element to a given value based on structured indices
{
  arr[flatIndex(row, column)] = value;
}

DDIM Matrix2D::flatIndex(DIM row, DIM column)
{
  return row * _ncolumns + column;
}



Matrix3D::Matrix3D(DIM slices, DIM rows, DIM columns)
  : _nslices{slices}, Matrix2D(rows, columns)
{
  // Allocate memory for the actual array
  _nelements = _nslices * _nrows * _ncolumns;
  arr = new DTYPE[_nelements];
}

DTYPE Matrix3D::getElement(DIM slice, DIM row, DIM column)
// Retrieve a data element based on structured indices
{
  return arr[flatIndex(slice, row, column)];
}

void Matrix3D::setElement(DIM slice, DIM row, DIM column, DTYPE value)
// Set a data element to a given value based on structured indices
{
  arr[flatIndex(slice, row, column)] = value;
}

DDIM Matrix3D::flatIndex(DIM slice, DIM row, DIM column)
{
  return slice * _nrows * _ncolumns + row * _ncolumns + column;
}

