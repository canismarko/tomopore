#include <cstdint>

// Type definitions
typedef uint64_t DIM;
typedef uint64_t DDIM;
typedef double VEC;
typedef float DTYPE;
typedef struct {
  DIM z;
  DIM y;
  DIM x;
} Point;
typedef struct {
  VEC z;
  VEC y;
  VEC x;
} Vector;


class Matrix2D {
 protected:
  DIM _nrows;
  DIM _ncolumns;
  DDIM _nelements;
 public:
  DTYPE *arr;
  Matrix2D(DIM row, DIM columns);
  ~Matrix2D();
  DIM getNRows() {return _nrows;}
  DIM getNColumns() {return _ncolumns;}
  DTYPE getElement(DIM row, DIM column);
  void setElement(DIM row, DIM column, DTYPE value);
  DDIM flatIndex(DIM row, DIM column);
};


class Matrix3D : public Matrix2D {
 protected:
  DIM _nslices;
 public:
  Matrix3D(DIM slices, DIM row, DIM columns);
  DIM getNSlices() {return _nslices;}
  DTYPE getElement(DIM slice, DIM row, DIM column);
  void setElement(DIM slice, DIM row, DIM column, DTYPE value);
  DDIM flatIndex(DIM slice, DIM row, DIM column);
};
