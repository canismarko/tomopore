#include <CUnit/CUnit.h>
#include <stdio.h>
#include "../src/tomopore.h"


void test_tp_apply_kernel()
{
  // Prepare a test volume with five slices, and the middle one is ones
  Matrix3D *test_volume = tp_matrixmalloc(5, 5, 5);
  for (int i=0; i<(5*5*5); i++) {
    if ((i >= tp_indices(test_volume, 3, 0, 0)) && (i <= tp_indices(test_volume, 3, 5, 5))) {
      test_volume->arr[i] = 1;
    } else {
      test_volume->arr[i] = 0;
    }
  }
  
  // Create a cubic kernel for testing
  Matrix3D *box_kernel = tp_matrixmalloc(3, 3, 3);
  tp_box(box_kernel);
  
  printf("%f\n", test_volume->arr[tp_indices(test_volume, 0, 0, 0)]);
  // Check what happens at the boundaries
  /* float result = tp_apply_kernel(test_volume, box_kernel, 0, 0, 0); */
  /* if (result != 1) { */
  /*   fprintf(stderr, "tp_apply_kernel: boundary condition failed\n"); */
  /* } */
}


int main()
{
  // Run the tests
  test_tp_apply_kernel();
  
  return 0;
}
