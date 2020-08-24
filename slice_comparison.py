import numpy as np
import h5py
import matplotlib.pyplot as plt

inpt = np.loadtxt('slice32_input_test.txt')
outpt = np.loadtxt('slice32_output_test.txt')
prewrite = np.loadtxt('slice32_prewrite_test.txt')

with h5py.File('data/phantom3d.h5', mode='r') as h5fp:
    initial = h5fp['volume'][32]
    final = h5fp['_tomopore_temp'][32]

print("Initial: ", np.array_equal(initial, initial))
print("Input: ", np.array_equal(initial, inpt))
print("Output: ", np.array_equal(initial, outpt))
print("Final: ", np.array_equal(initial, final))

plt.imshow(initial)
plt.show()

plt.imshow(inpt)
plt.show()

with h5py.File('data/phantom3d.h5', mode='r+') as h5fp:
    del h5fp['slice_comparison']
    h5fp.create_dataset('slice_comparison', data=[initial, inpt, outpt, prewrite, final])
