import os
import numpy as np
import matplotlib.pyplot as plt

##
result_dir = './results/numpy'

lst_data = os.listdir(result_dir)

lst_label = [f for f in lst_data if f.startswith('label')]
lst_input = [f for f in lst_data if f.startswith('input')]
lst_output = [f for f in lst_data if f.startswith('output')]

lst_label.sort()
lst_input.sort()
lst_output.sort()

##
id = 0

# 데이터 로드
label = np.load(os.path.join(result_dir, lst_label[id]))
input = np.load(os.path.join(result_dir, lst_input[id]))
output = np.load(os.path.join(result_dir, lst_output[id]))

##
plt.subplot(131)
plt.imshow(label, cmap = 'gray')
plt.title('label')

plt.subplot(132)
plt.imshow(input, cmap = 'gray')
plt.title('input')

plt.subplot(133)
plt.imshow(output, cmap = 'gray')
plt.title('output')

