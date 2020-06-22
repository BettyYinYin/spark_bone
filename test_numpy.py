import numpy as np


arr = []
arr_np = np.zeros([2, 7])

# print(arr_np)
arr_np[0][0] = 1
print(arr_np)

arr.append(arr_np)

# print(arr)
print(np.array(arr).shape)
print(np.array(arr))


import numpy as np

W_train = [[[1,2,3], [1, 2]], [1, 2]]
R_train = [[4,5,6], [3, 4]]
Y_train = [[7,8,9], [7, 7]]
L_train = [[10,11,12], [1, 1]]

data_train = [(w, r, y, l) for w, r, y, l in zip(W_train, R_train, Y_train, L_train)]

np.random.shuffle(data_train)
print(data_train )
W_train = [w for w, r, y, l in data_train]
R_train = [r for w, r, y, l in data_train]
Y_train = [y for w, r, y, l in data_train]
L_train = [l for w, r, y, l in data_train]
print(W_train, R_train, Y_train, L_train)

