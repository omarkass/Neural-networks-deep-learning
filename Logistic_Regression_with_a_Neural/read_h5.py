import h5py
import numpy as np
filename = "train_catvnoncat.h5"
f = h5py.File(filename, 'r')
print("Keys: %s" % f.keys())
a_group_key = list(f.keys())[0]
#data = list(f['train_set_y'])
filename2 ="test_catvnoncat.h5"
f2 = h5py.File(filename2 , 'r')
print (f2.keys())
print (  np.array(list(f2['test_set_y'])).shape)
#print (data)
