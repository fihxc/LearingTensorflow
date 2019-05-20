import pickle as pic
import numpy as np

num=[]
for i in range(73):
    a = input().split()
    for j in range(len(a)):
        a[j] = int(a[j])
    num.append(a)
num = np.array(num)
file_a = open('y.pickle', 'wb')
pic.dump(num, file_a)
file_a.close()

file_b = open('y.pickle', 'rb')
data1 = pic.load(file_b)
file_b.close()
print(data1)