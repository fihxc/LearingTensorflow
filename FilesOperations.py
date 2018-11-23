import pickle as pic

data = {
    'position': 'N2 E3',
    'pocket': ['key', 'knife'],
    'money': 160
}
print(data)
file_a = open('pickle_save_learning.dat', 'wb')
pic.dump(data, file_a)
file_a.close()

file_b = open('pickle_save_learning.dat', 'rb')
data1 = pic.load(file_b)
file_b.close()
print(data1)