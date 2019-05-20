import pickle as pic

data = {
    'position': 'N2 E3',
    'pocket': ['key', 'knife'],
    'money': 160
}
print(data)
file_a = open('01.pickle_save_learning.pickle', 'wb')
pic.dump(data, file_a)
file_a.close()

file_b = open('01.pickle_save_learning.pickle', 'rb')
data1 = pic.load(file_b)
file_b.close()
