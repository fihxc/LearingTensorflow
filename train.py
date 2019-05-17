import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

torch.manual_seed(1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dic = {}
file_mark = open("dic.txt",encoding = 'GB2312',mode ='r')

for line in file_mark:
    word_id = line.split()
    dic[word_id[0]] = word_id[1]

file_mark.close()


train_data = []
test_data = []

#load data

train_data_num = 0
test_data_num = 0

file_train = open("train_data.txt",encoding = 'GB2312',mode ='r')
file_test = open("test_data.txt",encoding = 'GB2312',mode = 'r')

for line in file_train:
    tag_list = []
    word_list = line.split()
    train_data_num += len(word_list)
    for i in range(len(word_list)):
        tag_list.append(dic[word_list[i]])
    tmp = (word_list,tag_list)
    train_data.append(tmp)

file_train.close()

for line in file_test:
    tag_list = []
    word_list = line.split()
    test_data_num += len(word_list)
    for i in range(len(word_list)):
        tag_list.append(dic[word_list[i]])
    tmp = (word_list,tag_list)
    test_data.append(tmp)


word_to_id = {}

for sent,tag in train_data + test_data:
    for word in sent:
        if word not in word_to_id:
            word_to_id[word]=len(word_to_id)

tag_to_id = {"1":0, "2":1, "3":2, "4":3, "?":4}

EMBEDDING_DIM = 512
HIDDEN_DIM = 512

#define...
def prepare_sequence(seq,to_id):
    idxs=[to_id[w] for w in seq]
    return torch.tensor(idxs,dtype=torch.long)

class LSTMTagger(nn.Module):
    def __init__(self,embedding_dim,hidden_dim,vocab_size,tagset_size):
        super(LSTMTagger,self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size,embedding_dim).to(device)
        self.lstm = nn.LSTM(embedding_dim,hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim,tagset_size).to(device)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.zeros(1,1,self.hidden_dim).to(device),
                (torch.zeros(1,1,self.hidden_dim).to(device)))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence).to(device)
        lstm_out,self.hidden = self.lstm(embeds.view(len(sentence),1,-1),self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence),-1)).to(device)
        tag_scores = F.log_softmax(tag_space,dim=1).to(device)
        return tag_scores

model = LSTMTagger(EMBEDDING_DIM,HIDDEN_DIM,len(word_to_id),len(tag_to_id)).to(device)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(),lr = 0.01)

best_count = 0
best_per = 0

#train...
for epoch in range(100):
    print("\n")
    print("round :" + str(epoch))
    for word,tags in train_data:
        model.zero_grad()
        model.hidden = model.init_hidden()
        word_in = prepare_sequence(word,word_to_id).to(device)
        targets = prepare_sequence(tags,tag_to_id).to(device)
        tag_scores = model(word_in)
        loss = loss_function(tag_scores,targets).to(device)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        print("Waiting Test!")
        right_count = 0
        count_1 = 0
        count_2 = 0
        count_3 = 0
        count_4 = 0
        count_what = 0
        for i in range(len(test_data)):#sentence
            inputs = prepare_sequence(test_data[i][0],word_to_id).to(device)
            tag_scores = model(inputs)
            for j in range(len(tag_scores)):
                max_place = 0
                if tag_scores[j][1] > tag_scores[j][max_place]:
                    max_place = 1
                elif tag_scores[j][2] > tag_scores[j][max_place]:
                    max_place = 2
                elif tag_scores[j][3] > tag_scores[j][max_place]:
                    max_place = 3
                elif tag_scores[j][4] > tag_scores[j][max_place]:
                    max_place = 4

                if max_place == 0:
                    count_1 += 1
                elif max_place == 1:
                    count_2 += 1
                elif max_place == 2:
                    count_3 += 1
                elif max_place == 3:
                    count_4 += 1  
                else:
                    count_what += 1
                
                if max_place == 0 and test_data[i][1][j] == '1':
                    right_count += 1
                elif max_place == 1 and test_data[i][1][j] == '2':
                    right_count += 1
                elif max_place == 2 and test_data[i][1][j] == '3':
                    right_count += 1
                elif max_place == 3 and test_data[i][1][j] == '4':
                    right_count += 1
                elif max_place == 4 and test_data[i][1][j] == '?':
                    right_count += 1
        if right_count > best_count:
            best_count = right_count
            best_per = right_count/test_data_num
        print("now: ",right_count)
        print("now: ",right_count/test_data_num)
        print("1: ",count_1)
        print("2: ",count_2)
        print("3: ",count_3)
        print("4: ",count_4)
        print("?: ",count_what)
        print("best: ",best_count)
        print("best: ",best_per)

torch.save(model.state_dict(), "model_all.pth")