
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from bert import BERT
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Create class for data preparation
class SimpleDataset:
    def __init__(self, tokenized_texts, tokenized_class):
        self.tokenized_texts = tokenized_texts
        self.tokenized_class = tokenized_class
    def __len__(self):
        return len(self.tokenized_texts["input_ids"])

    def __getitem__(self, idx):
        return {
            k: v[idx]for k, v in self.tokenized_texts.items()} ,  self.tokenized_class[idx]

# Create class for data preparation
class SimpleDataset_only:
    def __init__(self, tokenized_texts):
        self.tokenized_texts = tokenized_texts
    def __len__(self):
        return len(self.tokenized_texts["input_ids"])

    def __getitem__(self, idx):
        return {
            k: v[idx]for k, v in self.tokenized_texts.items()}

class CNN_BiLSTM(nn.Module):

    def __init__(self):
        super(CNN_BiLSTM, self).__init__()
        self.hidden_dim = 512
        self.num_layers = 1
        V = 30000
        D = 512
        C = 7
        self.C = C
        Ci = 1
        Co = 512
        Ks = [3,4,5]
        self.embed = nn.Embedding(V, D)
        # pretrained  embedding

        # CNN
        self.convs1 = [nn.Conv2d(Ci, Co, (K, D), padding=(K//2, 0), stride=1) for K in Ks]
        # print(self.convs1)
        # for cnn cuda

        for conv in self.convs1:
            conv = conv.cuda()

        # BiLSTM
        self.bilstm = nn.LSTM(D, self.hidden_dim, num_layers=self.num_layers, dropout= 0, bidirectional=True, bias=True)

        # linear
        L = len(Ks) * Co + self.hidden_dim * 2
        self.hidden2label1 = nn.Linear(2048, 512).cuda()
        self.hidden2label2 = nn.Linear(512, 2048).cuda()

        # dropout
        self.dropout = nn.Dropout(0)

        self.Linear1 = nn.Linear(2048,1).cuda()
        self.Linear2 = nn.Linear(768, 1).cuda()
        self.Linear3 = nn.Linear(7, 1).cuda()
        self.Linear4 = nn.Linear(1536, 2).cuda()

    def forward(self, x , result_b , result_rob):
        embed = self.embed(x)

        # CNN
        cnn_x = embed
        cnn_x = torch.transpose(cnn_x, 0, 1)
        cnn_x = cnn_x.unsqueeze(1).unsqueeze(0).permute(1,0,2,3).cuda()
        i = 0
        for conv in self.convs1 :
            cnn_x = conv(cnn_x)# [(N,Co,W), ...]*len(Ks)
            i = i +1
            if i ==  1 :
                cnn_x = cnn_x.permute(0,3,2,1)
            if i == 2 :
                cnn_x = cnn_x.permute(0,3,2,1)
        cnn_x = [F.tanh(F.max_pool1d(i, i.size(2)).squeeze(2)) for i in cnn_x]  # [(N,Co), ...]*len(Ks)
        cnn_x = torch.cat(cnn_x, 1)
        cnn_x = self.dropout(cnn_x)

        # BiLSTM
        bilstm_x = embed.view(-1,len(x), embed.size(1))
        bilstm_out, _ = self.bilstm(bilstm_x)
        bilstm_out = torch.transpose(bilstm_out, 0, 1)
        bilstm_out = torch.transpose(bilstm_out, 1, 2)
        bilstm_out = F.max_pool1d(bilstm_out, bilstm_out.size(2)).squeeze(2)
        bilstm_out = F.tanh(bilstm_out)

        # CNN and BiLSTM CAT
        cnn_x = torch.transpose(cnn_x, 0, 1).cuda()
        bilstm_out = torch.transpose(bilstm_out, 0, 1).cuda()
        cnn_bilstm_out = torch.cat((cnn_x, bilstm_out), 0).cuda()
        cnn_bilstm_out = torch.transpose(cnn_bilstm_out, 0, 1).cuda()

        # linear
        cnn_bilstm_out = self.hidden2label1(F.tanh(cnn_bilstm_out))
        cnn_bilstm_out = self.hidden2label2(F.tanh(cnn_bilstm_out))
        result_b= result_b.cuda()
        result_rob = result_rob.cuda()
        # output
        result_c = self.Linear1(cnn_bilstm_out).squeeze(-1).unsqueeze(0)
        result_b = self.Linear2(result_b).squeeze(-1)
        result_rob = self.Linear3(result_rob.expand(512,7)).squeeze(-1).unsqueeze(0)
        result = torch.cat((result_rob,result_b,result_c),dim=-1)
        result = self.Linear4(F.softmax(result))

        return result

if __name__=='__main__':
    # load tokenizer and model, create trainer
    model_name = "emotion-english-distilroberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    trainer = Trainer(model=model)

    # specify your filename
    file_name = "GEN-sarc-notsarc1.csv"  # note: you can right-click on your file and copy-paste the path to it here
    text_column = "text"  # select the column in your csv that contains the text to be classified

    # read in csv
    df_pred = pd.read_csv(file_name)
    dataset = []
    pred_texts = df_pred[text_column].dropna().astype('str').tolist()
    pred_class = df_pred["class"].dropna().astype('int').tolist()


    # Tokenize texts and create prediction data set
    tokenized_texts = tokenizer( pred_texts , truncation = True,padding=True)
    pred_dataset = SimpleDataset(tokenized_texts , pred_class)
    pred_dataset_only = SimpleDataset_only(tokenized_texts)


    # Transform predictions to labels
    predictions = trainer.predict(pred_dataset_only)
    preds = predictions.predictions.argmax(-1)
    labels = torch.from_numpy(predictions.predictions)  # 6507 * 7
    cnn = CNN_BiLSTM()
    # 加载预训练model
    # model_dict = cnn.state_dict()
    # pretrain = torch.load('result.pth')  # ,map_location ='cpu'
    # cnn.load_state_dict(pretrain)

    model_B = BERT('bert-base-uncased', 'bert/bert-base-uncased.tar.gz', False, 256, 512, 12).cuda()
    optimizer = torch.optim.AdamW(cnn.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()
    cnn.train()
    for epoch in range(10):
        cnt_loss = []
        for index, data in enumerate(pred_dataset):
            optimizer.zero_grad()
            datas = data[0]
            datas = datas['input_ids']
            if len(datas) >=  512 :
                datas = datas[0:512]
            else:
                datas.extend([0 for i in range(512 - len(datas))])
            datas = torch.tensor(datas)
            datas = torch.clamp(datas,0,29999)
            result_rob = labels[index]
            result_b = model_B(datas.unsqueeze(0))
            # print(result_b)
            result = cnn(datas , result_b , result_rob)
            # print(result)
            if data[1] == 0:
                rs = torch.zeros(1)
            else:
                rs = torch.ones(1)
            loss = criterion(result, rs.long().cuda())
            loss.backward()
            optimizer.step()
            cnt_loss.append(loss.cpu().detach().numpy())
        print(np.mean(cnt_loss))
    torch.save(cnn.state_dict(), 'result.pth')




    # labels = pd.Series(preds).map(model.config.id2label)
    # scores = (np.exp(predictions[0])/np.exp(predictions[0]).sum(-1,keepdims=True)).max(1)
    #
    # # scores raw
    # temp = (np.exp(predictions[0])/np.exp(predictions[0]).sum(-1,keepdims=True))
    #
    # # work in progress
    # # container
    # anger = []
    # disgust = []
    # fear = []
    # joy = []
    # neutral = []
    # sadness = []
    # surprise = []
    #
    # # extract scores (as many entries as exist in pred_texts)
    # for i in range(len(pred_texts)):
    #
    #   anger.append(temp[i][0])
    #   disgust.append(temp[i][1])
    #   fear.append(temp[i][2])
    #   joy.append(temp[i][3])
    #   neutral.append(temp[i][4])
    #   sadness.append(temp[i][5])
    #   surprise.append(temp[i][6])
    #
    #   # Create DataFrame with texts, predictions, labels, and scores
    # df = pd.DataFrame(list(zip(pred_texts, preds, labels, scores, anger, disgust, fear, joy, neutral, sadness, surprise)),
    #                     columns=['text', 'pred', 'label', 'score', 'anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness',
    #                              'surprise'])
    # df.head()
    #
    # # save results to csv
    # YOUR_FILENAME = "baozhiqi.csv"  # name your output file
    # df.to_csv(YOUR_FILENAME)
