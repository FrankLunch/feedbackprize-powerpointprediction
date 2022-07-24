#! /usr/bin/env python
from __future__ import print_function
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras import models
from keras import layers

from nltk import tokenize
import pickle
import time
import pandas as pd
from keras import initializers
import os
from keras.engine.topology import Layer

from keras import backend as K
from sklearn.model_selection import StratifiedKFold, KFold
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns


class Word_Attention(Layer):

    
    def __init__(self, attention_dim: int = 100, **kwargs):
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(Word_Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        #print(len(input_shape))
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
        self.b = K.variable(self.init((self.attention_dim, )))
        self.u = K.variable(self.init((self.attention_dim, 1)))
        self.trainable_weights = [self.W, self.b, self.u]
        super(Word_Attention, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return None

    def call(self, x, mask=None):
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)

        ait = K.exp(ait)

        if mask is not None:
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        config = {'attention_dim': self.attention_dim}
        base_config = super(Word_Attention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        #return {'attention_dim': self.attention_dim}



class Attention(Layer):
    def __init__(self, nb_head: int = 1, size_per_head: int = 32, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head*size_per_head
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.WQ = self.add_weight(name='WQ',
                                  shape=(input_shape[0][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK',
                                  shape=(input_shape[1][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV',
                                  shape=(input_shape[2][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(Attention, self).build(input_shape)

    def Mask(self, inputs, seq_len, mode='mul'):
        if seq_len == None:
            return inputs
        else:
            mask = K.one_hot(seq_len[:,0], K.shape(inputs)[1])
            mask = 1 - K.cumsum(mask, 1)
            for _ in range(len(inputs.shape)-2):
                mask = K.expand_dims(mask, 2)
            if mode == 'mul':
                return inputs * mask
            if mode == 'add':
                return inputs - (1 - mask) * 1e12

    def call(self, x):
        if len(x) == 3:
            Q_seq,K_seq,V_seq = x
            Q_len,V_len = None,None
        elif len(x) == 5:
            Q_seq,K_seq,V_seq,Q_len,V_len = x

        Q_seq = K.dot(Q_seq, self.WQ)
        Q_seq = K.reshape(Q_seq, (-1, K.shape(Q_seq)[1], self.nb_head, self.size_per_head))
        Q_seq = K.permute_dimensions(Q_seq, (0,2,1,3))
        K_seq = K.dot(K_seq, self.WK)
        K_seq = K.reshape(K_seq, (-1, K.shape(K_seq)[1], self.nb_head, self.size_per_head))
        K_seq = K.permute_dimensions(K_seq, (0,2,1,3))
        V_seq = K.dot(V_seq, self.WV)
        V_seq = K.reshape(V_seq, (-1, K.shape(V_seq)[1], self.nb_head, self.size_per_head))
        V_seq = K.permute_dimensions(V_seq, (0,2,1,3))
        #计算内积，然后mask，然后softmax
        A = K.batch_dot(Q_seq, K_seq, axes=[3,3]) / self.size_per_head**0.5
        A = K.permute_dimensions(A, (0,3,2,1))
        A = self.Mask(A, V_len, 'add')
        A = K.permute_dimensions(A, (0,3,2,1))
        A = K.softmax(A)
        #输出并mask
        O_seq = K.batch_dot(A, V_seq, axes=[3,2])
        O_seq = K.permute_dimensions(O_seq, (0,2,1,3))
        O_seq = K.reshape(O_seq, (-1, K.shape(O_seq)[1], self.output_dim))
        O_seq = self.Mask(O_seq, Q_len, 'mul')
        return O_seq

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)

    def get_config(self):
        config = {'nb_head': self.nb_head, 'size_per_head': self.size_per_head}
        base_config = super(Attention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

"""
SEED = 9  # 随机数种子

MAX_NB_WORDS = 20000  # 词的最大整数
MAX_SEQUENCE_LENGTH = 50  # 词向量长度
#MAX_SEQUENCE_LENGTH = 256  # 词向量长度
EMBEDDING_DIM = 400  # Embedding层
MAX_SENTS = 15  # 句子中词的个数

def build_hfan(maxlen=MAX_SEQUENCE_LENGTH, max_sent_len=MAX_SENTS, max_words=MAX_NB_WORDS, 
               embedding_dim=EMBEDDING_DIM, classification_type=3):
    S_inputs = layers.Input(shape=(maxlen,), dtype='int32')
    # 将向量映射为一个指定维度的向量
    O_seq = layers.Embedding(max_words, embedding_dim, input_length=maxlen)(S_inputs)
    O_seq = layers.Bidirectional(layers.GRU(100, return_sequences=True))(O_seq)
    O_seq = Word_Attention(100)(O_seq)  # 改 100
    sentences_model = models.Model(inputs=S_inputs, outputs=O_seq)
    sentences_model.summary()

    review_input = layers.Input(shape=(max_sent_len, maxlen), dtype='int32')
    review_encoder = layers.TimeDistributed(sentences_model)(review_input)
    l_lstm_sent = Attention(1, 32)([review_encoder, review_encoder, review_encoder])
    l_att_sent = layers.GlobalAveragePooling1D()(l_lstm_sent)
    preds = layers.Dense(classification_type, activation='softmax')(l_att_sent)
    model = models.Model(review_input, preds)
    return model


os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'

reviews = []
tokenized_text = []

df = pd.read_csv('E:/专业实习/数据集feedback-prize-effectiveness/train.csv')
X = df['discourse_text']  # 改
X = [str(x) for x in X]
Y = np.asarray(df['discourse_effectiveness'])  # 改
Y = [0 if star == 'Ineffective' else 1 if star == 'Effective' else 2 for star in list(Y)]


# 按句子分割
for txt in list(X):
    sentences = tokenize.sent_tokenize(txt)  
    reviews.append(sentences)

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(list(X))
#print("reviews:",np.shape(reviews))  # (36765,)

# 将文本转换为序列
for sentences in reviews:
    tokenized_sentences = tokenizer.texts_to_sequences(sentences)  
    tokenized_sentences = tokenized_sentences[:MAX_SENTS]
    tokenized_text.append(tokenized_sentences)
#print("tokenized_text:",np.shape(tokenized_text))  # (36765,)

X = np.zeros((len(tokenized_text), MAX_SENTS, MAX_SEQUENCE_LENGTH), dtype='int32')
for i in range(len(tokenized_text)):
    sentences = tokenized_text[i]
    # maxlen设置最大的序列长度，长于该长度的序列将会截短，短于该长度的序列将会填充
    seq_sequences = pad_sequences(sentences, maxlen=MAX_SEQUENCE_LENGTH)  
    for j in range(len(seq_sequences)):
        X[i, j] = seq_sequences[j]
        
x_train = X[:1500,:,:]
x_val = X[1500:1700,:,:]
x_test = X[1700:1900,:,:]
y_train = Y[:1500]
y_val = Y[1500:1700]
y_test = Y[1700:1900]
y_train = to_categorical(np.asarray(y_train))
y_val = to_categorical(np.asarray(y_val))

model = build_hfan()
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop', metrics=['acc'])
model.summary()
history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                    epochs=5, batch_size=50)  # 改



# 结果可视化
def result_view(y, ypre):
    matrix = confusion_matrix(y, ypre)
    sns.heatmap(matrix.T, square=True, annot=True, fmt='d', cbar=False)
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    plt.show()

ypre = model.predict(x_test)
ypred = np.argmax(ypre, axis=1)

# precisoin:即准确率，也称查准率。
# recall:召回率 ，也称查全率，
# f1-score:简称F1
# support:个数
# accruracy :整体的准确率 即正确预测样本量与总样本量的比值。（不是针对某个标签的预测的正确率）
# macro avg :即宏均值，可理解为普通的平均值。
# weighted avg :宏查准率，上面类别各分数的加权（权值为support）平均

print(classification_report(y_test, ypred))
result_view(y_test, ypred)



sub = pd.DataFrame()
#sub.head()
sub['Ineffective'] = ypre[:,0]
sub['Effective'] = ypre[:,1]
sub['Adequate'] = ypre[:,2]
sub['ypre'] = ypred
sub['ytrue'] = y_test
sub.to_csv("submission.csv", index=False)


def  Draw_bar(features, l1, l2, l3):
    features = features
    ine = l1
    effec = l2
    adaq = l3
    bar_width = 0.2
    index_ine = np.arange(len(features))
    index_effec = index_ine + bar_width
    index_adaq = index_effec + bar_width
    plt.figure(figsize=(15,8))
    plt.bar(index_ine, height = ine, width = bar_width, 
            color = 'salmon', label = 'Ineffective')
    plt.bar(index_effec, height = effec, width = bar_width, 
            color = 'yellow', label = 'Effective')
    plt.bar(index_adaq, height = adaq, width = bar_width, 
            color = 'lightgreen', label = 'Adequate')
    
    plt.legend()
    plt.xticks(index_ine + bar_width, features)
    plt.xlabel('discourse_type')
    plt.ylabel('discourse_effectiveness')
    plt.show()

xlabel = ['Lead', 'Position', 'Claim', 'Evidence', 
        'Counterclaim', 'Rebuttal', 'Concluding Statement']
ineffe = ypre[0:7,0]
Effec = ypre[0:7,1]
Adeq = ypre[0:7,2]
Draw_bar(xlabel, ineffe, Effec, Adeq)

"""