from keras.models import load_model
from S2S_3 import Word_Attention
from S2S_3 import Attention
import numpy as np
import os
from nltk import tokenize
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


SEED = 9  # 随机数种子

MAX_NB_WORDS = 20000  # 词的最大整数
MAX_SEQUENCE_LENGTH = 50  # 词向量长度
#MAX_SEQUENCE_LENGTH = 256  # 词向量长度
EMBEDDING_DIM = 400  # Embedding层
MAX_SENTS = 15  # 句子中词的个数

# 编码函数
def to_code(X):
    os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'

    reviews = []
    tokenized_text = []

    X = [str(x) for x in X]

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

# 结果可视化函数
def Draw_bar(features, l1, l2, l3):
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

# 导入训练好的模型
cu_ob = {'Word_Attention': Word_Attention,'Attention': Attention}
new_model = load_model('E:/专业实习/saves/S2SAN_1_1658473613.711423.h5', custom_objects=cu_ob)

# 导入数据
cl_result = pd.read_csv('./submission.csv')
X = cl_result['discourse_text']
X = [str(x) for x in X]
Y = np.asarray(cl_result['class'])
#Y = [0 if star == 'Ineffective' else 1 if star == 'Effective' else 2 for star in list(Y)]
X = to_code(X)

# 预测
ypre = new_model.predict(X)  # 输出属于各个类的概率
ypred = np.argmax(ypre, axis=1)  # 输出最大概率的类

ineffe = ypre[:,0]
Effec = ypre[:,1]
Adeq = ypre[:,2]
Draw_bar(Y, ineffe, Effec, Adeq)