import tensorflow as tf
import pandas as pd
import numpy as np
import os
import random
import collections
import time
import pickle
import re


# In[2]:

hw_dir = '/home/yao/gm/mlds/final'
# hw_dir = '/Users/Apple/Desktop/MLDS/final'
# hw_dir = '/Users/yao/Desktop/trueman/big4/MLDS/final'
test_max_slen = 20
train_softmax_slen = 50
train_max_slen = 70

max_slen = 70
min_slen = 5

batch_size = 1280


# In[3]:

with open('%s/all_pickles'%hw_dir,'rb') as f1:
    [train_data,test_data,word2ix,ix2word,vocab_size] = pickle.load(f1)


# In[4]:

blank_ix = word2ix[' ']
new_data = []
for data in train_data:
    temp = []
    sc = 0
    if(int(data[0]) < 3):
        sc = 0
    elif(int(data[0]) > 3):
        sc = 1
    else:
        continue
    temp.append(sc)
    temp.append([int(x) for x in data[1].split(" ")])
    new_data.append(temp)
train_data = new_data
train_data = [i for i in train_data if len(i[1])<=70 and len(i[1])>=5]

blank_ix = word2ix[' ']
new_data = []
for data in test_data:
    temp = []
    sc = 0
    if(int(data[0]) < 3):
        sc = 0
    elif(int(data[0]) > 3):
        sc = 1
    else:
        continue
    temp.append(sc)
    temp.append([int(x) for x in data[1].split(" ")])
    new_data.append(temp)
test_data = new_data

len(train_data),len(test_data)


# In[5]:

train_data_dict = dict()
for i in train_data:
    len_cmmt = len(i[1])
    if len_cmmt in train_data_dict:
        train_data_dict[len_cmmt].append(i)
    else:
        train_data_dict[len_cmmt] = [i]

del train_data

# In[6]:

len_pos    = max_slen  # 現在在長度多少的file，從長的開始
batch_pos  = 0
def get_batch(batch_size):
    global train_data_dict, len_pos,batch_pos
    batch_size = (batch_size//len_pos)+1 # 越長的句子一個batch就讀越少句，+1是避免0的狀況
    # ====================回傳用的參數=============================
    is_eoi = False  # end of iteration
    is_eof = False
    slen  = len_pos
    # ===========================================================
    # ====================從檔案中讀batch_size句===================
    now_datas = train_data_dict[len_pos][batch_pos:batch_pos+batch_size]
    batch_pos+=batch_size
    if len(now_datas)<batch_size:
        batch_pos=0
        is_eof = True
    # ===========================================================
    batch_size = len(now_datas) # 因為有可能在檔案的尾巴 讀不滿完整的batch_size 故在此更新batch_size
    # ==================若現在的檔案讀完============================
    if is_eof == True:
        #random.shuffle(train_data_dict[len_pos])
        if len_pos == min_slen: # 一個iteration跑完 重新再跑
            len_pos = max_slen
            is_eoi = True
        else:                      # 現在這長度的file完了 開始讀len_pos-1的檔案
            len_pos -= 1
        if batch_size == 0: # 剛好讀到尾巴了
            return is_eoi,is_eof,None,None,slen,batch_size
    # ============================================================
        
    score = np.array([i[0] for i in now_datas])
    cmmt = np.array([i[1] for i in now_datas])
    return is_eoi,is_eof,score,cmmt,slen,batch_size

# In[9]:

lr = 0.0001                   
training_iters = 100      

n_vocab_size = vocab_size
n_filter = 64
n_filter_size = 4
n_classes = 2
n_max_pool = 2
n_hidden_units = 512
n_embed_size = 300
n_batch_size = 5000
rnn_layer_num = 4
filter_sizes = np.array([3,4,5])
n_filter_kind = 3

# In[10]:

tf.reset_default_graph()


# In[11]:

word_embed = tf.Variable(tf.random_uniform([vocab_size,n_embed_size]))
y_embed = tf.constant(np.eye(n_classes))


# # CNN

# In[12]:

# n_sentence_length = tf.placeholder(tf.int32,[1])
x = tf.placeholder(tf.int32, [None, None])
# [2,4,3]
y = tf.placeholder(tf.int32, [None,])
ph_batch_size = tf.placeholder(tf.int32,[])
ph_sen_len = tf.placeholder(tf.int32,[])

sen_index = x
x_emb = tf.nn.embedding_lookup(word_embed,sen_index)
y_index = y
y_emb = tf.nn.embedding_lookup(y_embed,y_index)

x_4dim = tf.reshape(x_emb,[-1,ph_sen_len,1,n_embed_size])
#[batch,sen_len,1,emb]


# [2,4,3,1]
# w_conv = tf.Variable(tf.truncated_normal([n_filter_size,n_embed_size,1,n_filter], stddev=0.1))
# # [2,3,1,6]
# b_conv = tf.Variable(tf.constant(0.1, shape=[n_filter]))

# # In[15]:

# h_conv = tf.nn.relu(tf.nn.conv2d(x_4dim, w_conv, strides=[1, 1, 1, 1], padding='VALID') + b_conv)
# [2,2,1,6]
# h_pool = tf.nn.max_pool(h_conv, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='VALID')
# [2,1,1,6]
max_pooled_output = []
for i, filter_size in enumerate(filter_sizes):
    # Convolution layer
    filter_shape = [filter_size, 1 , n_embed_size , n_filter]

    # initialize weight and bias
    weight = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="weight")
    bias = tf.Variable(tf.constant(0.1, shape=[n_filter]), name="bias")
    conv = tf.nn.conv2d(
        x_4dim,
        weight,
        strides=[1,1,1,1],
        padding="SAME",    #narrow conv
        name="conv"
    )
    # 過relu
    h = tf.nn.relu(tf.nn.bias_add(conv, bias), name="relu")
    #[batch,seq_len,1,fil]

    # max-pooling
    # ksize 這邊的 第二維 就是指filter 總共會滑過去幾次 -> sequence_length - filter_size + 1
    # input : [1,0,3,4] , filter_size : 2 , output: [[1,0],[0,3],[3,4]] -> length : 4 - 2 + 1 = 3 
    pooled = tf.nn.max_pool(
        h,
        ksize=[1,n_max_pool, 1, 1],     
        strides=[1, n_max_pool, 1, 1],
        padding='VALID',
        name="pool" 
    )
    #[batch,seq_len/2,1,fil]
    max_pooled_output.append(pooled)
    #[batch,seq_len/2,1,fil]*3
# 合併 所有的 feature
num_filters_total = n_filter * len(filter_sizes)
h_pool = tf.concat(max_pooled_output, 3)
#[batch,seq_len/2,1,fil*3]
# h_pool_flat = tf.reshape(h_pool, [-1,num_filters_total])


# In[16]:

RNN_input = tf.reshape(h_pool,[ph_batch_size,tf.shape(h_pool)[1],-1])


# # RNN

# In[17]:

inW = tf.Variable(tf.random_normal([n_filter*n_filter_kind, n_hidden_units]))
outW = tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
inB = tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ]))
outB = tf.Variable(tf.constant(0.1, shape=[n_classes, ]))


# In[18]:

n_steps = tf.shape(RNN_input)[1]
RNN_input1 = tf.reshape(RNN_input,[-1,n_filter*n_filter_kind])#[batch*seq_len/2,fil*3]
X_in = tf.matmul(RNN_input1, inW) + inB#[batch*seq_len/2,hidden]
X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])
##[batch,seq_len/2,hidden]
X_in = tf.nn.relu(X_in)
lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0)
mul_lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell]*rnn_layer_num)
outputs, final_state = tf.nn.dynamic_rnn(mul_lstm, X_in,dtype=tf.float32)
##[batch,seq_len/2,hidden]
outputs = tf.transpose(outputs, [1, 0, 2])[-1]
#[1,batch,hidden]
outputs1 = tf.reshape(outputs,[-1,n_hidden_units])
#[batch,hidden]
pred = tf.matmul(outputs1, outW) + outB
#[batch,n_class]

# In[19]:

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_emb, logits=pred))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y_emb, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))    


# In[20]:

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# a = sess.run(word_embed)
# print(a)
# In[ ]:

saver = tf.train.Saver()
for i in range(training_iters):
    is_eoi,is_eof,score,cmmt,slen,batch_size = get_batch(n_batch_size)
    batch_now = 0
    while(not is_eoi):
        batch_now += 1
        if batch_size == 0:
            is_eoi,is_eof,score,cmmt,slen,batch_size = get_batch(n_batch_size)
            continue
        inp = {x:cmmt,y:score,ph_batch_size:batch_size,ph_sen_len:slen}
        _,c,a = sess.run([train_op,cost,accuracy],inp)
        print("iter:%d batch:%d loss:%f accuracy:%f %d"%(i,batch_now,c,a,slen))
        # if(batch_now % 10 == 0):
        #     loss,acc = sess.run([cost,accuracy],inp)
        #     print("iter:%d batch:%d loss:%f accuracy:%f"%(i,batch_now,loss,acc))`
        # if (batch_now % 100 == 0):
        #     save_path = saver.save(sess,"./RCNN3_model.bin")
        is_eoi,is_eof,score,cmmt,slen,batch_size = get_batch(n_batch_size)

