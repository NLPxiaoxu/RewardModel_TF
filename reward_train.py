#! -*- coding:utf-8 -*-
import os
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from reward_model import Reward_model
from data_helper import rm_data_loader
import copy
import logging
logging.disable(30)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
# tf.config.experimental.set_virtual_device_configuration(gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8000)])


#bert配置
checkpoint_path = "./bert_model/roberta_chinese_base/"
tokenizer = BertTokenizer.from_pretrained(checkpoint_path, lowercase=True, add_special_tokens=True)
bert_model = TFBertModel.from_pretrained(checkpoint_path)


rm_model = Reward_model(bert_model)


lr = 1e-5
epsilon = 1e-06
num_epochs = 15

Data_loader = rm_data_loader(['reward_data/dev.txt','reward_data/train.txt'])
all_inputs = Data_loader.get_data_input()
n = 14000
inputs = all_inputs[:n]
dev_inputs = all_inputs[n:]
print(len(inputs))


def loss_function(rank_reword):
    add_count = 0
    _loss = 0.0
    for i in  range(len(rank_reword)-1):
        for j in  range(i+1, len(rank_reword)):
            _loss += rank_reword[i] - rank_reword[j]
            add_count += 1
    loss = _loss / add_count
    return -loss


def padding(input, seed):
    max_len = max([len(key) for key in input])
    input = [tokenizer.convert_tokens_to_ids(token) for token in input]
    mask = [[1 for _ in k] for k in input]
    segment = [[0 for _ in k] for k in input]
    input_ids = tf.keras.preprocessing.sequence.pad_sequences(input, max_len, padding='post', truncating='post')
    mask_ids = tf.keras.preprocessing.sequence.pad_sequences(mask, max_len, padding='post', truncating='post')
    segment_ids = tf.keras.preprocessing.sequence.pad_sequences(segment, max_len, padding='post', truncating='post')
    if len(seed) == 4:
        return tf.cast(input_ids[seed],tf.int32),  tf.cast(mask_ids[seed],tf.int32), tf.cast(segment_ids[seed],tf.int32)
    else:
        return tf.cast(input_ids,tf.int32),  tf.cast(mask_ids,tf.int32), tf.cast(segment_ids,tf.int32)
print('data loading')

'''
测试集采用accu进行评价
评价方式将一个batch顺序打乱,利用模型打分
打分后结果与随机种子一致则正确结果+1
'''
def evaluate(dev_data):
    acc = 1e-10
    for key in dev_data:
        dev_random = np.random.permutation([0,1,2,3]) #测试数据随机种子
        dev_input,dev_mask, dev_segment = padding(key, dev_random)
        dev_score = rm_model(dev_input, dev_mask, dev_segment)
        value =[k[0] for k in  dev_score.numpy().tolist()]

        _dev_score = sorted(copy.deepcopy(value), reverse=True)
        indics = [value.index(k) for k in _dev_score]
        # print(_dev_score, indics)
        # print(value, dev_random.tolist())
        if indics == dev_random.tolist():
            acc += 1
        
    return acc/len(dev_data)


def pred(_list):
    _input, _mask, _segment = padding(_list, 'N')
    _score = rm_model(_input, _mask, _segment)
    return _score.numpy().tolist()


sen_list = [['[CLS]', '衣', '服', '质', '量', '不', '错', '，', '做', '工', '很', '好', '，', '款', '式', '也', '好', '看', '，', '全', '五', '星', '。', '[SEP]'], ['[CLS]', '豪', '华', '房', '，', '浴', '室', '都', '没', '有', '挂', '毛', '巾', ' ', '衣', '服', ' ', '的', '地', '方', '，', '也', '没', '有', '他', '写', '的', '那', '么', '大', '的', '面', '积', '，', '小', '的', '很', '。', '伤', '心', '死', '了', '再', '也', '不', '会', '去', '了', '[SEP]'], ['[CLS]', '第', '一', '次', '京', '东', '买', '水', '果', '，', '所', '谓', '的', '中', '果', '都', '没', '有', '手', '掌', '大', '，', '有', '些', '失', '望', '，', '一', '星', '是', '给', '快', '递', '员', '的', '！', '[SEP]'], ['[CLS]', '差', '评', '！', '严', '重', '脱', '毛', '！', '湿', '手', '指', '一', '抓', '手', '指', '头', '都', '是', '黑', '的', '！', '便', '宜', '没', '好', '货', '！', '[SEP]']]

global_step = 1
best = 0.50

optimizer = tf.keras.optimizers.Adam(learning_rate=lr, epsilon=epsilon)
checkpoint = tf.train.Checkpoint(optimizer=optimizer, RewardModel=rm_model)
for epoch in range(num_epochs):
    ave_loss = []
    print('Epoch:', epoch + 1)
    for input in inputs:
        input_ids, mask_ids, segmengt_ids = padding(input, 'N')
        global_step += 1
        with tf.GradientTape() as tape:
            score  = rm_model(input_ids, mask_ids, segmengt_ids)
            loss = loss_function(score)[0]
            ave_loss.append(loss)
            if global_step % 1400 == 0:
                print('Batch {} Loss {:.4f}'.format(global_step, loss))
        variables = rm_model.variables
        grads = tape.gradient(loss, variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, variables))

    print('\n')
    print('Ave_Loss {:.4f}'.format(np.mean(ave_loss)))
    print('\n')

    acc = evaluate(dev_inputs)
    print(acc)

    print('-----')
    result = pred(sen_list)
    print(result)
    print('-----')

    '''
    每次保存最优模型
    模型保存基线accu>0.5
    '''
    if acc > best:
        best = acc
        print('saving_model')
        checkpoint.save('./rm_save/reward_model.ckpt')
