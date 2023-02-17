import random
import json
import numpy as np

class rm_data_loader():
    def __init__(self, data_list) -> None:
        self.data = []
        for data_path in data_list:
            self.data += [key.strip().split('\t') for key in open(data_path, 'r', encoding='utf-8')]
    
    def get_data_input(self):
        self.inputs =  []
        self.attention_mask = []
        self.token_type_ids = []

        for k in self.data:
            text_list = [['[CLS]'] + [t for t in text]+ ['[SEP]'] for text in k]

            self.inputs.append(text_list)

        # c = list(zip(self.inputs, self.attention_mask, self.token_type_ids))
        random.shuffle(self.inputs)
        # self.inputs, self.attention_mask, self.token_type_ids = zip(*c)
        # print('data loading')
        return self.inputs
    
    # def get_batch(self, ids, masks, segments,  global_batch_size):
    #     batch_num = len(ids) // global_batch_size
    #     print('batch_num', str(batch_num))
    #     for i in range(batch_num):
    #         input = ids[global_batch_size * i: global_batch_size * (i + 1)]
    #         mask = masks[global_batch_size * i: global_batch_size * (i + 1)]
    #         segment = segments[global_batch_size * i: global_batch_size * (i + 1)]
    #         yield input, mask, segment


# import os
# file_list = ['./reward_data/'+key for key in os.listdir('./reward_data')]
# Data_loader = rm_data_loader(file_list)
# inputs = Data_loader.get_data_input()
# max_len_input = max([len(key) for key in inputs])

# print(len(inputs))
# print(inputs[0])
