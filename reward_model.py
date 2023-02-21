import tensorflow as tf

class Reward_model(tf.keras.Model):
    def __init__(self, bert_model):
        super(Reward_model, self).__init__()
        self.bert = bert_model
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs, mask, segment):
        outputs = self.bert([inputs, mask, segment])[:2]
        pooled_layer = outputs[1]
        score = self.dense(pooled_layer)
        return score
