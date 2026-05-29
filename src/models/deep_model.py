import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional, Layer
import tensorflow.keras.backend as K

# 自定义 Attention 层 (增加项目学术性)
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                               initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                               initializer="zeros")
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        et = K.squeeze(K.tanh(K.dot(x, self.W) + self.b), axis=-1)
        at = K.softmax(et)
        at = K.expand_dims(at, axis=-1)
        output = x * at
        return K.sum(output, axis=1)

class LottoModelEngine:
    def __init__(self, window_size=12, feature_dim=45):
        self.window_size = window_size
        self.feature_dim = feature_dim
        self.model = self._build_model()

    def _build_model(self):
        inputs = Input(shape=(self.window_size, self.feature_dim))
        # 双向 LSTM 捕捉前后联系
        lstm_out = Bidirectional(LSTM(128, return_sequences=True))(inputs)
        # Attention 机制自动分配历史权重
        attention_out = AttentionLayer()(lstm_out)
        
        # 增加网络容量以融合原始数据与形态学特征
        x = Dense(256, activation='relu')(attention_out)
        x = Dropout(0.2)(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        # 预测 45 个号码的出现概率
        outputs = Dense(45, activation='sigmoid')(x)
        
        model = Model(inputs, outputs)
        model.compile(optimizer='adam', loss='binary_crossentropy')
        return model

    def train(self, X, y, epochs=50):
        print(f"[2/4] 正在训练深度混合模型 (Bi-LSTM + Attention)...")
        self.model.fit(X, y, epochs=epochs, batch_size=32, verbose=0)
        print("   - 模型训练完成。")

    def predict(self, last_seq):
        return self.model.predict(last_seq)
