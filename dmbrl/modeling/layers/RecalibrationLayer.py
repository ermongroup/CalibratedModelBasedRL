import tensorflow as tf

class RecalibrationLayer(tf.keras.layers.Layer):
    def __init__(self, out_dim):
        super(RecalibrationLayer, self).__init__()
        
        shape = [1, out_dim]
        self.out_dim = out_dim

        self.A = self.add_variable(name='A',
                                 shape=shape,
                                 initializer='uniform',
                                 trainable=True, dtype=tf.float32)

        self.B = self.add_variable(name='B',
                                 shape=shape,
                                 initializer='uniform',
                                 trainable=True, dtype=tf.float32)

    def get_vars(self):
        return [self.A, self.B]

    def get_output_dim(self):
        return self.out_dim

    def call(self, x, activation=True):
        out = x * self.A + self.B
        if not activation:
            return out

        return tf.nn.sigmoid(out)

    def inv_call(self, y, activation=True):
        out = y
        if activation:
            out  = tf.log(y/(1 - y))

        return (out - self.B)/self.A
