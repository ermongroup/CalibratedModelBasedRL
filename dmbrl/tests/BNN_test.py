import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

from dotmap import DotMap

from dmbrl.modeling.models import BNN
from dmbrl.modeling.layers import FC, RecalibrationLayer

NUM_SAMPLES = 1024
IN_DIM = 100
HIDDEN_DIM = 10
OUT_DIM = 2


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def stub_data():
    X = np.random.random(size=(NUM_SAMPLES, IN_DIM))
    # W_tru = np.random.random(size=(IN_DIM, OUT_DIM))
    # b_tru = 5
    # y = np.matmul(X, W_tru) + b_tru

    W_hidden = np.random.random(size=(IN_DIM, HIDDEN_DIM))
    W_last = np.random.random(size=(HIDDEN_DIM, OUT_DIM))

    y_mid = np.matmul(X, W_hidden) + 5
    y_mid[y_mid < 0] = 0
    # y_mid = sigmoid(y_mid)
    y = np.matmul(y_mid, W_last) + 2

    return (X, y)


def create_bnn(X, y):
    model = BNN(DotMap(name="test"))
    model.add(FC(OUT_DIM, input_dim=IN_DIM, weight_decay=0.0005)) # linear model for simplicity
    model.finalize(tf.train.AdamOptimizer, {"learning_rate": 0.03})

    model.train(X, y, epochs=1000, batch_size=64)
    model.calibrate(X, y)
    # model.plot_calibration(X, y)

    return model

def test_calibration_sampling(bnn, X):
    recalib = True

    inputs = tf.placeholder(shape=X.shape, dtype=tf.float32)
    mean_tf, var_tf = bnn.create_prediction_tensors(inputs)
    predictions_op = bnn.sample_predictions(mean_tf, var_tf, calibrate=recalib)

    predictions, mean, var = bnn.sess.run([predictions_op, mean_tf, var_tf], feed_dict={inputs: X})

    cdfs_normal = norm.cdf(predictions, loc=mean, scale=np.sqrt(var))
    cdfs = bnn.sess.run(bnn.recalibrator(cdfs_normal)) if recalib else cdfs_normal


    print(cdfs_normal.shape)
    for d in range(cdfs.shape[1]):
        cdf_pred = cdfs[:, d]
        cdf_emp = [np.sum(cdf_pred < p)/len(cdf_pred) for p in cdf_pred]
        plt.figure()
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.scatter(cdf_pred, cdf_emp, alpha=0.4)
        plt.show()


def main():
    X, y = stub_data()
    bnn = create_bnn(X, y)
    test_calibration_sampling(bnn, X)


if __name__ == '__main__':
    main()