##########################
# Code for Ex. #2 in IDL #
##########################


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras import models, layers, optimizers
import tensorflow as tf
import bz2
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import tensorflow.keras.backend as K

import sys

sys.setrecursionlimit(2500)

import os

import loader as ld

train_texts, train_labels, test_texts, test_labels, test_ascii, embedding_matrix, MAX_LENGTH, MAX_FEATURES = ld.get_dataset()

#####################
# Execution options #
#####################

TRAIN = True

RECR = True  # recurrent netowrk (RNN/GRU) or a non-recurrent network

ATTN = False  # use attention layer in global sum pooling or not
LSTM = False  # use LSTM or otherwise RNN
FC_WEIGHTED = False  # use weighted sum to get score in FC model


# Getting activations from model
def get_act(net, name):
    sub_score = [layer for layer in net.layers if name in layer.name][0].output
    # functor = K.function([test_texts]+ [K.learning_phase()], sub_score)

    OutFunc = K.function([net.input], [sub_score])
    return OutFunc([test_texts])[0]


# RNN Cell Code
def RNN(dim, x):
    # Learnable weights in the cell
    Wh = layers.Dense(dim, use_bias=False)
    Wx = layers.Dense(dim)

    # unstacking the time axis
    x = tf.unstack(x, axis=1)

    H = []

    h = tf.zeros_like(Wx(x[0]))
    H.append(h)

    for i in range(len(x)):
        # Apply the basic step in each time step

        h = tf.keras.activations.relu(Wh(H[i - 1]) + Wx(x[i]))

        H.append(h)

    H = tf.stack(H, axis=1)

    return h, H


# GRU Cell Code
def GRU(dim, x):
    # Learnable weights in the cell
    Wzx = layers.Dense(dim)
    Wzh = layers.Dense(dim, use_bias=False)

    Wrx = layers.Dense(dim)
    Wrh = layers.Dense(dim, use_bias=False)

    Wx = layers.Dense(dim)
    Wh = layers.Dense(dim, use_bias=False)

    # unstacking the time axis
    x = tf.unstack(x, axis=1)

    H = []

    h = tf.zeros_like(Wx(x[0]))
    H.append(h)

    for i in range(len(x)):
        z = tf.sigmoid(Wzx(x[i]) + Wzh(H[i - 1]))
        r = tf.sigmoid(Wrx(x[i]) + Wrh(H[i - 1]))
        ht = tf.tanh(Wx(x[i]) + Wh(r * H[i - 1]))
        h = (1 - z) * H[i - 1] + (z * ht)

        H.append(h)

    H = tf.stack(H, axis=1)

    return h, H


# (Spatially-)Restricted Attention Layer
# k - specifies the -k,+k neighbouring words
def restricted_attention(x, k):
    dim = x.shape[2]

    Wq = layers.Dense(dim)
    Wk = layers.Dense(dim)

    wk = Wk(x)

    paddings = tf.constant([[0, 0, ], [k, k], [0, 0]])
    pk = tf.pad(wk, paddings)
    pv = tf.pad(x, paddings)

    keys = []
    vals = []
    for i in range(-k, k + 1):
        keys.append(tf.roll(pk, i, 1))
        vals.append(tf.roll(pv, i, 1))

    keys = tf.stack(keys, 2)
    keys = keys[:, k:-k, :, :]
    vals = tf.stack(vals, 2)
    vals = vals[:, k:-k, :, :]

    query = Wq(x)

    dot_product = tf.linalg.matvec(keys, query) / tf.sqrt(float(dim))

    atten_weights = tf.keras.layers.Softmax(name="atten_weights")(dot_product) # MAKE SURE you have ,name= "atten_weights" in the attention weight step

    val_out = tf.linalg.matvec(tf.transpose(vals, perm=[0, 1, 3, 2]), atten_weights)

    return x + val_out


def model0(x):
    x = layers.Dense(75, activation="relu")(x)
    x = layers.Dense(50, activation="relu")(x)
    x = layers.Dense(25, activation="relu")(x)

    return x


def model1(x):
    x = layers.Dense(75, activation="relu")(x)
    x = layers.Dense(50, activation="relu")(x)
    x = layers.Dense(50, activation="relu")(x)
    x = layers.Dense(25, activation="relu")(x)
    x = layers.Dense(25, activation="relu")(x)

    return x


def model2(x):
    x = layers.Dense(75, activation="relu")(x)
    x = layers.Dense(25, activation="relu")(x)

    return x


def model3(x):
    x = layers.Dense(500, activation="relu")(x)
    x = layers.Dense(250, activation="relu")(x)
    x = layers.Dense(100, activation="relu")(x)

    return x


def model4(x):
    x = layers.Dense(300, activation="relu")(x)
    x = layers.Dense(100, activation="relu")(x)

    return x


def model5(x):
    x = layers.Dense(75, activation="sigmoid")(x)
    x = layers.Dense(50, activation="sigmoid")(x)
    x = layers.Dense(25, activation="sigmoid")(x)

    return x


def model6(x):
    x = layers.Dense(75, activation="sigmoid")(x)
    x = layers.Dense(50, activation="sigmoid")(x)
    x = layers.Dense(50, activation="sigmoid")(x)
    x = layers.Dense(25, activation="sigmoid")(x)
    x = layers.Dense(25, activation="sigmoid")(x)

    return x


def model7(x):
    x = layers.Dense(75, activation="sigmoid")(x)
    x = layers.Dense(25, activation="sigmoid")(x)

    return x


def model8(x):
    x = layers.Dense(500, activation="sigmoid")(x)
    x = layers.Dense(250, activation="sigmoid")(x)
    x = layers.Dense(100, activation="sigmoid")(x)

    return x


def model9(x):
    x = layers.Dense(300, activation="sigmoid")(x)
    x = layers.Dense(100, activation="sigmoid")(x)

    return x


# Building Entire Model
def build_model(dim, model_fun=None):
    sequences = layers.Input(shape=(MAX_LENGTH,))
    embedding_layer = layers.Embedding(MAX_FEATURES, 100, weights=[embedding_matrix], input_length=MAX_LENGTH,
                                       trainable=False)

    # embedding the words into 100 dim vectors

    x = embedding_layer(sequences)

    if not RECR:
        print("non recurrent networks")
        # non recurrent networks

        if ATTN:
            print("attention layer")
            # attention layer
            x = restricted_attention(x, k=5)

        # word-wise FC layers -- MAKE SURE you have ,name= "sub_score" in the sub_scores step
        # E.g., sub_score = layers.Dense(2,name="sub_score")(x)

        x = model_fun(x)

        if FC_WEIGHTED:
            print("fc weighted")
            sub_score = layers.Dense(2, name="sub_score")(x)

            # weighted sum
            weights = tf.keras.activations.softmax(sub_score[:, :, 1])
            x = tf.reduce_sum(weights * sub_score[:, :, 0], axis=1)
        else:
            print("fc not weighted")
            sub_score = layers.Dense(1, name="sub_score")(x)

            # sum
            x = tf.reduce_sum(sub_score, axis=1)

        # final prediction

        x = tf.sigmoid(x)

        predictions = x

    else:
        print("recurrent networks")
        # recurrent networks
        if LSTM:
            print("gru")
            x, _ = GRU(dim, x)
        else:
            print("rnn")
            x, _ = RNN(dim, x)

        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dense(1, activation='sigmoid')(x)

        predictions = x

    model = models.Model(inputs=sequences, outputs=predictions)

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['binary_accuracy']
    )
    return model


def run_rnn_or_gru(model_name):
    accuracy_scores = []
    f1_scores = []
    roc_auc_scores = []
    dimensions = list(range(64, 129, 8))
    for d in dimensions:
        print("dim =", d)
        acc, f1, roc = train_and_test_model(d=d)

        accuracy_scores.append(acc)
        f1_scores.append(f1)
        roc_auc_scores.append(roc)

        print()
        print("********************************************************************")
        print()

    plt.figure()
    plt.title(model_name + " accuracy, f1 and roc scores")
    plt.plot(list(range(1, len(dimensions) + 1)), accuracy_scores, label='accuracy')
    plt.plot(list(range(1, len(dimensions) + 1)), f1_scores, label='f1')
    plt.plot(list(range(1, len(dimensions) + 1)), roc_auc_scores, label='roc')
    plt.xticks(list(range(1, len(dimensions) + 1)), dimensions)
    plt.legend()
    return accuracy_scores


def run_fc(models_fun):
    accuracies = []
    for model_fun in models_fun:
        auc, f1, roc = train_and_test_model(model_fun=model_fun)
        accuracies.append(auc)
    print(" the best model is= ", np.argmax(np.array(accuracies)))
    print(" the best accuracy is= ", np.max(np.array(accuracies)))
    return int(np.argmax(np.array(accuracies)))


def run_fc_with_attention(model_fun):
    auc, f1, roc = train_and_test_model(model_fun=model_fun)


# if model is FC, model_fun is function name of function describing the architecture, input and output is x
def train_and_test_model(d=64, model_fun=None):
    model = build_model(d, model_fun)
    checkpoint_path = "model_save/cp.ckpt"
    if TRAIN:
        print("Training")

        checkpoint_dir = os.path.dirname(checkpoint_path)

        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         save_weights_only=True, save_best_only=True)
        # print(model.summary())

        model.fit(
            train_texts,
            train_labels,
            batch_size=128,
            epochs=15,
            validation_data=(test_texts, test_labels), callbacks=[cp_callback])
    # else:
    model.load_weights(checkpoint_path)
    #############
    # test code #
    #############
    print("Example Predictions:")
    preds = model.predict(test_texts)
    if not RECR:
        sub_score = get_act(model, "sub_score")
    for i in range(4):

        print("-" * 20)

        if not RECR:
            # print words along with their sub_score

            num = min((len(test_ascii[i]), 100))
            print(test_ascii[i])
            print("score:", preds[i])
            print("sub-score for each word:")
            for k in range(num):
                if FC_WEIGHTED:
                    print(test_ascii[i][k] + ": " + str(sub_score[i][k][0]) + " weight: " + str(sub_score[i][k][1]))
                else:
                    print(test_ascii[i][k] + ": " + str(sub_score[i][k][0]))

            print("\n")
        else:
            print(test_ascii[i])
            print(preds[i])

        if preds[i] > 0.5:
            print("Positive")
        else:
            print("Negative")
        print("-" * 20)
    acc = accuracy_score(test_labels, 1 * (preds > 0.5))
    f1 = f1_score(test_labels, 1 * (preds > 0.5))
    roc = roc_auc_score(test_labels, preds)
    print('Accuracy score: {:0.4}'.format(acc))
    print('F1 score: {:0.4}'.format(f1))
    print('ROC AUC score: {:0.4}'.format(roc))

    return acc, f1, roc


if __name__ == '__main__':
    # Question 1:
    # rnn_accuracy_scores = run_rnn_or_gru("RNN")
    # LSTM = True
    # gru_accuracy_scores = run_rnn_or_gru("GRU")
    # LSTM = False
    #
    # print("RNN accuracies:", rnn_accuracy_scores)
    # print("GRU accuracies:", gru_accuracy_scores)
    # plt.show()

    models_fun = [model0, model1, model2, model3, model4, model5, model6, model7, model8, model9]
    # Question 2:
    RECR = False
    run_fc(models_fun)

    # Question 3:
    # FC_WEIGHTED = True
    # best_model = run_fc(models_fun)

    # Question 5:
    # ATTN = True
    # run_fc_with_attention(models_fun[best_model])
