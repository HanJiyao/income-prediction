import numpy as np
import matplotlib.pyplot as plt
# import sys

# X_train_fpath = sys.argv[1]
# Y_train_fpath = sys.argv[2]
# X_test_fpath = sys.argv[3]
# output_fpath = sys.argv[4]

X_train_fpath = 'data/X_train'
Y_train_fpath = 'data/Y_train'
X_test_fpath = 'data/X_test'
output_fpath = 'data/output.csv'


def _normalize_column_0_1(x, trained=True, specified_column=None, x_min=None, x_max=None):
    if trained:
        if specified_column is None:
            specified_column = np.arange(x.shape[1])
        length = len(specified_column)
        x_max = np.reshape(np.max(x[:, specified_column], 0), (1, length))
        x_min = np.reshape(np.min(x[:, specified_column], 0), (1, length))

    x[:, specified_column] = np.divide(np.subtract(x[:, specified_column], x_min), np.subtract(x_max, x_min))

    return x, x_max, x_min


def _normalize_column_normal(x, trained=True, specified_column=None, x_mean=None, x_std=None):
    if trained:
        if specified_column is None:
            specified_column = np.arange(x.shape[1])
        length = len(specified_column)
        x_mean = np.reshape(np.mean(x[:, specified_column], 0), (1, length))
        x_std = np.reshape(np.std(x[:, specified_column], 0), (1, length))

    x[:, specified_column] = np.divide(np.subtract(x[:, specified_column], x_mean), x_std)

    return x, x_mean, x_std


def _shuffle(x, y):
    randomize = np.arange(len(x))
    np.random.shuffle(randomize)
    return x[randomize], y[randomize]


def train_dev_split(x, y, dev_size=0.25):
    train_len = int(round(len(x)*(1-dev_size)))
    return x[:train_len], y[:train_len], x[train_len:], y[train_len:]


x_train_data = np.genfromtxt(X_train_fpath, delimiter=',', skip_header=1)
y_train_data = np.genfromtxt(Y_train_fpath, delimiter=',', skip_header=1)
col = [0, 1, 3, 4, 5, 7, 10, 12, 25, 26, 27, 28]
x_train_data, x_mean_data, x_std_data = _normalize_column_normal(x_train_data, specified_column=col)
y_train_data = y_train_data.reshape((-1, 1))


def _sigmoid(z):
    return np.clip(1/(1.0+np.exp(-z)), 1e-6, 1-1e-6)


def get_prob(x, w):
    return _sigmoid(np.dot(x, w))


def infer(x, w):
    return np.round(get_prob(x, w))


def _cross_entropy(y_pred, y_label):
    y_pred = y_pred.reshape((y_pred.shape[0],))
    y_label = y_label.reshape((y_label.shape[0],))
    return -np.dot(y_label, np.log(y_pred))-np.dot((1-y_label), np.log(1-y_pred))


def _gradient_regularization(x, y_label, w, lamda):
    y_pred = get_prob(x, w)
    pred_error = y_label - y_pred
    w_gradient = -2*np.transpose(x).dot(pred_error)/x.shape[0]+lamda*w
    b_gradient = -np.mean(pred_error)
    return w_gradient, b_gradient


def _loss(y_pred, y_label, lamda, w):
    return _cross_entropy(y_pred, y_label) + lamda * np.sum(np.square(w))


def accuracy(y_pred, y_label):
    return np.sum(y_pred == y_label) / len(y_pred)


def train(x_train, y_train):
    dev_size = 0
    x_train, y_train, x_dev, y_dev = train_dev_split(x_train, y_train, dev_size)
    w = np.zeros((x_train.shape[1]+1, 1))

    lamda = 0
    regularize = True
    if regularize:
        lamda = 0.001

    max_iter = 10000
    batch_size = 10
    learning_rate = 0.2
    num_train = len(y_train)
    num_dev = len(y_dev)
    step = 1

    loss_train = []
    loss_validation = []
    train_acc = []
    dev_acc = []
    # learning_rate_w = np.ones((x_train.shape[1], 1))
    for epoch in range(max_iter):
        x_train, y_train = _shuffle(x_train, y_train)
        # w_sum = np.zeros((x_train.shape[1], 1))
        for i in range(int(np.floor(len(y_train)/batch_size))):
            x = x_train[i*batch_size:(i+1)*batch_size]
            y = y_train[i*batch_size:(i+1)*batch_size]
            x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1).astype(float)
            w_gradient, b_gradient = _gradient_regularization(x, y, w, lamda)
            # w_sum += w_gradient ** 2
            w = w - learning_rate / np.sqrt(step) * w_gradient
            step = step + 1

        x_train = np.concatenate((np.ones((x_train.shape[0], 1)), x_train), axis=1).astype(float)
        y_train_pred = get_prob(x_train, w)
        train_acc_item = accuracy(np.round(y_train_pred), y_train)
        train_acc.append(train_acc_item)
        loss_train_item = _loss(y_train_pred, y_train, lamda, w)/num_train
        loss_train.append(loss_train_item)
        x_train = x_train[:, 1:]

        x_dev = np.concatenate((np.ones((x_dev.shape[0], 1)), x_dev), axis=1).astype(float)
        y_dev_pred = get_prob(x_dev, w)
        dev_acc.append(accuracy(np.round(y_dev_pred), y_dev))
        loss_validation.append(_loss(y_dev_pred, y_dev, lamda, w) / num_dev)
        x_dev = x_dev[:, 1:]

        print(str(epoch+1)+'/'+str(max_iter), ':', train_acc_item, loss_train_item)

    return w, loss_train, loss_validation, train_acc, dev_acc


w_data, loss_train_data, loss_validation_data, train_acc_data, dev_acc_data = train(x_train_data, y_train_data)
np.save('data/weight.npy', w_data)
# plt.plot(loss_train_data)
# plt.plot(loss_validation_data)
# plt.legend(['train', 'dev'])
# plt.show()
# plt.plot(train_acc_data)
# plt.plot(dev_acc_data)
# plt.legend(['train', 'dev'])
# plt.show()

x_test = np.genfromtxt(X_test_fpath, delimiter=',', skip_header=1)
x_test, _, _ = _normalize_column_normal(x_test, specified_column=col)
x_test = np.concatenate((np.ones((x_test.shape[0], 1)), x_test), axis=1).astype(float)
w_result = np.load('data/weight.npy')
result = infer(x_test, w_result)
with open(output_fpath, 'w') as f:
    f.write('id,label\n')
    for j, v in enumerate(result):
        f.write('%d,%d\n' % (j+1, v))
