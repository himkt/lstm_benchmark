import argparse
import chainer
import numpy
import torch
import time


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpuid', type=int, default=-1)
    parser.add_argument('--trial', type=int, default=10)
    args = parser.parse_args()

    n_trial = args.trial

    # batch_size=32, seq_length=30, feature_dim=200
    # -> (seq_length, batch_size, feature_dim)
    shape = (30, 32, 200)
    input_tensor = numpy.random.random(shape)
    input_tensor = input_tensor.astype(numpy.float32)

    # NStepLSTM takes batch-wise inputs
    # -> (batch_size, seq_length, feature_dim)
    input_list_chainer = list(input_tensor.transpose(1, 0, 2))

    # make variables and tensor
    # (seq_length, batch_size, feature_dim)
    input_tensor_chainer = chainer.Variable(input_tensor)
    input_list_chainer = [chainer.Variable(input_vector) for input_vector in input_list_chainer]  # NOQA
    input_tensor_pytorch = torch.Tensor(input_tensor)

    # https://pytorch.org/docs/stable/_modules/torch/nn/modules/rnn.html#LSTM
    lstm_layer = chainer.links.LSTM(200, 200)
    n_step_lstm_layer = chainer.links.NStepLSTM(1, 200, 200, 0.0)
    lstm_layer_pytorch = torch.nn.LSTM(200, 200)

    if args.gpuid >= 0:
        cupy = chainer.cuda.cupy
        input_tensor_chainer.to_gpu()
        [input_vector.to_gpu() for input_vector in input_list_chainer]
        input_tensor_pytorch = input_tensor_pytorch.to('cuda')

        lstm_layer = lstm_layer.to_gpu()
        n_step_lstm_layer = n_step_lstm_layer.to_gpu()
        lstm_layer_pytorch = lstm_layer_pytorch.to('cuda')

    times_lstm_chainer = []
    times_n_step_lstm_chainer = []
    times_lstm_pytorch = []

    for _ in range(n_trial):
        start = time.time()
        # lstm_layer(input_tensor_chainer)  # <- ダメ
        n_step_lstm_layer(None, None, input_list_chainer)
        times_n_step_lstm_chainer.append(time.time() - start)

    def forward_lstm_layer(input_list_chainer):
        for input_vector in input_list_chainer:
            lstm_layer(input_vector)

    for _ in range(n_trial):
        start = time.time()
        forward_lstm_layer(input_list_chainer)
        times_lstm_chainer.append(time.time() - start)

    for _ in range(n_trial):
        start = time.time()
        lstm_layer_pytorch(input_tensor_pytorch)
        times_lstm_pytorch.append(time.time() - start)

    # times_lstm_chainer = sorted(times_lstm_chainer)
    # times_n_step_lstm_chainer = sorted(times_n_step_lstm_chainer)
    # times_lstm_pytorch = sorted(times_lstm_pytorch)

    mean_time_lstm_chainer = numpy.mean(times_lstm_chainer)
    std_time_lstm_chainer = numpy.std(times_lstm_chainer)

    mean_time_n_step_lstm_chainer = numpy.mean(times_n_step_lstm_chainer)
    std_time_n_step_lstm_chainer = numpy.std(times_n_step_lstm_chainer)

    mean_time_lstm_pytorch = numpy.mean(times_lstm_pytorch)
    std_time_lstm_pytorch = numpy.std(times_lstm_pytorch)

    device = 'GPU' if args.gpuid >= 0 else 'CPU'

    with open(f'result/{device}.summary.md', 'w') as result_fp:
        print(f'## {device}\n', file=result_fp)
        print('```', file=result_fp)
        print(f'chainer lstm: {mean_time_lstm_chainer:.4f} ({std_time_lstm_chainer:.4f})', file=result_fp)  # NOQA
        print(f'chainer n_step_lstm: {mean_time_n_step_lstm_chainer:.4f} ({std_time_n_step_lstm_chainer:.4f})', file=result_fp)  # NOQA
        print(f'pytorch lstm: {mean_time_lstm_pytorch:.4f} ({std_time_lstm_pytorch:.4f})', file=result_fp)  # NOQA
        print('```', file=result_fp)

    times_list = [times_lstm_chainer, times_n_step_lstm_chainer, times_lstm_pytorch]  # NOQA
    label_list = ['lstm_chainer', 'n_step_lstm_chainer', 'lstm_pytorch']

    for label, times in zip(label_list, times_list):
        with open(f'result/{label}_{device}.time.txt', 'w') as time_fp:
            for time_item in times:
                print(time_item, file=time_fp)
