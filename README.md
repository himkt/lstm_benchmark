## Benchmark for Chainer's LSTM and PyTorch's LSTM


### GPU

> python benchmark_lstms.py --gpuid 0 --trial 100

https://github.com/himkt/lstm_benchmark/blob/master/result/GPU.summary.md


### CPU

> python benchmark_lstms.py --gpuid -1 --trial 100

https://github.com/himkt/lstm_benchmark/blob/master/result/CPU.summary.md

It looks PyTorch's LSTM is significantly slow.

PyTorch's LSTM [ellapsed times](https://github.com/himkt/lstm_benchmark/blob/master/result/lstm_pytorch_CPU.time.txt) seems to be unstable.

