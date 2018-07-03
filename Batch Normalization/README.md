# Batch Normalization
*learning rate를 너무 높게 잡을 경우 gradient가 explode/vanish 하거나, 비정상 local minima에 빠지는 문제발생. 이는 parameter들의 scale 때문, Batch Normalization을 사용시 backpropagation 할 때 parameter의 scale에 영향을 받지 않게 되며, 따라서, learning rate를 크게 잡을 수 있게 되고 이는 빠른 학습을 가능하게 함.
*Batch Normalization의 경우 자체적인 regularization 효과가 있음.
* test - MNIST

syntax: [API](https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization)
syntax: [pdf](https://arxiv.org/pdf/1502.03167.pdf)
