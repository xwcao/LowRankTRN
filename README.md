# LowRankTRN
Code for the paper "Tensor Regression Networks with various Low-Rank Tensor Approximations"

## Low Rank Tensor Regression Layer
<img src="https://i.imgur.com/1z4tpf7.png">

## Example
Inserting TRL to replace fully connected layer is quite simple. Given a convlutional tensor *h_conv2*, one can add one line code ```out = ttrl(tf.nn.relu(h_pool2), [1,1,1,10,1], 10)```. For example, a part of the mnist.py code is given in the following.

```python
# relu layer. input of h_pool to W_conv2 then add bias
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)
		
# second pooling layer then the size will be batchsize*7*7*32
h_pool2 = max_pool_2x2(h_conv2)

# Low rank Tensor Regression Layer
# ttrl : Tensor Train Regression Layer
# trl  : Tucker Regression Layer
# cprl : CP Regression Layer

out = ttrl(tf.nn.relu(h_pool2), [1,1,1,10,1], 10)
```

To change the type of tensor regression layer, change ```ttrl``` to ```trl``` or ```cprl``` (set appropriate ranks as well).

Tensor train regression layer
```python
out = ttrl(tf.nn.relu(h_pool2), [1,1,1,10,1], 10)
```

Tucker regression layer

```python
out = trl(tf.nn.relu(h_pool2), [1,1,1,10], 10)
```

CP regression layer

```python
out = cprl(tf.nn.relu(h_pool2), 5, 10)
```


## TRL.py
* **Tensor Train Regression Layer**
```python
ttrl(x, ranks, n_outputs)
  # INPUTS 
      x         : the input tensor
      ranks     : List. TT rank of the weight tensor W.
      n_outputs : Scalar. the size of the row vector of the output matrix.
  # OUTPUT
      A tensor of size batchsize times the n_outputs
```

* **Tucker Regression Layer**
```python
trl(x, ranks, n_outputs)
  # INPUTS 
      x         : the input tensor
      ranks     : List. The Tucker rank of the weight tensor W.
      n_outputs : Scalar. the size of the row vector of the output matrix.
  # OUTPUT
      A tensor of size batchsize times the n_outputs
```

* **CP Regression Layer**
```python
cprl(x, rank, n_outputs)
  # INPUTS 
      x         : the input tensor
      ranks     : Scalar. CP rank of the weight tensor W.
      n_outputs : Scalar. the size of the row vector of the output matrix.
  # OUTPUT
      A tensor of size batchsize times the n_outputs
```

## Arxiv and Bib
https://arxiv.org/abs/1712.09520
