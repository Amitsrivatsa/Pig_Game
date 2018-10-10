import numpy as np
import numpy as np
try:
    from Modules.im2col_cython import col2im_cython, im2col_cython
    from Modules.im2col_cython import col2im_6d_cython
except ImportError:
    print ('run the following from the cs231n directory and try again:')
    print ('python setup.py build_ext --inplace')
    print ('You may also need to restart your iPython kernel')

from Modules.im2col import *
# Common ReLu Layer implementation
def relu_forward(x):

    out = np.maximum(0, x)
    cache = x

    return out, cache


# Common Affine layer Implementation

def affine_forward(x, w, b):        # Final Linear Output Layer
    
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    # dimension
    N = x.shape[0]
    D = w.shape[0]

    xreshaped = x.reshape((N,D))
    out = xreshaped.dot(w) + b

    cache = (x, w, b)
    return out, cache



# Common Convolutional layer implementation

def conv_forward_fast(x, w, b, conv_param):
    """
  A fast implementation of the forward pass for a convolutional layer
  based on im2col and col2im.
  """
    N, C, H, W =x.shape
    num_filters=w.shape[0]
    filter_height=filter_width = w.shape[2]
    stride, pad = conv_param['stride'], conv_param['pad']
    N=int(N)
    num_filters=int(num_filters)
    # Check dimensions
    assert (W + 2 * pad - filter_width) % stride == 0, 'width does not work'
    assert (H + 2 * pad - filter_height) % stride == 0, 'height does not work'

    # Create output
    out_height = int((H + 2 * pad - filter_height) / stride) + 1
    out_width = int((W + 2 * pad - filter_width) / stride )+ 1
    out = np.zeros((N, num_filters, out_height, out_width), dtype=int)

    # x_cols = im2col_indices(x, w.shape[2], w.shape[3], pad, stride)
    x_cols = im2col_cython(x, w.shape[2], w.shape[3], pad, stride)
    res = w.reshape((w.shape[0], -1)).dot(x_cols) + b.reshape(-1, 1)

    out = res.reshape(w.shape[0], out.shape[2], out.shape[3], x.shape[0])
    out = out.transpose(3, 0, 1, 2)

    cache = (x, w, b, conv_param, x_cols)
    return out, cache

# Conv_relu_pool_forward Layer (1st Layer)

def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
    """
    Convenience layer that performs a convolution, a ReLU, and a pool.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    - pool_param: Parameters for the pooling layer

    Returns a tuple of:
    - out: Output from the pooling layer
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    s, relu_cache = relu_forward(a)
    out, pool_cache = max_pool_forward_fast(s, pool_param)
    cache = (conv_cache, relu_cache, pool_cache)
    
    return out, cache



# Affine_relu_forward Layer (2nd layer)

def affine_relu_forward(x, w, b):
  

  a, fc_cache = affine_forward(x, w, b)
  
  out, relu_cache = relu_forward(a)
 
  cache = (fc_cache, relu_cache)
  
  return out, cache


def max_pool_forward_fast(x, pool_param):
    """
  A fast implementation of the forward pass for a max pooling layer.

  This chooses between the reshape method and the im2col method. If the pooling
  regions are square and tile the input image, then we can use the reshape
  method which is very fast. Otherwise we fall back on the im2col method, which
  is not much faster than the naive method.
  """
    N, C, H, W = x.shape
    pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']
    stride = pool_param['stride']

    same_size = pool_height == pool_width == stride
    tiles = H % pool_height == 0 and W % pool_width == 0
    if same_size and tiles:
        out, reshape_cache = max_pool_forward_reshape(x, pool_param)
        cache = ('reshape', reshape_cache)
    else:
        out, im2col_cache = max_pool_forward_im2col(x, pool_param)
        cache = ('im2col', im2col_cache)
    return out, cache

def max_pool_forward_reshape(x, pool_param):
    """
  A fast implementation of the forward pass for the max pooling layer that uses
  some clever reshaping.

  This can only be used for square pooling regions that tile the input.
  """
    N, C, H, W = x.shape
    pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']
    stride = pool_param['stride']
    assert pool_height == pool_width == stride, 'Invalid pool params'
    assert H % pool_height == 0
    assert W % pool_height == 0
    x_reshaped = x.reshape(int(N), int(C), int(H / pool_height), int(pool_height),int(W / pool_width), int(pool_width))
    out = x_reshaped.max(axis=3).max(axis=4)

    cache = (x, x_reshaped, out)
    return out, cache

def max_pool_forward_im2col(x, pool_param):
    """
  An implementation of the forward pass for max pooling based on im2col.

  This isn't much faster than the naive version, so it should be avoided if
  possible.
  """
    N, C, H, W = x.shape
    pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']
    stride = pool_param['stride']

    assert (H - pool_height) % stride == 0, 'Invalid height'
    assert (W - pool_width) % stride == 0, 'Invalid width'

    out_height = ((H - pool_height) / stride) + 1
    out_width = ((W - pool_width) / stride) + 1

    x_split = x.reshape(N * C, 1, H, W)
    x_cols = im2col(x_split, pool_height, pool_width, padding=0, stride=stride)
    x_cols_argmax = np.argmax(x_cols, axis=0)
    x_cols_max = x_cols[x_cols_argmax, np.arange(x_cols.shape[1])]
    out = x_cols_max.reshape(int(out_height), int(out_width), N, C).transpose(2, 3, 0, 1)

    cache = (x, x_cols, x_cols_argmax, pool_param)
    return out, cache



