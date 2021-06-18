import numpy as np

class Conv2D:
    '''
    An implementation of the convolutional layer. We convolve the input with out_channels different filters
    and each filter spans all channels in the input.
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        '''
        :param in_channels: the number of channels of the input data
        :param out_channels: the number of channels of the output(aka the number of filters applied in the layer)
        :param kernel_size: the specified size of the kernel(both height and width)
        :param stride: the stride of convolution
        :param padding: the size of padding. Pad zeros to the input with padding size.
        '''
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.cache = None

        self._init_weights()

    def _init_weights(self):
        np.random.seed(1024)
        self.weight = 1e-3 * np.random.randn(self.out_channels, self.in_channels,  self.kernel_size, self.kernel_size)
        self.bias = np.zeros(self.out_channels)

        self.dx = None
        self.dw = None
        self.db = None

    def forward(self, x):
        '''
        The forward pass of convolution
        :param x: input data of shape (N, C, H, W)
        :return: output data of shape (N, self.out_channels, H', W') where H' and W' are determined by the convolution
                 parameters. Save necessary variables in self.cache for backward pass
        '''
        out = None
        #############################################################################
        # TODO: Implement the convolution forward pass.                             #
        # Hint: 1) You may use np.pad for padding.                                  #
        #       2) You may implement the convolution with loops                     #
        #############################################################################
        kernel_size = self.kernel_size
        stride = self.stride
        pad = self.padding
        N, C, H, W = x.shape
        H_ = int(1 + (H + 2*pad - kernel_size) / stride)
        W_ = int(1 + (W + 2*pad - kernel_size) / stride)
        weight = self.weight
        b = self.bias
        #print(kernel_size)
        out = np.random.randn(N, self.out_channels, H_, W_)
        x_padded = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)), 'constant', constant_values = ((0,0),(0,0),(0,0),(0,0)))
        for n in range(N):
            for c in range(self.out_channels):
                for h in range(H_):
                    for w in range(W_):
                        out[n, c, h, w] = np.sum(x_padded[n, :, h*stride : h*stride + kernel_size, w*stride : w*stride + kernel_size] * weight[c, :, :, :])
                out[n, c, :, :] += b[c]
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = x
        return out

    def backward(self, dout):
        '''
        The backward pass of convolution
        :param dout: upstream gradients
        :return: nothing but dx, dw, and db of self should be updated
        '''
        x = self.cache
        #############################################################################
        # TODO: Implement the convolution backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the convolution with loops                     #
        #       2) don't forget padding when computing dx                           #
        #############################################################################
        kernel_size = self.kernel_size
        stride = self.stride
        pad = self.padding
        N, C, H, W = x.shape
        H_ = int(1 + (H + 2*pad - kernel_size) / stride)
        W_ = int(1 + (W + 2*pad - kernel_size) / stride)
        weight = self.weight
        b = self.bias
        x_padded = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)), 'constant', constant_values = ((0,0),(0,0),(0,0),(0,0)))

        dx_padded = np.zeros_like(x_padded)
        dw = np.zeros_like(weight)
        db = np.zeros_like(b)
        for n in range(N):
            for c in range(self.out_channels):
                for h in range(H_):
                    for w in range(W_):
                        dw[c, :, :, :] += dout[n, c, h, w]*x_padded[n, :, h*stride : h*stride + kernel_size, w*stride : w*stride + kernel_size]
                        dx_padded[n, :, h*stride : h*stride + kernel_size, w*stride : w*stride + kernel_size] += dout[n, c, h, w] * weight[c, :, :, :]
                db[c] += np.sum(dout[n, c, :, :])
        dx = dx_padded[:, :, pad:pad + H, pad:pad + W]
        self.dx = dx
        self.dw = dw
        self.db = db
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
