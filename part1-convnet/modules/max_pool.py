import numpy as np

class MaxPooling:
    '''
    Max Pooling of input
    '''
    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride
        self.cache = None
        self.dx = None

    def forward(self, x):
        '''
        Forward pass of max pooling
        :param x: input, (N, C, H, W)
        :return: The output by max pooling with kernel_size and stride
        '''
        out = None
        #############################################################################
        # TODO: Implement the max pooling forward pass.                             #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                         #
        #############################################################################
        kernel_size = self.kernel_size
        stride = self.stride
        N, C, H, W = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        H_out, W_out = int(1 + (H - kernel_size) / stride), int(1 + (W - kernel_size) / stride)
        out = np.random.randn(N, C, H_out, W_out)
        for n in range(N):
            for c in range(C):
                for h in range(H_out):
                    for w in range(W_out):
                        out[n, c, h, w] = np.max(x[n, c, h*stride : h*stride + kernel_size, w*stride : w*stride + kernel_size])

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = (x, H_out, W_out)
        return out

    def backward(self, dout):
        '''
        Backward pass of max pooling
        :param dout: Upstream derivatives
        :return:
        '''
        x, H_out, W_out = self.cache
        #############################################################################
        # TODO: Implement the max pooling backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                     #
        #       2) You may find np.unravel_index useful                             #
        #############################################################################
        kernel_size = self.kernel_size
        stride = self.stride
        N, C, H, W = x.shape
        dx = np.zeros_like(x)
        for n in range(N):
            for c in range(C):
                for h in range(H_out):
                    for w in range(W_out):
                        search_range = x[n, c, h*stride : h*stride + kernel_size, w*stride : w*stride + kernel_size]
                        #max_id = np.argmax(np.mat(search_range))
                        max_id = np.argmax(search_range)
                        max_h = int(max_id/kernel_size) + h*stride
                        max_w = max_id % kernel_size + w*stride
                        dx[n, c, max_h, max_w] += dout[n, c, h, w]
        self.dx = dx
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
