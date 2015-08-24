import numpy as np
import theano
import theano.tensor.nnet.conv as conv

class ConvLayer(object):
	def __init__(self, rng, input, filter_shape, image_shape, pool_size):
		self.input = input

		fan_in = np.prod(filter_shape[1:])
		fan_out = filter_shape[0] * np.prod(filter_shape[2:]) / np.prod(pool_size)
		w_bound = np.sqrt(6. / (fan_in + fan_out))
		np_w = np.asarray(rng.uniform(low=-w_bound, high=w_bound, size=filter_shape), dtype=theano.config.floatX)
		self.w = theano.shared(np_w, borrow=True)

		conv_out = conv.conv2d(
			input=self.input,
			filters=self.w,
			filter_shape=filter_shape,
			image_shape=image_shape
		)

		self.output = conv_out
		self.params = [self.w]

		self.shape = (
			(image_shape[2]-filter_shape[2]+1),
			(image_shape[3]-filter_shape[3]+1)
		)


