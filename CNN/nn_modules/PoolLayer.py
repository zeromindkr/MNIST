import theano
from theano.tensor.signal import downsample

class PoolLayer(object):
	def __init__(self, rng, input, shape, pool_size=(2, 2)):
		self.input = input
		self.params = []

		pooled_out = downsample.max_pool_2d(
			input = input,
			ds = pool_size,
			ignore_border = True
		)
	
		self.output = pooled_out;
		self.shape = (shape[0]/pool_size[0], shape[1]/pool_size[1])