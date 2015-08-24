import numpy as np
import theano
import theano.tensor as T
import ConvLayer
import PoolLayer
import ActivateLayer
from HiddenLayer import *
from logistic_sgd import *

class CNN(object):
    def __init__(self, learning_rate=0.1, n_epochs=200, nkerns=[20, 50], batch_size=500, rng=np.random.RandomState(1234), image_size=(28, 28), pool_sizes=[(2, 2), (2,2)], filter_sizes=[(5,5), (5,5)]):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.nkerns = nkerns
        self.batch_size = batch_size

        x = T.matrix('x')
        y = T.ivector('y')
        index = T.lscalar()  # index to a [mini]batch
        
        self.x = x
        self.y = y
        self.index = index

        #input = x.reshape((batch_size, 1, image_size[0], image_size[1]))
        input = x.reshape((x.shape[0], 1, image_size[0], image_size[1]))

        layer_input = input
        layer_shape = image_size;
        params = []
        for kern0, kern1, pool_size, filter_size in zip([1] + nkerns[:-1], nkerns, pool_sizes, filter_sizes):
            
            filter_shape=(kern1, kern0, filter_size[0], filter_size[1])

            convLayer = ConvLayer.ConvLayer(
                rng=rng,
                input=layer_input,
                filter_shape=filter_shape,
                image_shape=(batch_size, kern0, layer_shape[0], layer_shape[1]),
                pool_size=pool_size
            )

            poolLayer = PoolLayer.PoolLayer(
                rng=rng,
                input=convLayer.output,
                shape=convLayer.shape,
                pool_size=pool_size
            )

            activateLayer = ActivateLayer.ActivateLayer(
                input=poolLayer.output,
                activation=T.tanh,
                shape=poolLayer.shape,
                filter_shape=filter_shape,
            )

            layer_input = activateLayer.output
            layer_shape = activateLayer.shape
            params += convLayer.params + poolLayer.params + activateLayer.params

        layer_hidden = HiddenLayer(
            rng=rng,
            input=layer_input.flatten(2), 
            n_in=nkerns[-1] * layer_shape[0] * layer_shape[1],
            n_out=batch_size,
            activation=T.tanh
        )

        layer_regression = LogisticRegression(rng=rng, input=layer_hidden.output, n_in=batch_size, n_out=10)
        self.predict = theano.function(
            inputs=[x], 
            outputs=layer_regression.y_pred
        )

        params += layer_hidden.params + layer_regression.params
        self.params = params
        self.cost = layer_regression.negative_log_likelihood(y)
        self.errors = layer_regression.errors(y)

    def train(self, datasets):
        train_set_x, train_set_y = datasets[0]
        valid_set_x, valid_set_y = datasets[1]
        test_set_x, test_set_y = datasets[2]

        n_train_batches = train_set_x.get_value(borrow=True).shape[0]
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_train_batches /= self.batch_size
        n_valid_batches /= self.batch_size
        n_test_batches /= self.batch_size

        x = self.x
        y = self.y
        index = self.index

        test_model = theano.function(
            [index],
            self.errors,
            givens={
                x: test_set_x[index * self.batch_size: (index + 1) * self.batch_size],
                y: test_set_y[index * self.batch_size: (index + 1) * self.batch_size]
            }
        )

        validate_model = theano.function(
            [index],
            self.errors,
            givens={
                x: valid_set_x[index * self.batch_size: (index + 1) * self.batch_size],
                y: valid_set_y[index * self.batch_size: (index + 1) * self.batch_size]
            }
        )

        grads = T.grad(self.cost, self.params)
        updates = [
            (param_i, param_i - self.learning_rate * grad_i)
            for param_i, grad_i in zip(self.params, grads)
        ]

        train_model = theano.function(
            [index],
            self.cost,
            updates=updates,
            givens={
                x: train_set_x[index * self.batch_size: (index + 1) * self.batch_size],
                y: train_set_y[index * self.batch_size: (index + 1) * self.batch_size]
            }
        )

        ###############
        # TRAIN MODEL #
        ###############
        print '... training'
        # early-stopping parameters
        patience = 10000  # look as this many examples regardless
        patience_increase = 2  # wait this much longer when a new best is
                               # found
        improvement_threshold = 0.995  # a relative improvement of this much is
                                       # considered significant
        validation_frequency = min(n_train_batches, patience / 2)
                                      # go through this many
                                      # minibatche before checking the network
                                      # on the validation set; in this case we
                                      # check every epoch

        best_validation_loss = np.inf
        best_iter = 0
        test_score = 0.
        start_time = timeit.default_timer()

        epoch = 0
        done_looping = False

        while (epoch < self.n_epochs) and (not done_looping):
            epoch = epoch + 1
            for minibatch_index in xrange(n_train_batches):

                iter = (epoch - 1) * n_train_batches + minibatch_index

                if iter % 100 == 0:
                    print 'training @ iter = ', iter
                cost_ij = train_model(minibatch_index)

                if (iter + 1) % validation_frequency == 0:

                    # compute zero-one loss on validation set
                    validation_losses = [validate_model(i) for i
                                         in xrange(n_valid_batches)]
                    this_validation_loss = np.mean(validation_losses)
                    print('epoch %i, minibatch %i/%i, validation error %f %%' %
                          (epoch, minibatch_index + 1, n_train_batches,
                           this_validation_loss * 100.))

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:

                        #improve patience if loss improvement is good enough
                        if this_validation_loss < best_validation_loss *  \
                           improvement_threshold:
                            patience = max(patience, iter * patience_increase)

                        # save best validation score and iteration number
                        best_validation_loss = this_validation_loss
                        best_iter = iter

                        # test it on the test set
                        test_losses = [
                            test_model(i)
                            for i in xrange(n_test_batches)
                        ]
                        test_score = np.mean(test_losses)
                        print(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%') %
                              (epoch, minibatch_index + 1, n_train_batches,
                               test_score * 100.))

                if patience <= iter:
                    done_looping = True
                    break

        end_time = timeit.default_timer()
        print('Optimization complete.')
        print('Best validation score of %f %% obtained at iteration %i, '
              'with test performance %f %%' %
              (best_validation_loss * 100., best_iter + 1, test_score * 100.))
        print >> sys.stderr, ('The code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.2fm' % ((end_time - start_time) / 60.))
        return best_validation_loss


