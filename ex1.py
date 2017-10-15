import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions


def mnist_test():
    train, test = datasets.get_mnist()
    train_iter = iterators.SerialIterator(train, batch_size=100, shuffle=True)
    test_iter = iterators.SerialIterator(test, batch_size=100, repeat=False, shuffle=False)

    class MLP(Chain):
        def __init__(self, n_units, n_out):
            super(MLP, self).__init__()
            with self.init_scope():
                self.l1 = L.Linear(None, n_units)
                self.l2 = L.Linear(None, n_units)
                self.l3 = L.Linear(None, n_out)

        def __call__(self, x):
            h1 = F.relu(self.l1(x))
            h2 = F.relu(self.l2(h1))
            y = self.l3(h2)
            return y

    class Classifier(Chain):
        def __init__(self, predictor):
            super(Classifier, self).__init__()
            with self.init_scope():
                self.predictor = predictor

        def __call__(self, x, t):
            print(x, t)
            y = self.predictor(x)
            loss = F.softmax_cross_entropy(y, t)
            accuracy = F.accuracy(y, t)
            report({'loss': loss, 'accuracy': accuracy}, self)
            return loss

    model = Classifier(MLP(100, 10))
    optimizer = optimizers.SGD()
    optimizer.setup(model)

    updater = training.StandardUpdater(train_iter, optimizer)
    trainer = training.Trainer(updater, (6, 'epoch'), out='result')

    trainer.extend(extensions.Evaluator(test_iter, model))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(['epoch', 'main/accuracy', 'validation/main/accuracy']))
    trainer.extend(extensions.ProgressBar())

    trainer.run()

if __name__=='__main__':
    mnist_test()
