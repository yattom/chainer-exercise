import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

from tictoctoe import Game, TicTocToe

W_WIN = 1.0
W_LOST = -1.0
W_DRAW = 0.5

def create_dataset():
    train_raw = []
    score_raw = []
    for i in range(3):
        g = Game()
        g.play()
        result, winner = g.result()
        if result == TicTocToe.WIN:
            for b, r in g.playing.record(winner):
                b_num = [0 if v == None else 1 if v == winner else -1 for v in b]
                r_num = [0] * len(b)
                r_num[h] = 1
                train_raw.append(np.array(b_num + r_num, np.float32))
                score_raw.append(1.0)
            for b, r in g.playing.record(TicTocToe.SIDE_O if winner == TicTocToe.SIDE_X else TicTocToe.SIDE_X):
                b_num = [0 if v == None else 1 if v == winner else -1 for v in b]
                r_num = [0] * len(b)
                r_num[h] = 1
                train_raw.append(b_num + r_num)
                score_raw.append(-1.0)
    return datasets.TupleDataset(train_raw, score_raw)



def tictoctoe_test():
    _, test = datasets.get_mnist()
    train = create_dataset()
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
    tictoctoe_test()
