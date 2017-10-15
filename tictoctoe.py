import random

class TicTocToe:
    WIN_LENGTH = 3

    PLAYING = 0
    WIN = 1
    DRAW = -1

    SIDE_O = 1
    SIDE_X = 2

    def __init__(self):
        self._width = self._height = 3
        self._board = [None] * (self._width * self._height)
        self._record = []
        self.result = TicTocToe.PLAYING
        self.winner = None
        self.turn = TicTocToe.SIDE_O

    def board(self, side):
        return [None if v is None else True if v == side else False for v in self._board]

    def record(self, side):
        return [([None if v is None else True if v == side else False for v in b], h)
                for b, s, h in self._record if s == side]

    def move(self, side, pos):
        assert self.turn == side
        assert self._board[pos] is None
        self._record.append((self._board[:], side, pos))
        self._board[pos] = side
        self.judge()
        self.turn = TicTocToe.SIDE_O if side == TicTocToe.SIDE_X else TicTocToe.SIDE_X

    def _pos(self, row, col):
        return row * self._width + col

    def judge(self):
        dirs = [(-1, -1), (-1, 0), (-1, 1),
                ( 0, -1),          ( 0, 1),
                ( 1, -1), ( 1, 0), ( 1, 1)]
        for row in range(self._height):
            for col in range(self._width):
                for d in dirs:
                    s = []
                    for i in range(TicTocToe.WIN_LENGTH):
                        r = row + d[0] * i
                        c = col + d[1] * i
                        if r < 0 or self._height <= r or c < 0 or self._width <= c:
                            break
                        i = self._pos(r, c)
                        s.append(self._board[i])
                    if s == [TicTocToe.SIDE_O] * TicTocToe.WIN_LENGTH:
                        print(s)
                        self.result = TicTocToe.WIN
                        self.winner = TicTocToe.SIDE_O
                        return
                    if s == [TicTocToe.SIDE_X] * TicTocToe.WIN_LENGTH:
                        print(s)
                        self.result = TicTocToe.WIN
                        self.winner = TicTocToe.SIDE_X
                        return
        if None not in self._board:
            self.result = TicTocToe.DRAW

    def dump(self):
        s = "".join(['.' if v is None else 'O' if v == TicTocToe.SIDE_O else 'X' for v in self._board])
        return s[:3] + '\n' + s[3:6] + '\n' + s[6:] + '\n'


class Game:
    def __init__(self, demo=False):
        self.playing = TicTocToe()
        self._demo = demo

    def play(self):
        while self.playing.result == TicTocToe.PLAYING:
            side = self.playing.turn
            board = self.playing.board(side)
            available = [i for i in range(len(board)) if board[i] == None]
            hand = random.choice(available)
            self.playing.move(side, hand)
            if self._demo:
                print(self.playing.dump())
        if self._demo:
            if self.playing.result == TicTocToe.DRAW:
                print("DRAW")
            if self.playing.result == TicTocToe.WIN:
                if self.playing.winner == TicTocToe.SIDE_O:
                    print("O is the WINNER")
                if self.playing.winner == TicTocToe.SIDE_X:
                    print("X is the WINNER")

    def result(self):
        return self.playing.result, self.playing.winner


def demo():
    g = Game(demo=True)
    g.play()


if __name__=='__main__':
    demo()


