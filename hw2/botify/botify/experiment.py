from enum import Enum

import mmh3


class Treatment(Enum):
    C = 0
    T1 = 1
    T2 = 2
    T3 = 3
    T4 = 4
    T5 = 5
    T6 = 6
    T7 = 7
    T8 = 8
    T9 = 9


class Split(Enum):
    HALF_HALF = 2
    THREE_WAY = 3
    FOUR_WAY = 4
    FIVE_WAY = 5
    SEVEN_WAY = 7
    EIGHT_WAY = 8
    NINE_WAY = 9


class Experiment:
    def __init__(self, name, split):
        self.name = name
        self.split = split
        self.hash = mmh3.hash(self.name)

    def assign(self, user):
        user_hash = mmh3.hash(str(user), self.hash, False)
        return Treatment(user_hash % self.split.value)

    def __repr__(self):
        return "{}:{}".format(self.name, self.split)


class Experiments:
    STICKY_ARTIST = Experiment("STICKY_ARTIST", Split.HALF_HALF)
    AA = Experiment("AA", Split.HALF_HALF)
    I2I = Experiment("I2I", Split.THREE_WAY)
    HSTU = Experiment("HSTU", Split.HALF_HALF)
    SESSION_GRAPH_MF = Experiment("SESSION_GRAPH_MF", Split.HALF_HALF)

    def __init__(self):
        self.experiments = [Experiments.SESSION_GRAPH_MF]
