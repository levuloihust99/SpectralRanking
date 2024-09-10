import random


class IdxsGenerator:
    """
    Attributes:
        num (int): total number of indices in one epoch
        seed (int): for reproducible
        epoch (int): current epoch
        iteration (int): current iteration
        idxs (list[int]): current indices to be fetched, depends on `epoch`
    """

    def __init__(self, num: int, seed: int):
        self.num = num
        self.seed = seed
        self.__init_state()

    def __init_state(self):
        """Initial state of the idxs generator."""

        self.epoch = 0
        self.iteration = 0
        self.update_idxs()

    def update_idxs(self):
        """Must be called when setting epoch to a different value."""

        self.idxs = list(range(self.num))
        rnd = random.Random(self.seed + self.epoch)
        rnd.shuffle(self.idxs)
        self.__active_epoch = self.epoch

    def __iter__(self):
        return self

    def __next__(self):
        item = self.idxs[self.iteration]
        self.iteration += 1
        if self.iteration == len(self.idxs):
            self.iteration = 0
            self.epoch += 1
            self.update_idxs()
            self.__active_epoch = self.epoch
        return item

    def set_state(self, epoch: int, iteration: int):
        self.epoch = epoch
        self.iteration = iteration
        if self.epoch != self.__active_epoch:
            self.update_idxs()

    def step_back(self):
        self.iteration -= 1
        if self.iteration == -1:
            self.iteration = 0
            self.epoch -= 1
            self.update_idxs()
