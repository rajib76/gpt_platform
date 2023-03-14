from abc import abstractmethod, ABC


class VectorOps(ABC):
    def __init__(self):
        self.module ='VectorOps'

    @abstractmethod
    def create_index(self,**kwargs):
        pass

