import numpy as np

class Network:
    def __init__(self, structure: list):
        self.structure = structure
        self.size = len(structure)
        self.weights = []
        self.bias = []