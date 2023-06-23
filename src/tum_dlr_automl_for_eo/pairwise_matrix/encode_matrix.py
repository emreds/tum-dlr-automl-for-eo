from typing import List

import numpy as np
import pandas as pd

INPUT = 'input'
OUTPUT = 'output'
CONV3X3 = 'conv3x3-bn-relu'
CONV1X1 = 'conv1x1-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
NUM_VERTICES = 7
MAX_EDGES = 9

class ArchitectureEncoder:
    """
    Encodes the architecture into a matrix.
    It encodes the architecture into 17x17 matrix.
    
    NOTE: Since we have 5 layers, but the first and last layer are fixed,
    we have 3x5=15 variant operations and 2 fixed operations.
    This means we can encode every architecture in a 17x17 matrix.
    """
    def __init__(self):
        self.CODING = [
            "input",
            "conv1x1-bn-relu_0",
            "conv1x1-bn-relu_1",
            "conv1x1-bn-relu_2",
            "conv1x1-bn-relu_3",
            "conv1x1-bn-relu_4",
            "conv3x3-bn-relu_0",
            "conv3x3-bn-relu_1",
            "conv3x3-bn-relu_2",
            "conv3x3-bn-relu_3",
            "conv3x3-bn-relu_4",
            "maxpool3x3_0",
            "maxpool3x3_1",
            "maxpool3x3_2",
            "maxpool3x3_3",
            "maxpool3x3_4",
            "output",
        ]
    
    def _encode_matrix(self, adj_matrix:List, ops: List[str]) -> np.ndarray:
        """
        Encodes the given architecture into a 17x17 matrix.
        Converts the given operations to the CODING format.
        Args:
            adj_matrix (List): _description_
            ops (List[str]): _description_

        Returns:
            np.ndarray: _description_
        """
        
        
        enc_matrix = np.zeros((len(self.CODING), len(self.CODING)))
        pos = [self.CODING.index(op) for op in ops]
        trans = dict()
        for i, ix in enumerate(pos):
            trans[i] = ix
        i, j = np.nonzero(adj_matrix)
        ix = [trans.get(n) for n in i]
        jy = [trans.get(n) for n in j]
        for p in zip(ix, jy):
            enc_matrix[p] = 1
        return enc_matrix
    
    @staticmethod
    def _rename_ops(ops: List[str]) -> List[str]:
        """
        Converts the given operations to the CODING format.

        Args:
            ops (List[str]): _description_

        Returns:
            List[str]: _description_
        """
        c1x1 = 0
        c3x3 = 0
        mp3x3 = 0
        new_ops = []
        for op in ops:
            if op == CONV1X1:
                new_ops = new_ops + [op + "_" + str(c1x1)]
                c1x1 = c1x1 + 1
            elif op == CONV3X3:
                new_ops = new_ops + [op + "_" + str(c3x3)]
                c3x3 = c3x3 + 1
            elif op == MAXPOOL3X3:
                new_ops = new_ops + [op + "_" + str(mp3x3)]
                mp3x3 = mp3x3 + 1
            else:
                new_ops = new_ops + [op]
        return new_ops

    def encode_architecture(self, adj_matrix:List, ops:List[str]) -> np.ndarray:
        """
        Encodes the given architecture into a 17x17 matrix.

        Args:
            adj_matrix (List): _description_
            ops (List[str]): _description_

        Returns:
            np.ndarray: _description_
        """
        #print(f"encode_architecture: {locals().keys()}")
        
        renamed_ops = self._rename_ops(ops)
        encoded = self._encode_matrix(adj_matrix, renamed_ops)
        return encoded
