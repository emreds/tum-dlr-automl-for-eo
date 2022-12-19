""""
This file contains some constants and common functions to be used
for LHC Sampling and to create {encoded_architecture:nb101_architecture} dictionary file.

Following functions have been modified from https://github.com/kalifou/fitness_landscape_analysis_NAS

"""
from naslib.utils import nb101_api as api
from pyDOE import *

# Useful constants

INPUT = 'input'
OUTPUT = 'output'
CONV3X3 = 'conv3x3-bn-relu'
CONV1X1 = 'conv1x1-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
NUM_VERTICES = 7
MAX_EDGES = 9
EDGE_SPOTS = NUM_VERTICES * (NUM_VERTICES - 1) / 2  # Upper triangular matrix
OP_SPOTS = NUM_VERTICES - 2  # Input/output vertices are fixed
ALLOWED_OPS = [CONV3X3, CONV1X1, MAXPOOL3X3]
ALLOWED_EDGES = [0, 1]  # Binary adjacency matrix

CODING = ['input', 'conv1x1-bn-relu_0',
          'conv1x1-bn-relu_1', 'conv1x1-bn-relu_2',
          'conv1x1-bn-relu_3', 'conv1x1-bn-relu_4',
          'conv3x3-bn-relu_0', 'conv3x3-bn-relu_1',
          'conv3x3-bn-relu_2', 'conv3x3-bn-relu_3',
          'conv3x3-bn-relu_4', 'maxpool3x3_0',
          'maxpool3x3_1', 'maxpool3x3_2',
          'maxpool3x3_3', 'maxpool3x3_4',
          'output']


def encoded_architecture_to_key(encoded_architecture):
    """
    @param encoded_architecture: encoded architecture array
    @return key: encoded array as string to be used a key
    """
    key = ''.join([str(elem) for elem in encoded_architecture])
    return key


def rename_ops(ops):
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


def sample_single_op(NUM_VERTICES, ALLOWED_OPS):
    x = lhs(NUM_VERTICES, samples=1)[0]
    vv = np.floor(x * len(ALLOWED_OPS))
    op = [ALLOWED_OPS[int(k)] for k in vv]
    op[0] = INPUT
    op[-1] = OUTPUT
    return op


def sample_single_configurations_lhs(N_dimensions):
    sum_edges = 0
    while sum_edges != 9:
        v_m = lhs(N_dimensions, samples=1)
        idx = v_m > 0.5
        v_m[idx == True] = 1
        v_m[idx == False] = 0
        sum_edges = sum(v_m[0])
    return v_m[0]


def recover_incidence_matrix(a0, N_l=7):
    mat = np.zeros((N_l, N_l))
    idx_new = 0
    idx_old = 0
    for i in range(N_l):
        idx_new += N_l - i - 1
        values = a0[idx_old:idx_new]
        idx_old = idx_new
        mat[i, i + 1:] = values
    return mat


def sample_single_valid_spec(NUM_vert, allowed_ops, nasbench):
    is_valid = False
    while not is_valid:
        current_op = sample_single_op(NUM_vert, allowed_ops)
        current_config = sample_single_configurations_lhs(N_dimensions=7 * 3)
        current_mat = recover_incidence_matrix(current_config, N_l=NUM_vert)
        current_spec = api.ModelSpec(matrix=current_mat, ops=current_op)
        is_valid = nasbench.is_valid(current_spec)

    return current_mat, current_op, current_spec


def encode_matrix(adj_matrix, ops):
    enc_matrix = np.zeros((len(CODING), len(CODING)))
    pos = [CODING.index(op) for op in ops]
    trans = dict()
    for i, ix in enumerate(pos):
        trans[i] = ix
    i, j = np.nonzero(adj_matrix)
    ix = [trans.get(n) for n in i]
    jy = [trans.get(n) for n in j]
    for p in zip(ix, jy):
        enc_matrix[p] = 1
    encoded = enc_matrix[np.triu_indices(len(CODING), k=1)]
    return encoded.astype(int)


def encode_architecture(arhitecture):
    adj_matrix = arhitecture['module_adjacency']
    ops = rename_ops(arhitecture['module_operations'])
    encoded = encode_matrix(adj_matrix, ops)
    return encoded


def rename_ops_fixed(ops):
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
