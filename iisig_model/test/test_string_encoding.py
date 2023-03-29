from unittest import TestCase
import numpy as np
from iisig_model.string_encoding import *

class TestSample(TestCase):
    def setUp(self) -> None:
        self.encoded_testing = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    def test_max_corpus_value(self):
        '''This should be the max value in the map_dict'''
        self.assertEqual(len(map_dict), 41)

    def test_get_vec(self):
        self.assertEqual(get_vec(7, 2),
                         [0, 1, 0 ,0, 0, 0, 0],
                         msg='Equal')

    def test_get_matrix(self):
        '''
        And Array [1,2,3,4] should be broadcast into a matrix of shape 4x4
        where the diagonals are have value 1 and the rest are 0.
        This is called an identity matrix

        !This test will not work with other values!
        '''
        sut = get_matrix([1,2,3,4],4)
        ctrl = np.identity(4, dtype='int')
        self.assertTrue((sut & ctrl).any())

    def test_one_hot_input(self):
        sut = one_hot_input('testing', 20)
        self.assertTrue((sut & self.encoded_testing).any())


