import numpy as np


## Do I want to change the ord() calculation to a custom one to keep the dimensions down??

def get_vec(len_vect, letter_index):
    oh_vector = [0] * len_vect
    oh_vector[letter_index - 1] = 1 #this is because I 1 indexed the letters
    return oh_vector


def get_matrix(encoded, max_value):
    mat = []
    for i in encoded:
        vec = get_vec(max_value, i)
        mat.append(vec)

    return np.asarray(mat)


def one_hot_input(input_str, max_value):
    # encoded = [ord(character) for character in input_str]
    encoded = [map_dict[a] if a in map_dict else 0 for a in input_str]
    matrix = get_matrix(encoded, max_value)
    return matrix


def calculate_corpus_max_value(corpus_array):
    # character_matrix = [[ord(character) for character in corpus_array[i]] for i in range(len(corpus_array))]
    character_matrix = [[map_dict[a] if a in map_dict else 0 for a in corpus_array[i]] for i in
                        range(len(corpus_array))]
    flat_list = [item for sublist in character_matrix for item in sublist]
    return max(flat_list)

map_dict = {'a': 1,
            'b': 2,
            'c': 3,
            'd': 4,
            'e': 5,
            'f': 6,
            'g':7,
            'h':8,
            'i':9,
            'j':10,
            'k':11,
            'l':12,
            'm':13,
            'n':14,
            'o':15,
            'p':16,
            'q':17,
            'r':18,
            's':19,
            't':20,
            'u':21,
            'v':22,
            'w':23,
            'x':24,
            'y':25,
            'z':26,
            '0':27,
            '1':28,
            '2':29,
            '3':30,
            '4':31,
            '5':32,
            '6':33,
            '7':34,
            '8':35,
            '9':36,
            ' ':37,
           '.':38,
           ',':39,
           '@':40,
           '%':41}
