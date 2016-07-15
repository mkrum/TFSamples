#! /usr/bin/env python2.7

import os
import numpy as np
pos = 14 

text     = {0: 'zero',
            1: 'one',
            2: 'two',
            3: 'three',
            4: 'four',
            5: 'five',
            6: 'six',
            7: 'seven',
            8: 'eight',
            9: 'nine',
            10: 'ten'}

operand  = {0: 'plus',
            1: 'minus'}

convert  = {'zero'   : 0,
            'one'    : 1,
            'two'    : 2,
            'three'  : 3,
            'four'   : 4,
            'five'   : 5,
            'six'    : 6,
            'seven'  : 7,
            'eight'  : 8,
            'nine'   : 9,
            'ten'    : 10,
            'plus'   : 11,
            'minus'  : 12,
            'equals' : 13}




def onehot(inp):
    blank = [0.] * pos
    if isinstance(inp, int):
        blank[inp] = 1.
    else:
        blank[convert[inp]] = 1.
    return np.array(blank, dtype=np.float32)

def test_ar(test):
    data = []
    sent = test.split(' ')
    if len(sent) != 4:
        print 'Inputs must be in the form x plus/minus y equals'
        return np.zeros((4,14), dtype=np.int)
    for spl in sent:
        data.append(onehot(spl))
    return np.array(data)

def ar_to_text(ar):
    ind = np.argmax(ar)
    print text[ind]

def text_to_array(sent):
    data = []
    sent = sent.split(' ')
    for spl in sent[:-1]:
        data.append(onehot(spl))
    return np.array(data), np.array(onehot(sent[-1]))

def generate_sentence():
    first   = np.random.randint(6) 
    second  = np.random.randint(6)
    op      = np.random.randint(2)
    if op == 1:
        if first < second:
            first, second = second, first 
        result = first - second
    else:
        result = first + second
    return text[first]+' '+operand[op]+' '+text[second]+' equals '+text[result]

def get_n_vals(n):
    dvals = []
    lvals = []
    for _ in xrange(n):
        data, label = text_to_array(generate_sentence())
        dvals.append(data)
        lvals.append(label)
    return np.array(dvals), np.array(lvals)

if __name__ == '__main__':
    print get_n_vals(20)
