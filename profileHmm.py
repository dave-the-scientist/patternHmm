""" Implementation of the Viterbi algorithm using the C extension. See the
patternHmm/__init__.py file for more details.
"""
import Viterbi, math
from src.profileHmm_base import Hmm_base

class Hmm(Hmm_base):
    def __init__(self, matchEmissions, transitionProbabilities):
        self.modelSize = 0
        self.transProbs = {}
        self.rawMatchEmissions = {}
        self.columnProbs = []
        super(Hmm, self).__init__(matchEmissions, transitionProbabilities)
        

    # # #  Object setup methods for the C algorithm
    def _setupColumnProbs(self, transProbs):
        """Creates a list that represents the transmission probabilities that
        would be accessed when filling out 1 column of the Viterbi matrix, so
        it doesn't have to be calculated each time. For each tuple the order
        is ('M','I','D') for match, insert, and delete states, and ('M','R','D')
        for the random state."""
        neg_inf = -float('inf')
        def get(*args):
            nums = (transProbs.get(a1, {}).get(a2, 0.0) for a1, a2 in args)
            return [math.log(num) if num else neg_inf for num in nums]
        probs = []
        prevM = 'M%i'%(self.modelSize)
        prevI = 'R'
        prevD = 'D%i'%(self.modelSize)
        for i in range(1, self.modelSize+1):
            M, I, D, = 'M%i'%i, 'I%i'%i, 'D%i'%i
            if i == self.modelSize: I = 'R'
            probs.extend( get((prevM,M), (prevI,M), (prevD,M),
                              (M,I), (I,I), (D,I),
                              (prevM,D), (prevI,D), (prevD,D)) )
            prevM, prevI, prevD = 'M%i'%i, 'I%i'%i, 'D%i'%i
        return probs
    def _setupEmissions(self, symbols):
        """Formats the emission probabilities."""
        neg_inf = -float('inf')
        num_symbols = len(symbols)
        randProb = 1.0 / num_symbols
        l = []
        for i in range(self.modelSize):
            match = 'M%i' % (i+1)
            matchDict = self.rawMatchEmissions[match]
            l.extend([matchDict.get(symb, 0.0)/randProb for symb in symbols])
        return [math.log(num) if num else neg_inf for num in l]
    def _sequenceToInts(self, symbols, sequence):
        num = dict((c,i) for i, c in enumerate(symbols))
        return [num[c] for c in sequence]
    # # #  Creates and traverses the Viterbi matrices in C extension
    def _findPath(self, sequence):
        """Calls the C extension to calculate the most likely sequence of
        states that would generate the given sequence."""
        symbols = sorted(set(sequence))
        seq = self._sequenceToInts(symbols, sequence)
        ems = self._setupEmissions(symbols)
        del(sequence); del(symbols)
        return Viterbi.findPath(list(seq), ems, self.columnProbs)
