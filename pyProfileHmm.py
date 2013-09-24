"""Python implementation of the Viterbi algorithm. Requires numpy. See the
patternHmm/__init__.py file for more details."""
import numpy as np
import warnings
from src.profileHmm_base import Hmm_base

class Hmm(Hmm_base):
    def __init__(self, matchEmissions, transitionProbabilities):
        self.modelSize = 0
        self.transProbs = {}
        self.rawMatchEmissions = {}
        self.columnProbs = []
        super(Hmm, self).__init__(matchEmissions, transitionProbabilities)

    # # #  Object setup methods for the Python algorithm
    def _setupEmissions(self, symbols):
        d = {}
        randProb = 1.0 / len(symbols)
        for match, ems in self.rawMatchEmissions.items():
            for symb in symbols: d[match,symb] = -np.inf
            for symb, prob in ems.items(): d[match,symb.upper()] = np.log(prob/randProb)
        return d
    def _setupColumnProbs(self, tp):
        """Creates a list that represents the transmission probabilities that
        would be accessed when filling out 1 column of the Viterbi matrix, so
        it doesn't have to be calculated each time. For each tuple the order
        is ('M','I','D') for match, insert, and delete states, and ('M','R','D')
        for the random state."""
        def get(*args):
            return np.log([tp.get(a1, {}).get(a2, 0.0) for a1, a2 in args])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            l = [()]
            for i in range(2, self.modelSize+1):
                M, prevM, I, prevI, D, prevD = 'M%i'%i, 'M%i'%(i-1), 'I%i'%i, \
                                               'I%i'%(i-1), 'D%i'%i, 'D%i'%(i-1)
                if i == self.modelSize: I = 'R'
                t = (i-1, M, np.array([ get((prevM,M), (prevI,M), (prevD,M)),
                                        get((M,I), (I,I), (D,I)),
                                        get((prevM,D), (prevI,D), (prevD,D)) ]))
                l.append(t)
            prevM, prevI, prevD = M, 'R', D
            M, I, D = 'M1', 'I1', 'D1'
            l[0] = (0, M, np.array([ get((prevM,M), (prevI,M), (prevD,M)),
                                     get((M,I), (I,I), (D,I)),
                                     get((prevM,D), (prevI,D), (prevD,D)) ]))
        return l
    # # #  Creating and traversing the Viterbi matrices in Python
    def _findPath(self, sequence):
        paths, finalProbs = self._calculateMatrices(sequence)
        return self._tracePaths(paths, finalProbs)
    def _calculateMatrices(self, sequence):
        """i iterates through the sequence, j through the model. In paths
        0 means the value came from a match state, 1 is from an insert state,
        and 2 is from a delete state. For the random state 1 indicates it
        came from itself. A -1 indicates the end of the path."""
        symbols = set(sequence)
        ems = self._setupEmissions(symbols)
        cps = self._setupColumnProbs(self.transProbs)
        argmax = np.argmax
        nmax = np.max
        m, n = self.modelSize, len(sequence)+1
        vs = np.zeros((3,3))
        Vs = np.zeros((m, n, 3))
        paths = np.zeros((m, n, 3))
        paths[:,0,:] = -1  # First column of paths, indicates end.
        Vs[:] = -np.inf  # Fills matrix with all -inf.
        Vs[-1,0,1] = np.log(self.transProbs['R']['R'])
        Vs[0,0,0] = np.log(self.transProbs['R']['M1'])
        Vs[0,0,2] = np.log(self.transProbs['R']['D1'])
        for i, symb in enumerate(sequence):
            i += 1
            Vs[m-1,i] = nmax(cps[-1][2] + Vs[m-1,i-1])
            for j, matchSymb, probs in cps:
                vs = probs + [Vs[j-1,i-1], Vs[j,i-1], Vs[j-1,i]]
                paths[j,i] = argmax(vs,1)
                nmax(vs,1,Vs[j,i])
                Vs[j,i,0] += ems[matchSymb, symb]
        return paths, Vs[:,n-1]
    def _tracePaths(self, paths, finalProbs):
        seqLen = paths.shape[1] - 1
        modelLen = paths.shape[0]
        states = []
        m = modelLen - 1
        n = int(np.argmax(finalProbs))
        j = n / 3  # Because there are 3 states.
        s = n % 3  # 0,1,2 for match, insert, deletion.
        i = seqLen
        ptr = int(paths[j,i,s])
        while ptr != -1:
            if s == 0:
                i -= 1
                j = j - 1 if j else m
                states.append('M')
            elif s == 1:
                i -= 1
                states.append('R' if j == m else 'I')
            elif s == 2:
                j = j - 1 if j else m
            s, ptr = ptr, int(paths[j,i,ptr])
        return ''.join(states[::-1])
    def _tracePaths2(self, paths, finalProbs):
        seqLen = paths.shape[1] - 1
        modelLen = paths.shape[0]
        states = []
        names = []
        for n in range(1, self.modelSize+1):
            names.append(['M%i'%n, 'I%i'%n, 'D%i'%n])
        names[-1][1] = 'R'
        m = modelLen - 1
        n = int(np.argmax(finalProbs))
        j = n / 3  # Because there are 3 states.
        s = n % 3  # 0,1,2 for match, insert, deletion.
        i = seqLen
        ptr = int(paths[j,i,s])
        while ptr != -1:           
            states.append(names[j][s])
            if s != 1:
                j -= 1
                if j == -1: j = modelLen - 1
            if s != 2: i -= 1
            s, ptr = ptr, int(paths[j,i,ptr])
        return states[::-1]

    
        
