"""Parent class inherited by both versions of the Hmm() object implemented
in this package.
"""
import itertools


class Hmm_base(object):
    def __init__(self, matchEmissions, transitionProbabilities):
        self.modelSize = len(matchEmissions)
        if self.modelSize < 2:
            raise TypeError('The model must be at least 2 positions long')
        self.transProbs = transitionProbabilities
        self.rawMatchEmissions = matchEmissions
        self.columnProbs = self._setupColumnProbs(transitionProbabilities)

    # # # # #  Public Methods  # # # # #
    def print_path(self, sequence, cleanSequence=True, printWidth=80):
        """Given some sequence this predicts and prints the most likely path
        through the current model to generate it, using the Viterbi algorithm."""
        path = self.find_path(sequence, cleanSequence)
        self._printPath(path, sequence, printWidth)
    def print_matches(self, sequence, minimumMatches=None, cleanSequence=True,
                      printWidth=80):
        """Given some sequence this predicts and prints out all of the predicted
        matches to the current model."""
        self._printMatches(
            *self.find_matches(sequence, minimumMatches, cleanSequence),
            printWidth=printWidth)
    
    def find_path(self, sequence, cleanSequence=True):
        """Given some sequence this predicts the most likely path through
        the current model to generate it, returning the path as a string.
        Runs the algorithm 200x faster in C if the Viterbi module is found,
        otherwise the local Python implementation is used."""
        if cleanSequence: sequence = self._cleanSequence(sequence)
        if not sequence: return []
        return self._findPath(sequence)
    def find_matches(self, sequence, minimumMatches=None, cleanSequence=True):
        """Given some sequence, this finds each of the predicted matches.
        Returns 2 lists, the first where each entry is the sequence of states,
        and the second where each entry is the corresponding (index, sequence)."""
        if cleanSequence: sequence = self._cleanSequence(sequence)
        if not sequence: return []
        path = self.find_path(sequence, cleanSequence=False)
        return self._findMatches(path, sequence, minimumMatches)

    # # #  Output Methods
    def _printPath(self, path, sequence, printWidth=80):
        """Given a list of states and some sequence, this prints an alignment
        of the two."""
        if not path or not sequence:
            print 'No path was found.'
            return
        path = path.replace('R','r')
        print '\nSequence of length %i aligned to predicted states:\n' % len(sequence)
        sequence, path = self._lineupSeqPath(sequence, path)
        self._printAligned(sequence, path, printWidth)
        
    
    def _printMatches(self, stateMatches, seqMatches, printWidth=80):
        """Given a list of state matches and sequence matches, prints each
        of them.."""
        i = 0
        for path, (index, sequence) in itertools.izip(stateMatches, seqMatches):
            sequence, path = self._lineupSeqPath(sequence, path)
            i += 1
            ind = '(%i)' % index
            sequence = ind + sequence
            path = ' '*len(ind) + path
            print '\nMatch %i:' % i
            self._printAligned(sequence, path, printWidth)
        print 'Found %i matches in total.' % i

    # # # # #  Private Methods  # # # # #
    # # #  Output and formatting methods
    def _cleanSequence(self, seq):
        return [symb.upper() for symb in itertools.imap(str, seq) if symb.isalnum()]
    def _lineupSeqPath(self, sequence, path):
        """Formats both sequences into strings, with comma-separators if any sequence
        symbol is wider than a single character."""
        maxWidth = max(itertools.imap(len, sequence))
        if maxWidth == 1:
            sequence = ''.join(sequence)
            path = ''.join(path)
        else:
            strFmt = '%%%is' % maxWidth
            sequence = ','.join(strFmt % symb for symb in sequence)
            path = ','.join(strFmt % state for state in path)
        return sequence, path
    def _printAligned(self, seq1, seq2, printWidth=80):
        """Both seqs are assumed to be the same length."""
        for i in xrange(0, len(seq1), printWidth):
            print seq1[i:i+printWidth]
            print seq2[i:i+printWidth]
            print
    def _findMatches(self, filteredStates, seq, minimumMatches):
        if not minimumMatches or minimumMatches < 1:
            minimumMatches = self.modelSize / 2
        i, isMatch = 0, False
        seqMatches, stateMatches = [], []
        seqBuff, stateBuff = [], []
        for state, c in itertools.izip(filteredStates, seq):
            if state == 'R':
                if isMatch:
                    if stateBuff.count('M') >= minimumMatches:
                        seqMatches.append((seqBuff[0], seqBuff[1:]))
                        stateMatches.append(stateBuff[:])
                    isMatch = False
            else:
                if not isMatch:
                    seqBuff, stateBuff = [i], []
                    isMatch = True
                if state == 'I':
                    seqBuff.append(c.lower())
                    stateBuff.append(state.lower())
                else:
                    seqBuff.append(c)
                    stateBuff.append(state)
            i += 1
        if isMatch:
            if stateBuff.count('M') >= minimumMatches:
                seqMatches.append((seqBuff[0], seqBuff[1:]))
                stateMatches.append(stateBuff[:])
        return stateMatches, seqMatches
