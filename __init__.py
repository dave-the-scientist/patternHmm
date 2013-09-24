""" Profile hidden Markov model for searching for patterns of symbols.

   There are 3 kinds of states in this model, matches (M), insertions (I) which
are random insertions between matches, and deletions (D) which allow one or
matches to be skipped while maintaining the rest of the pattern. There is also a
general random state that is active between  matches.

   This random state connects to the first match and delete states, as well as
looping onto itself. There are any number of positions in the model after this
1st, each containing 1 match, insert, and delete state. Match states connect to
their own insert state, and to the match and delete states in the next model
position. Insert states connect to themselves and to the match and delete states
in the next position. Delete states connect only to the match and delete states
in the next position. There is no insert state in the final model position, so
the final match and delete states connect only to the general random state.

   This package defines one function and one class.
   The function:
-- generate_model_file(modelSize, outfile='new_pattern_model.py') This generates
the options file for a profile HMM of the desired size, saving it to the
specified location. The parameters in this generated model file should be filled
out, specifying the desired pattern to search for. The model file then defines
an object called 'model', which is an instance of the class defined below. This
object should be imported by your run file, and any methods run from there.

   The class:
-- Hmm(matchEmissions, transitionProbabilities) -- Both of these arguments are
dictionaries, and should be created by using the generate_model_file function
described above. The intent is that this object is not used manually, but only
ever created by one of the model files described above. This package actually
defines 2 versions of the Hmm object; one implements the Viterbi algorithm in a
C extension, which runs ~200x faster. If the C extension fails to build on your
system, there is also a Python implementation of the algorithm that will
automatically be used instead. This is generally fast enough, but does require a
working numpy installation. 

   The Hmm class defines several methods:
-- print_path(sequence, cleanSequence=True, printWidth=80) -- Prints out the
full predicted Viterbi path that would generate the given sequence. If this
is being run on a very long sequence and you can guarantee that the sequence is
in the correct case and doesn't contain any unecessary characters, the
cleanSequence argument should be set to False for ~15% faster performance.
-- print_matches(sequence, minimumMatches=None, cleanSequence=True,
printWidth=80) -- Finds the full Viterbi path that would generate the given
sequence, printing out each sub-match. The minimumMatches argument specifies
the minimum number of match states that must be present in some match to be
valid. If left as None, it is set as half of the length of the pattern.
-- find_path(sequence, cleanSequence=True) -- Predicts the full Viterbi sequence
of states that would generate the given sequence, returning a string.
-- find_matches(sequence, minimumMatches=None, cleanSequence=True) -- Runs the
current model on the given sequence, returning the information used to describe
each match within the sequence.
"""

__author__ = 'Dave Curran'
__version__ = '0.4'

__packageName = 'patternHmm'

try:
    import Viterbi
    from profileHmm import *
except ImportError:
    print "\nError importing Viterbi.so; will use a Python implementation instead."
    print "Note that this implementation requires numpy."
    from pyProfileHmm import *
    
__all__ = []

def generate_model_file(modelSize, outfile='new_pattern_model.py', extra_attribs=''):
    """This function generates the emissions and transition probabilities
    for a profile hidden Markov model of given size, outputting the text
    to the specified filename. Creates a Python script file that can be
    imported into a run file. If desired, extra_attribs should be properly formatted
	python strings, assigning values to variables. These may be used by other code
	using this package, but won't impact patternHmm itself."""
    modelSize = int(modelSize)
    if modelSize < 2:
        raise TypeError("the model must be at least 2 positions long. No options file created")
    if not outfile.endswith('.py'): outfile += '.py'
    __generateOptions(modelSize, outfile, extra_attribs)
    print "\nWrote an options file to %s\n" %outfile

def __generateOptions(modelSize, outfile, extra_attribs):
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    buff = ["from %s import Hmm" % __packageName]
    if extra_attribs:
        buff.append('\n'+extra_attribs)
    buff.extend([
            "\n# For theoretical reasons, each row in the two dictionaries below should sum to",
            "# 1.0. However there is nothing in the program that relies on this, and so it is",
            "# not strictly necessary. 'M' refers to the match states, 'I' refers to the",
            "# insert states, 'D' refers to the delete states, and 'R' is a general random",
            "# state, indicating sequence between matches.\n",
            "# Emission probabilities of the Match states. Replace the letters with any string.",
            "matchEmissions = {"])
    for i in range(1, modelSize+1):
        buff.append("\t'M%i': {'%s': 1.0}," % (i, alphabet[(i-1)%26]))
    buff.extend(["\t}", "\n# The transition probabilities leaving each state.",
                 "transitionProbabilities = {",
                 "\t'R':{'R':0.9, 'M1':0.05, 'D1':0.05},"])
    for i in range(1, modelSize):
        buff.append("\t'M%i':{'I%i':0.3, 'M%i':0.6, 'D%i':0.1}," % (i, i, i+1, i+1))
        buff.append("\t'I%i':{'I%i':0.4, 'M%i':0.5, 'D%i':0.1}," % (i, i, i+1, i+1))
        buff.append("\t'D%i':{'M%i':0.9, 'D%i':0.1}," % (i, i+1, i+1))
    buff.append("\t'M%i':{'R':0.9, 'M1':0.1, 'D1':0.0}," % (i+1))
    buff.append("\t'D%i':{'R':1.0, 'M1':0.0, 'D1':0.0}," % (i+1))
    buff.append("\t}\n")
    buff.append("model = Hmm(matchEmissions, transitionProbabilities)\n")
    with open(outfile, 'wb') as f:
        f.write('\n'.join(buff))

