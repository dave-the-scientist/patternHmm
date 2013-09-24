#include <Python.h>

#define PROB_DIM  3  // Dimension specific to this kind of profile HMM.

static char module_docstring[] = "This module is meant to be called by the profileHmm.py script. It describes one method, findPath(). This method uses the Viterbi algorithm to find the most likely path through the given hidden Markov model that would generate the given sequence. That path is returned as a string, where M indicates a match state, I an insert state, and R the random state.\n";
static char method_docstring[] = "This method takes 3 arguments, all Python lists. The first is the sequence, a list of ints. The second is a flattened 2D list of doubles describing the emission probabilities, and the third is a flattened 3D list of doubles describing the transition probabilities. This method is meant to be called only by profileHmm.py, which is able to build and format the 3 arguments correctly.\n";

static void pyArraysToCArrays(PyObject *seq_obj, PyObject *ems_obj, PyObject *probs_obj, 
			     int seq_len, int num_symbols, int model_len, int seq[seq_len], 
			     double ems[][num_symbols], double probs[][PROB_DIM][PROB_DIM]) {
  // Copies the Python objects into C arrays.
  Py_ssize_t t;

  // Fills out the 1D 'seq' array from the 'seq_obj' object.
  for (int i=0; i<seq_len; ++i) {
    t = (Py_ssize_t) i;
    seq[i] = (int)PyInt_AsSsize_t(PyList_GetItem(seq_obj, t));
  }

  // Fills out the 2D 'ems' array from the 'ems_obj' object.
  for (int m=0; m<model_len; ++m) {
    for (int n=0; n<num_symbols; ++n) {
      t = (m * num_symbols) + n;
      ems[m][n] = PyFloat_AsDouble(PyList_GetItem(ems_obj, t));
    } }

  // Fills out the 3D 'probs' array from the 'probs_obj' object.
  for (int m=0; m<model_len; ++m) {
    for (int i=0; i<PROB_DIM; ++i) {
      for (int j=0; j<PROB_DIM; ++j) {
	t = (m * PROB_DIM * PROB_DIM) + (i * PROB_DIM) + j;
	probs[m][i][j] = PyFloat_AsDouble(PyList_GetItem(probs_obj, t));
      } } }
}

/* Utility functions used in calculateStep and fillMatrices.*/
static int max(double vec[3]) {
  if (vec[2] >= vec[1] && vec[2] >= vec[0]) {
    return 2;
  } else if (vec[0] >= vec[1] && vec[0] >= vec[2]) {
    return 0;
  } else {
    return 1; 
  }
}
static void vecAdd(double a[3], double b[3], double out[3]) {
  out[0] = a[0] + b[0];
  out[1] = a[1] + b[1];
  out[2] = a[2] + b[2];
}

/* This function fills out a single entry in the vs and paths matrices.*/
static void calculateStep(int i, int j, int prevJ, double emsScore,
			  int seq_len, double vs[][seq_len + 1][PROB_DIM],
			  double probs[][PROB_DIM][PROB_DIM],
			  int paths[][seq_len + 1][PROB_DIM]) {
  int ind;
  double tempVs[3];
  vecAdd(probs[j][0], vs[prevJ][i-1], tempVs); // First state.
  ind = max(tempVs);
  paths[j][i][0] = ind;
  vs[j][i][0] = tempVs[ind] + emsScore;
  vecAdd(probs[j][1], vs[j][i-1], tempVs); // Second state.
  ind = max(tempVs);
  paths[j][i][1] = ind;
  vs[j][i][1] = tempVs[ind];
  vecAdd(probs[j][2], vs[prevJ][i], tempVs); // Third state
  ind = max(tempVs);
  paths[j][i][2] = ind;
  vs[j][i][2] = tempVs[ind];
}

static void fillMatrices(int *seq, int model_len, int num_symbols, int seq_len,
			 double ems[model_len][num_symbols], 
			 double probs[][PROB_DIM][PROB_DIM],
			 double vs[model_len][seq_len + 1][PROB_DIM],
			 int paths[model_len][seq_len + 1][PROB_DIM]) {
  /* Variables and initialization.*/
  int m = model_len - 1;
  int n = seq_len + 1;
  int symbol;
  double tempVs[3];
  double nInf = -1.0/0.0;

  for (int i=0; i<seq_len+1; ++i) {
    for (int j=0; j<model_len; ++j) {
      for (int s=0; s<PROB_DIM; ++s) {
	paths[j][i][s] = 1;
	vs[j][i][s] = nInf;
      } } }

  vs[m][0][1] = probs[m][1][1];  // These are the 
  vs[0][0][0] = probs[0][0][1];  // starting probabilities
  vs[0][0][2] = probs[0][2][1];  // for this type of model.

  /* Fill out vs and paths matrices.*/
  for (int i=1; i<n; ++i) {
    vecAdd(vs[m][i-1], probs[m][1], tempVs); // The final state for all model
    vs[m][i][1] = tempVs[max(tempVs)];       // positions must be partially calculated
    symbol = seq[i-1];                       // before the rest can be.
    calculateStep(i, 0, m, ems[0][symbol], seq_len, vs, probs, paths);
    for (int j=1; j<model_len; ++j)
      calculateStep(i, j, j-1, ems[j][symbol], seq_len, vs, probs, paths);
  }
}

static void findMaxCoords(int *maxJ, int *maxS, int seq_len,
			  int model_len, double vs[][seq_len+1][PROB_DIM]) {
  double largest = vs[0][seq_len][0];
  for (int j=0; j<model_len; ++j) {
    for (int s=0; s<PROB_DIM; ++s) {
      if (vs[j][seq_len][s] >= largest) {
	largest = vs[j][seq_len][s];
	*maxJ = j;
	*maxS = s;
      }
    } }
}
static void backTrack(int j, int i, int s, int m,
		      int paths[][i + 1][PROB_DIM], char path[]) {
  /* This function traces backwards through the 'paths' matrix to find the most
  probable calculated path. This is indicated by filling out the 'path' string.
  The j, i, and s ints are used for indexing the paths array, and
  represent the coordinates of the largest value in the final
  column. This means i is the sequence length. The m is model_len-1,
  used to index the final j position.*/
  int ptr = paths[j][i][s];
  path[i] = '\0';
  while (i > 0) {
    switch (s) {
    case 0:
      i -= 1;
      j = (j != 0) ? j - 1 : m;
      path[i] = 'M';
      break;
    case 1:
      i -= 1;
      path[i] = (j == m) ? 'R' : 'I';
      break;
    case 2:
      j = (j != 0) ? j - 1: m;
      break;
    }
    s = ptr;
    ptr = paths[j][i][s];
  }
}

/*******  THE PUBLIC FUNCTION IMPLEMENTED BY THIS EXTENSION.  *******/
static PyObject* vit_findPath(PyObject* self, PyObject* args) {
  PyObject *seq_obj;
  PyObject *ems_obj;
  PyObject *probs_obj;
  if (!PyArg_ParseTuple(args, "OOO", &seq_obj, &ems_obj, &probs_obj)) {
    PyErr_SetString(PyExc_TypeError, "error loading the arguments.");
    return NULL;
  }

  /* Minor variables.*/
  int seq_len = (int)PyList_Size(seq_obj);
  int model_len = (int)PyList_Size(probs_obj) / (PROB_DIM * PROB_DIM);
  int num_symbols = (int)PyList_Size(ems_obj) / model_len;
  if (PROB_DIM * PROB_DIM * model_len != (int)PyList_Size(probs_obj)) {
    PyErr_SetString(PyExc_TypeError, "the given probabilities array had the wrong dimensions.");
    return NULL;
  }
  if (num_symbols * model_len != (int)PyList_Size(ems_obj)) {
    PyErr_SetString(PyExc_TypeError, "the given emmissions array had the wrong dimensions.");
    return NULL;
  }

  /* Main objects. Also 3 of the 4 'malloc' calls made.*/
  double ems[model_len][num_symbols];
  double probs[model_len][PROB_DIM][PROB_DIM];
  int *seq = malloc(seq_len * sizeof(int));
  double (*vs)[seq_len + 1][PROB_DIM] = malloc(model_len * sizeof *vs);
  int (*paths)[seq_len + 1][PROB_DIM] = malloc(model_len * sizeof *paths);
  if (vs == NULL || paths == NULL || seq == NULL) {
    PyErr_SetString(PyExc_MemoryError, "unable to malloc C arrays.");
    return NULL;
  }

  /* Load the given PyObject lists into the above C arrays, then fill out
   the Viterbi and paths matrices.*/
  pyArraysToCArrays(seq_obj, ems_obj, probs_obj, seq_len, num_symbols, 
		    model_len, seq, ems, probs);
  fillMatrices(seq, model_len, num_symbols, seq_len, ems, probs, vs, paths);
  free(seq);

  /* Find most probable path and trace backwards. Final 'malloc' call.*/
  char *path = malloc((seq_len+1) * sizeof(char *));
  if (path == NULL) {
    PyErr_SetString(PyExc_MemoryError, "unable to malloc C array.");
    return NULL;
  }
  int maxJ = 0;
  int maxS = 0;
  findMaxCoords(&maxJ, &maxS, seq_len, model_len, vs);
  backTrack(maxJ, seq_len, maxS, model_len-1, paths, path);

  /* Create return value, free memory and return. */
  PyObject *ret = Py_BuildValue("s", path);
  free(vs);
  free(paths);
  free(path);
  return ret;
}

/* Necessary extension magic.*/
static PyMethodDef module_methods[] = {
  {"findPath", vit_findPath, METH_VARARGS, method_docstring},
  {NULL, NULL, 0, NULL}
};
PyMODINIT_FUNC initViterbi(void) {
  PyObject *m = Py_InitModule3("Viterbi", module_methods, module_docstring);
  if (m == NULL)
    return;
}
