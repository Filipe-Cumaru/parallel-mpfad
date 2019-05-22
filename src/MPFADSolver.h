#ifndef MPFADSOLVER_H
#define MPFADSOLVER_H

/* C++ STL includes */
#include <iostream>	/* std::cout, std::cin */
#include <numeric>	/* std::accumulate */
#include <cstdlib>	/* calloc, free */
#include <cstdio>	/* printf */
#include <cmath>	/* sqrt, pow */
#include <ctime>
#include <string>
#include <stdexcept>

/* MOAB includes */
#include "moab/Core.hpp"
#include "moab/MeshTopoUtil.hpp"
#ifdef MOAB_HAVE_MPI
#include "moab/ParallelComm.hpp"
#include "MBParallelConventions.h"
#endif

/* Trilinos includes */
#include "Epetra_MpiComm.h"
#include "Epetra_CrsMatrix.h"
#include "Epetra_Map.h"
#include "Epetra_Vector.h"
#include "AztecOO.h"
#include "ml_include.h"
#include "ml_epetra_preconditioner.h"

/* MPI header */
#include <mpi.h>

#define ALL_PROCS -1
#define ALL_DIM -1
#define GHOST_DIM 3
#define BRIDGE_DIM 2

using namespace std;
using namespace moab;

// Enumeration created to make the access to tags more readable.
enum TagsID {global_id, permeability, centroid, dirichlet,
                neumann, source, pressure};

class MPFADSolver {
private:
    Interface *mb;
    MeshTopoUtil *topo_util;
    ParallelComm *pcomm;
    Tag tags[7];
public:
    MPFADSolver ();
    MPFADSolver (Interface *moab_interface);
    void run ();
    void load_file (string fname);
    void write_file (string fname);
private:
    void setup_tags (Tag tag_handles[5]);
    void assemble_matrix (Epetra_CrsMatrix& A, Epetra_Vector& b, Range volumes, Tag* tag_handles);
    void set_pressure_tags (Epetra_Vector& X, Range& volumes);
};

#endif
