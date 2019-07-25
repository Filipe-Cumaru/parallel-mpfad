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
#include <utility> /* std::swap */

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
// #include "ml_include.h"
// #include "ml_epetra_preconditioner.h"

/* BLAS header */
#include "cblas.h"

/* MPI header */
#include <mpi.h>

#include "geoutils.h"
#include "LPEW3.h"

using namespace std;
using namespace moab;

// Enumeration created to make the access to tags more readable.
enum TagsID {global_id, permeability, centroid, dirichlet,
                neumann, source, pressure};

class MPFADSolver {
private:
    Interface *mb;
    MeshTopoUtil *mtu;
    ParallelComm *pcomm;
    Tag tags[7];
    std::map<EntityHandle, std::map<EntityHandle, double> > weights;
    Range dirichlet_nodes;
    Range neumann_nodes;
    Range internal_nodes;
public:
    MPFADSolver ();
    MPFADSolver (Interface *moab_interface);
    void run ();
    void load_file (string fname);
    void write_file (string fname);
private:
    void setup_tags (Tag tag_handles[5]);
    void assemble_matrix (Epetra_CrsMatrix& A, Epetra_Vector& b, Range volumes);
    void set_pressure_tags (Epetra_Vector& X, Range& volumes);
    void init_tags ();
    double get_cross_diffusion_term (double tan[3], double vec[3], double s,
                                    double h1, double Kn1, double Kt1,
                                    double h2, double Kn2, double Kt2,
                                    bool boundary);
    void node_treatment (EntityHandle node, int id_left, int id_right,
                        double k_eq, double d_JI, double d_JK, Epetra_CrsMatrix& A,
                        Epetra_Vector& b);
    void visit_neumann_faces (Epetra_CrsMatrix& A, Epetra_Vector& b, Range neumann_faces);
    void visit_dirichlet_faces (Epetra_CrsMatrix& A, Epetra_Vector& b, Range dirichlet_faces);
    void visit_internal_faces (Epetra_CrsMatrix& A, Epetra_Vector& b, Range internal_faces);
};

#endif
