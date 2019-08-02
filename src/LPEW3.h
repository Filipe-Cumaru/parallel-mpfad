#ifndef LPEW3_H
#define LPEW3_H

/* C++ STL includes */
#include <iostream>	/* std::cout, std::cin */
#include <numeric>	/* std::accumulate */
#include <cstdlib>	/* calloc, free */
#include <cstdio>	/* printf */
#include <cmath>	/* sqrt, pow */
#include <ctime>
#include <string>
#include <stdexcept>
#include <utility>  /* std::swap */
#include <algorithm>    /* std::copy */
#include <map>

/* MOAB includes */
#include "moab/Core.hpp"
#include "moab/MeshTopoUtil.hpp"
#ifdef MOAB_HAVE_MPI
#include "moab/ParallelComm.hpp"
#include "MBParallelConventions.h"
#endif

/* BLAS header */
#include "cblas.h"

#include "geoutils.h"

using namespace std;
using namespace moab;

class LPEW3 {
private:
    Interface *mb;
    MeshTopoUtil *mtu;
    Tag dirichlet_tag;
    Tag neumann_tag;
    Tag centroid_tag;
    Tag permeability_tag;
    double tau;
public:
    LPEW3 ();
    LPEW3 (Interface *moab_interface);
    void interpolate (EntityHandle node, bool is_neumann, std::map<EntityHandle, double>& weights);
    void init_tags ();
private:
    double neumann_treatment (EntityHandle node);
    double get_partial_weight (EntityHandle node, EntityHandle volume);
    double get_psi_sum (EntityHandle node, EntityHandle volume, EntityHandle face);
    double get_phi (EntityHandle node, EntityHandle volume, EntityHandle face);
    double get_sigma (EntityHandle node, EntityHandle volume);
    double get_csi (EntityHandle face, EntityHandle volume);
    double get_neta (EntityHandle node, EntityHandle volume, EntityHandle face);
    double get_lambda (EntityHandle node, EntityHandle aux_node, EntityHandle face);
    double get_flux_term (double v1[3], double k[9], double v2[3]);
};

#endif
