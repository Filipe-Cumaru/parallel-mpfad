#ifndef GEOUTILS_H
#define GEOUTILS_H

using namespace std;

/* C++ STL includes */
#include <cstdlib>	/* calloc, free */
#include <cmath>	/* sqrt, pow */
#include <utility>  /* std::swap */

/* BLAS header */
#include "cblas.h"

namespace geoutils {
    void cross_product(double u[3], double v[3], double n[3]);
    void get_normal_vector (double vert_coords[9], double n[3]);
    double get_face_area (double n[3]);
    double get_tetra_volume (double vert_coords[12]);
}

#endif
