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
    void normal_vector (double vert_coords[9], double n[3]);
    void normal_vector (double vert_coords[9], double ref[3], double n[3]);
    void normal_vector (double i[3], double j[3], double k[3], double ref[3], double n[3]);
    void normal_vector (double vert_coords[9], double ref[3], bool *spin);
    double face_area (double n[3]);
    double tetra_volume (double vert_coords[12]);
}

#endif
