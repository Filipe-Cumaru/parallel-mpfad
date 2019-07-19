#include "geoutils.h"

using namespace std;

namespace geoutils {
    /*
        Performs a cross product between two three dimensional vectors u and v
        and stores the result in n.
    */
    void cross_product(double u[3], double v[3], double n[3]) {
        n[0] = u[1]*v[2] - u[2]*v[1];
        n[1] = u[2]*v[0] - u[0]*v[2];
        n[2] = u[0]*v[1] - u[1]*v[0];
    }

    /*
        Calculates the normal vector to a triangular face with vertices's
        coordinates vert_coords and stores the result in n.
    */
    void get_normal_vector (double vert_coords[9], double n[3]) {
        double ab[3] = { vert_coords[3] - vert_coords[0],
            vert_coords[4] - vert_coords[1], vert_coords[5] - vert_coords[2] };
        double ac[3] = { vert_coords[6] - vert_coords[0],
            vert_coords[7] - vert_coords[1], vert_coords[8] - vert_coords[2] };
        geoutils::cross_product(ab, ac, n);
    }

    /*
        Returns the area of a face with normal area vector n.
    */
    double get_face_area (double n[3]) {
        return sqrt(cblas_ddot(3, &n[0], 1, &n[0], 1));
    }

    /*
        Calculate the volume of a tetrahedron.
    */
    double get_tetra_volume (double vert_coords[12]) {
        double v1[3] = { vert_coords[3] - vert_coords[0],
            vert_coords[4] - vert_coords[1], vert_coords[5] - vert_coords[2] };
        double v2[3] = { vert_coords[6] - vert_coords[0],
            vert_coords[7] - vert_coords[1], vert_coords[8] - vert_coords[2] };
        double v3[3] = { vert_coords[9] - vert_coords[0],
            vert_coords[10] - vert_coords[1], vert_coords[11] - vert_coords[2] };
        double temp[3], volume = 0.0;

        geoutils::cross_product(v1, v2, temp);
        volume = fabs(cblas_ddot(3, &temp[0], 1, &v3[0], 1)) / 6.0;

        return volume;
    }
}
