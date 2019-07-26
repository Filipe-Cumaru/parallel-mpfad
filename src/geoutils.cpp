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
    void normal_vector (double vert_coords[9], double n[3]) {
        double ab[3], ac[3];

        cblas_dcopy(3, &vert_coords[0], 1, &ab[0], 1);
        cblas_dscal(3, -1.0, &ab[0], 1);
        cblas_daxpy(3, 1, &vert_coords[3], 1, &ab[0], 1);

        cblas_dcopy(3, &vert_coords[0], 1, &ac[0], 1);
        cblas_dscal(3, -1.0, &ac[0], 1);
        cblas_daxpy(3, 1, &vert_coords[6], 1, &ac[0], 1);

        geoutils::cross_product(ab, ac, n);
        cblas_dscal(3, 0.5, &n[0], 1);
    }

    /*
        Calculates the normal vector to a triangular face with vertices's
        coordinates vert_coords corresponding to a reference point and stores
        the result in n.
    */
    void normal_vector (double vert_coords[9], double ref[3], double n[3]) {
        double ab[3], ac[3], ref_vector[3];

        cblas_dcopy(3, &ref[0], 1, &ref_vector[0], 1);
        cblas_dscal(3, -1.0, &ref_vector[0], 1);
        cblas_daxpy(3, 1, &vert_coords[0], 1, &ref_vector[0], 1);

        cblas_dcopy(3, &vert_coords[0], 1, &ab[0], 1);
        cblas_dscal(3, -1.0, &ab[0], 1);
        cblas_daxpy(3, 1, &vert_coords[3], 1, &ab[0], 1);

        cblas_dcopy(3, &vert_coords[0], 1, &ac[0], 1);
        cblas_dscal(3, -1.0, &ac[0], 1);
        cblas_daxpy(3, 1, &vert_coords[6], 1, &ac[0], 1);

        geoutils::cross_product(ab, ac, n);
        cblas_dscal(3, 0.5, &n[0], 1);

        double vector_orientation = cblas_ddot(3, &n[0], 1, &ref_vector[0], 1);
        if (vector_orientation < 0.0) {
            cblas_dscal(3, -1.0, &n[0], 1);
        }
    }

    void normal_vector (double vert_coords[9], double ref[3], bool *spin) {
        double ab[3], ac[3], ref_vector[3], n[3];

        cblas_dcopy(3, &ref[0], 1, &ref_vector[0], 1);
        cblas_dscal(3, -1.0, &ref_vector[0], 1);
        cblas_daxpy(3, 1, &vert_coords[0], 1, &ref_vector[0], 1);

        cblas_dcopy(3, &vert_coords[0], 1, &ab[0], 1);
        cblas_dscal(3, -1.0, &ab[0], 1);
        cblas_daxpy(3, 1, &vert_coords[3], 1, &ab[0], 1);

        cblas_dcopy(3, &vert_coords[0], 1, &ac[0], 1);
        cblas_dscal(3, -1.0, &ac[0], 1);
        cblas_daxpy(3, 1, &vert_coords[6], 1, &ac[0], 1);

        geoutils::cross_product(ab, ac, n);
        cblas_dscal(3, 0.5, &n[0], 1);

        double vector_orientation = cblas_ddot(3, &n[0], 1, &ref_vector[0], 1);
        if (vector_orientation < 0.0) {
            *spin = true;
        }
        else {
            *spin = false;
        }
    }

    void normal_vector (double i[3], double j[3], double k[3], double ref[3], double n[3]) {
        double ab[3], ac[3], ref_vector[3];

        cblas_dcopy(3, &ref[0], 1, &ref_vector[0], 1);
        cblas_dscal(3, -1.0, &ref_vector[0], 1);
        cblas_daxpy(3, 1, &i[0], 1, &ref_vector[0], 1);

        cblas_dcopy(3, &i[0], 1, &ab[0], 1);
        cblas_dscal(3, -1.0, &ab[0], 1);
        cblas_daxpy(3, 1, &j[0], 1, &ab[0], 1);

        cblas_dcopy(3, &i[0], 1, &ac[0], 1);
        cblas_dscal(3, -1.0, &ac[0], 1);
        cblas_daxpy(3, 1, &k[0], 1, &ac[0], 1);

        geoutils::cross_product(ab, ac, n);
        cblas_dscal(3, 0.5, &n[0], 1);

        double vector_orientation = cblas_ddot(3, &n[0], 1, &ref_vector[0], 1);
        if (vector_orientation < 0.0) {
            cblas_dscal(3, -1.0, &n[0], 1);
        }
    }

    /*
        Returns the area of a face with normal area vector n.
    */
    double face_area (double n[3]) {
        return sqrt(cblas_ddot(3, &n[0], 1, &n[0], 1));
    }

    /*
        Calculate the volume of a tetrahedron.
    */
    double tetra_volume (double vert_coords[12]) {
        double v1[3], v2[3], v3[3], temp[3], volume = 0.0;

        cblas_dcopy(3, &vert_coords[0], 1, &v1[0], 1);
        cblas_dscal(3, -1.0, &v1[0], 1);
        cblas_daxpy(3, 1, &vert_coords[3], 1, &v1[0], 1);

        cblas_dcopy(3, &vert_coords[0], 1, &v2[0], 1);
        cblas_dscal(3, -1.0, &v2[0], 1);
        cblas_daxpy(3, 1, &vert_coords[6], 1, &v2[0], 1);

        cblas_dcopy(3, &vert_coords[0], 1, &v3[0], 1);
        cblas_dscal(3, -1.0, &v3[0], 1);
        cblas_daxpy(3, 1, &vert_coords[9], 1, &v3[0], 1);

        geoutils::cross_product(v1, v2, temp);
        volume = fabs(cblas_ddot(3, &temp[0], 1, &v3[0], 1)) / 6.0;

        return volume;
    }
}
