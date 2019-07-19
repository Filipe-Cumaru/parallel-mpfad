#include "LPEW3.h"

using namespace std;
using namespace moab;

// NOTE: When implementing super class, remember to add a constructor that
// accepts a value for tau.

LPEW3::LPEW3 () : mb (new Core()),
                mtu (new MeshTopoUtil(mb)),
                tau (0.0) {}

LPEW3::LPEW3 (Interface *moab_interface) : mb (moab_interface),
                                        mtu (new MeshTopoUtil(mb)),
                                        tau (0.0) {}

void LPEW3::interpolate (EntityHandle node, bool is_neumann, std::map<EntityHandle, double>& weights) {
    return;
}

double LPEW3::neumann_treatment (EntityHandle node) {
    return 0.0;
}

double LPEW3::get_partial_weight (EntityHandle node, EntityHandle volume) {
    return 0.0;
}

double LPEW3::get_psi_sum (EntityHandle node, EntityHandle volume, EntityHandle face) {
    return 0.0;
}

double LPEW3::get_phi (EntityHandle node, EntityHandle volume, EntityHandle face) {
    return 0.0;
}

double LPEW3::get_sigma (EntityHandle node, EntityHandle volume) {
    return 0.0;
}

double LPEW3::get_csi (EntityHandle face, EntityHandle volume) {
    return 0.0;
}

double LPEW3::get_neta (EntityHandle node, EntityHandle volume, EntityHandle face) {
    return 0.0;
}

double LPEW3::get_lambda (EntityHandle node, EntityHandle aux_node, EntityHandle face) {
    Range adj_vols, face_nodes, ref_node, vol_nodes, ref_node_i;
    double *face_nodes_coords = (double*) calloc(9, sizeof(double));
    double *ref_node_coords = (double*) calloc(3, sizeof(double));
    double *aux_node_coords = (double*) calloc(3, sizeof(double));
    double *node_coords = (double*) calloc(3, sizeof(double));
    double *ref_node_i_coords = (double*) calloc(3, sizeof(double));
    double lambda_sum = 0.0, k[9], vol_centroid[3];

    this->mtu->get_bridge_adjacencies(face, 2, 3, adj_vols);
    this->mtu->get_bridge_adjacencies(face, 2, 0, face_nodes);

    // face_nodes = face_nodes - (node U aux_node)
    face_nodes.erase(node);
    face_nodes.erase(aux_node);

    this->mb->get_coords(face_nodes, face_nodes_coords);
    this->mb->get_coords(ref_node, ref_node_coords);
    this->mb->get_coords(&aux_node, 1, aux_node_coords);
    this->mb->get_coords(&node, 1, node_coords);

    for (Range::iterator it = adj_vols.begin(); it != adj_vols.end(); ++it) {
        this->mb->tag_get_data(this->permeability_tag, &(*it), 1, &k);
        this->mb->tag_get_data(this->centroid_tag, &(*it), 1, &vol_centroid);
        this->mb->get_adjacencies(&(*it), 1, 0, false, vol_nodes);
        // Calculate volume of the tetrahedron with vertices face_nodes and vol_centroid.
        ref_node_i = subtract(vol_nodes, face_nodes);
        this->mb->get_coords(ref_node_i, ref_node_i_coords);
        // Add routines to cross product, area and normal vector.
        vol_nodes.clear();
        ref_node_i.clear();
    }

    free(face_nodes_coords);
    free(ref_node_coords);
    free(aux_node_coords);
    free(node_coords);
    free(ref_node_i_coords);

    return lambda_sum;
}

double LPEW3::get_flux_term (double v1[3], double k[9], double v2[3], double face_area) {
    double flux_term = 0.0, temp[3];
    // <<v1, K>, v2> = trans(trans(v1)*K)*v2
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 1, 3, 3, 1.0, &v1[0], 1, &k[0], 3, 1.0, &temp[0], 3);
    flux_term = cblas_ddot(3, &temp[0], 1, &v2[0], 1) / face_area;
    return flux_term;
}

void LPEW3::init_tags () {
    this->mb->tag_get_handle("PERMEABILITY", this->permeability_tag);
    this->mb->tag_get_handle("CENTROID", this->centroid_tag);
    this->mb->tag_get_handle("DIRICHLET", this->dirichlet_tag);
    this->mb->tag_get_handle("NEUMANN", this->neumann_tag);
}
