#include "LPEW3.h"

using namespace std;
using namespace moab;

LPEW3::LPEW3 () : mb (new Core()),
                topo_util (new MeshTopoUtil(mb)),
                tau (1.0) {}

LPEW3::LPEW3 (Interface *moab_interface) : mb (moab_interface),
                                        topo_util (new MeshTopoUtil(mb)),
                                        tau (1.0) {}

// REVIEW: The LPEW3 is defined for tau = 1. This constructor should
// be implemented in a super class.
LPEW3::LPEW3 (Interface *moab_interface, double tau_value) : mb (moab_interface),
                                                            topo_util (new MeshTopoUtil(mb)),
                                                            tau (tau_value) {}

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
    return 0.0;
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
