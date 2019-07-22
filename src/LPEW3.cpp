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
    Range face_nodes, adj_faces, vol_faces, vol_nodes, aux_node, faces;
    double phi = 0.0, lambda_mult = 1.0, sigma = 1.0, neta = 0.0;

    this->mtu->get_bridge_adjacencies(face, 2, 0, face_nodes);
    this->mb->get_adjacencies(&volume, 1, 0, false, vol_nodes);
    aux_node = subtract(vol_nodes, face_nodes);
    this->mtu->get_bridge_adjacencies(node, 0, 2, adj_faces);
    this->mtu->get_bridge_adjacencies(volume, 3, 2, vol_faces);
    faces = intersect(adj_faces, vol_faces);
    faces.erase(face);

    for (Range::iterator it = faces.begin(); it != faces.end(); ++it) {
        lambda_mult *= this->get_lambda(node, aux_node[0], *it);
    }
    neta = this->get_neta(node, volume, face);
    sigma = this->get_sigma(node, volume);

    phi = lambda_mult * neta / sigma;

    return phi;
}

double LPEW3::get_sigma (EntityHandle node, EntityHandle volume) {
    Range vol_faces, adj_faces, in_faces, aux_nodes;
    double *node_coords = (double*) calloc(3, sizeof(double));
    double *aux_nodes_coords = (double*) calloc(6, sizeof(double));
    double sigma = 0.0, vol_centroid[3], clockwise = 1.0, counter_clockwise = 1.0,
        aux_vector[9], count = 0.0, clock = 0.0;
    bool spin = false;
    int index = 0;

    this->mtu->get_bridge_adjacencies(node, 0, 2, adj_faces);
    this->mtu->get_bridge_adjacencies(volume, 3, 2, vol_faces);
    in_faces = intersect(adj_faces, vol_faces);
    this->mb->tag_get_data(this->centroid_tag, &volume, 1, &vol_centroid);

    for (Range::iterator it = in_faces.begin(); it != in_faces.end(); ++it) {
        this->mtu->get_bridge_adjacencies(*it, 2, 0, aux_nodes);
        aux_nodes.erase(node);
        this->mb->get_coords(aux_nodes, aux_nodes_coords);
        std::copy(node_coords, node_coords + 3, aux_vector);
        std::copy(aux_nodes_coords, aux_nodes_coords + 6, aux_vector + 9);
        geoutils::normal_vector(aux_vector, vol_centroid, &spin);
        // This is a workaround because std::swap can't be used with
        // EntityHandle type.
        index = spin ? 1 : 0;
        count = this->get_lambda(node, aux_nodes[index % 2], *it);
        clock = this->get_lambda(node, aux_nodes[(index + 1) % 2], *it);
        counter_clockwise *= count;
        clockwise *= clock;
        aux_nodes.clear();
    }
    sigma = counter_clockwise + clockwise;

    free(node_coords);
    free(aux_nodes_coords);

    return sigma;
}

double LPEW3::get_csi (EntityHandle face, EntityHandle volume) {
    Range face_nodes;
    double *face_nodes_coords = (double*) calloc(9, sizeof(double));
    double k[9], vol_centroid[3], n_i[3], sub_vol[12], csi = 0.0, tetra_vol = 0.0;

    this->mb->tag_get_data(this->permeability_tag, &volume, 1, &k);
    this->mb->tag_get_data(this->centroid_tag, &volume, 1, &vol_centroid);

    this->mtu->get_bridge_adjacencies(face, 2, 3, face_nodes);
    this->mb->get_coords(face_nodes, face_nodes_coords);
    geoutils::normal_vector(face_nodes_coords, vol_centroid, n_i);

    std::copy(face_nodes_coords, face_nodes_coords + 9, sub_vol);
    std::copy(vol_centroid, vol_centroid + 3, sub_vol + 9);
    tetra_vol = geoutils::tetra_volume(sub_vol);

    csi = this->get_flux_term(n_i, k, n_i, 1.0);

    free(face_nodes_coords);

    return csi;
}

double LPEW3::get_neta (EntityHandle node, EntityHandle volume, EntityHandle face) {
    Range vol_nodes, face_nodes, ref_node;
    double *vol_nodes_coords = (double*) calloc(12, sizeof(double));
    double *face_nodes_coords = (double*) calloc(9, sizeof(double));
    double *face_nodes_i_coords = (double*) calloc(9, sizeof(double));
    double *node_coords = (double*) calloc(9, sizeof(double));
    double *ref_node_coords = (double*) calloc(3, sizeof(double));
    double k[9], n_out[3], n_i[3], tetra_vol = 0.0, neta = 0.0;

    this->mb->get_adjacencies(&volume, 1, 0, false, vol_nodes);
    this->mtu->get_bridge_adjacencies(face, 2, 0, face_nodes);
    ref_node = subtract(vol_nodes, face_nodes);

    this->mb->get_coords(vol_nodes, vol_nodes_coords);
    this->mb->get_coords(face_nodes, face_nodes_coords);
    this->mb->get_coords(ref_node, ref_node_coords);
    this->mb->get_coords(&node, 1, node_coords);

    vol_nodes.erase(node);
    this->mb->get_coords(vol_nodes, face_nodes_i_coords);

    geoutils::normal_vector(face_nodes_i_coords, node_coords, n_out);
    geoutils::normal_vector(face_nodes_coords, ref_node_coords, n_i);
    tetra_vol = geoutils::tetra_volume(vol_nodes_coords);

    this->mb->tag_get_data(this->permeability_tag, &volume, 1, &k);
    neta = this->get_flux_term(n_out, k, n_i, 1.0) / tetra_vol;

    free(vol_nodes_coords);
    free(face_nodes_coords);
    free(face_nodes_i_coords);
    free(node_coords);
    free(ref_node_coords);

    return neta;
}

double LPEW3::get_lambda (EntityHandle node, EntityHandle aux_node, EntityHandle face) {
    Range adj_vols, face_nodes, ref_node, vol_nodes, ref_node_i;
    double *face_nodes_coords = (double*) calloc(9, sizeof(double));
    double *ref_node_coords = (double*) calloc(3, sizeof(double));
    double *aux_node_coords = (double*) calloc(3, sizeof(double));
    double *node_coords = (double*) calloc(3, sizeof(double));
    double *ref_node_i_coords = (double*) calloc(3, sizeof(double));
    double lambda_sum = 0.0, tetra_vol = 0.0, k[9], vol_centroid[3], sub_vol[12],
        n_int[3], n_i[3];

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

        std::copy(face_nodes_coords, face_nodes_coords + 9, sub_vol);
        std::copy(vol_centroid, vol_centroid + 3, sub_vol + 9);
        tetra_vol = geoutils::tetra_volume(sub_vol);

        ref_node_i = subtract(vol_nodes, face_nodes);
        this->mb->get_coords(ref_node_i, ref_node_i_coords);

        geoutils::normal_vector(node_coords, aux_node_coords, vol_centroid, ref_node_coords, n_int);
        geoutils::normal_vector(face_nodes_coords, ref_node_i_coords, n_i);

        lambda_sum += this->get_flux_term(n_i, k, n_int, 1.0) / tetra_vol;

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
