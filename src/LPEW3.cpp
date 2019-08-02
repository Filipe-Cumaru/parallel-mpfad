#include "LPEW3.h"

using namespace std;
using namespace moab;

// NOTE: When implementing super class, remember to add a constructor that
// accepts a value for tau.

std::map<std::tuple<EntityHandle, EntityHandle, EntityHandle>, double> neta_mem;
std::map<std::tuple<EntityHandle, EntityHandle, EntityHandle>, double> lambda_mem;
std::map<std::tuple<EntityHandle, EntityHandle, EntityHandle>, double> phi_mem;
std::map<std::tuple<EntityHandle, EntityHandle, EntityHandle>, double> psi_sum_mem;
std::map<std::tuple<EntityHandle, EntityHandle>, double> sigma_mem;
std::map<std::tuple<EntityHandle, EntityHandle>, double> csi_mem;

LPEW3::LPEW3 () : mb (new Core()),
                mtu (new MeshTopoUtil(mb)),
                tau (0.0) {}

LPEW3::LPEW3 (Interface *moab_interface) : mb (moab_interface),
                                        mtu (new MeshTopoUtil(mb)),
                                        tau (0.0) {}

void LPEW3::interpolate (EntityHandle node, bool is_neumann, std::map<EntityHandle, double>& weights) {
    Range vols_around;
    double p_weight, p_weight_sum = 0.0, neu_term;

    this->mtu->get_bridge_adjacencies(node, 0, 3, vols_around);
    for (Range::iterator it = vols_around.begin(); it != vols_around.end(); ++it) {
        p_weight = this->get_partial_weight(node, *it);
        p_weight_sum += p_weight;
        weights[*it] = p_weight;
    }

    for (Range::iterator it = vols_around.begin(); it != vols_around.end(); ++it) {
        weights[*it] /= p_weight_sum;
    }

    if (is_neumann) {
        neu_term = this->neumann_treatment(node) / p_weight_sum;
        weights[node] = neu_term;
    }
}

double LPEW3::neumann_treatment (EntityHandle node) {
    Range adj_faces, neumann_faces, face_nodes, neu_vol;
    double face_nodes_coords[9];
    double neu_term_sum = 0.0, neu_term = 0.0, face_flux = 0.0, n[3], neu_psi,
        neu_phi, face_area;

    this->mtu->get_bridge_adjacencies(node, 0, 2, adj_faces);
    this->mb->get_entities_by_type_and_tag(0, MBTRI, &this->neumann_tag, NULL,
        1, neumann_faces);
    adj_faces = intersect(adj_faces, neumann_faces);

    for (Range::iterator it  = adj_faces.begin(); it != adj_faces.end(); ++it) {
        this->mb->tag_get_data(this->neumann_tag, &(*it), 1, &face_flux);
        this->mtu->get_bridge_adjacencies(*it, 2, 0, face_nodes);
        this->mb->get_coords(face_nodes, &face_nodes_coords[0]);
        geoutils::normal_vector(face_nodes_coords, n);
        face_area = geoutils::face_area(n);
        this->mtu->get_bridge_adjacencies(*it, 2, 3, neu_vol);
        neu_psi = this->get_psi_sum(node, neu_vol[0], *it);
        neu_phi = this->get_phi(node, neu_vol[0], *it);
        neu_term = -3.0 * (1 + neu_psi - neu_phi) * face_flux * face_area;
        neu_term_sum += neu_term;
        face_nodes.clear();
        neu_vol.clear();
    }

    return neu_term_sum;
}

double LPEW3::get_partial_weight (EntityHandle node, EntityHandle volume) {
    Range vol_faces, adj_faces, vol_node_faces, face_neigh;
    double partial_weight = 0.0, zepta = 0.0, delta = 0.0, csi = 0.0,
        psi_sum_neigh = 0.0, psi_sum_vol = 0.0, phi_neigh = 0.0, phi_vol = 0.0;

    this->mtu->get_bridge_adjacencies(volume, 3, 2, vol_faces);
    this->mtu->get_bridge_adjacencies(node, 0, 2, adj_faces);
    vol_node_faces = intersect(vol_faces, adj_faces);

    for (Range::iterator it = vol_node_faces.begin(); it != vol_node_faces.end(); ++it) {
        this->mtu->get_bridge_adjacencies(*it, 2, 3, face_neigh);
        face_neigh.erase(volume);
        csi = this->get_csi(*it, volume);
        // In case it is a boundary volume.
        if (!face_neigh.empty()) {
            psi_sum_neigh = this->get_psi_sum(node, face_neigh[0], *it);
            phi_neigh = this->get_phi(node, face_neigh[0], *it);
        }
        else {
            psi_sum_neigh = 0.0;
            phi_neigh = 0.0;
        }
        psi_sum_vol = this->get_psi_sum(node, volume, *it);
        phi_vol = this->get_phi(node, volume, *it);
        zepta += (psi_sum_vol + psi_sum_neigh) * csi;
        delta += (phi_vol + phi_neigh) * csi;
        face_neigh.clear();
    }

    partial_weight = zepta - delta;

    return partial_weight;
}

double LPEW3::get_psi_sum (EntityHandle node, EntityHandle volume, EntityHandle face) {
    Range face_nodes, vol_nodes, aux_node, adj_faces, vol_faces, faces,
        a_face_nodes, other_node;
    double psi_sum = 0.0, lambda1 = 0.0, lambda2 = 0.0, neta = 0.0, sigma = 0.0;

    std::tuple<EntityHandle, EntityHandle, EntityHandle> key (node, volume, face);
    if (psi_sum_mem.find(key) != psi_sum_mem.end()) {
        return psi_sum_mem[key];
    }

    this->mtu->get_bridge_adjacencies(face, 2, 0, face_nodes);
    this->mtu->get_bridge_adjacencies(volume, 3, 0, vol_nodes);
    aux_node = subtract(vol_nodes, face_nodes);

    this->mtu->get_bridge_adjacencies(node, 0, 2, adj_faces);
    this->mtu->get_bridge_adjacencies(volume, 3, 2, vol_faces);
    faces = intersect(adj_faces, vol_faces);
    faces.erase(face);

    int num_faces = faces.size(), j = num_faces - 1;
    for (int i = 0; i < num_faces; i++) {
        this->mtu->get_bridge_adjacencies(faces[i], 2, 0, a_face_nodes);
        other_node = subtract(face_nodes, a_face_nodes);
        lambda1 = this->get_lambda(node, aux_node[0], faces[i]);
        lambda2 = this->get_lambda(node, other_node[0], faces[j]);
        neta = this->get_neta(node, volume, faces[i]);
        psi_sum += lambda1*lambda2*neta;
        j = (j + 1) % num_faces;
        a_face_nodes.clear();
    }
    sigma = this->get_sigma(node, volume);
    psi_sum /= sigma;

    psi_sum_mem[key] = psi_sum;

    return psi_sum;
}

double LPEW3::get_phi (EntityHandle node, EntityHandle volume, EntityHandle face) {
    Range face_nodes, adj_faces, vol_faces, vol_nodes, aux_node, faces;
    double phi = 0.0, lambda_mult = 1.0, sigma = 1.0, neta = 0.0;

    std::tuple<EntityHandle, EntityHandle, EntityHandle> key (node, volume, face);
    if (phi_mem.find(key) != phi_mem.end()) {
        return phi_mem[key];
    }

    this->mtu->get_bridge_adjacencies(face, 2, 0, face_nodes);
    this->mtu->get_bridge_adjacencies(volume, 3, 0, vol_nodes);
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

    phi_mem[key] = phi;

    return phi;
}

double LPEW3::get_sigma (EntityHandle node, EntityHandle volume) {
    Range vol_faces, adj_faces, in_faces, aux_nodes;
    double node_coords[3], aux_nodes_coords[6];
    double sigma = 0.0, vol_centroid[3], clockwise = 1.0, counter_clockwise = 1.0,
        aux_vector[9], count = 0.0, clock = 0.0;
    bool spin = false;
    int index = 0;

    std::tuple<EntityHandle, EntityHandle> key (node, volume);
    if (sigma_mem.find(key) != sigma_mem.end()) {
        return sigma_mem[key];
    }

    this->mtu->get_bridge_adjacencies(node, 0, 2, adj_faces);
    this->mtu->get_bridge_adjacencies(volume, 3, 2, vol_faces);
    in_faces = intersect(adj_faces, vol_faces);
    this->mb->tag_get_data(this->centroid_tag, &volume, 1, &vol_centroid);
    this->mb->get_coords(&node, 1, &node_coords[0]);

    for (Range::iterator it = in_faces.begin(); it != in_faces.end(); ++it) {
        this->mtu->get_bridge_adjacencies(*it, 2, 0, aux_nodes);
        aux_nodes.erase(node);
        this->mb->get_coords(aux_nodes, &aux_nodes_coords[0]);
        std::copy(node_coords, node_coords + 3, aux_vector);
        std::copy(aux_nodes_coords, aux_nodes_coords + 6, aux_vector + 3);
        geoutils::normal_vector(aux_vector, vol_centroid, &spin);
        // This is a workaround because std::swap can't be used with
        // EntityHandle type.
        index = spin ? 1 : 0;
        count = this->get_lambda(node, aux_nodes[index], *it);
        clock = this->get_lambda(node, aux_nodes[(index + 1) % 2], *it);
        counter_clockwise *= count;
        clockwise *= clock;
        aux_nodes.clear();
    }
    sigma = counter_clockwise + clockwise;

    sigma_mem[key] = sigma;

    return sigma;
}

double LPEW3::get_csi (EntityHandle face, EntityHandle volume) {
    Range face_nodes;
    double face_nodes_coords[9];
    double k[9], vol_centroid[3], n_i[3], sub_vol[12], csi = 0.0, tetra_vol = 0.0;

    std::tuple<EntityHandle, EntityHandle> key (face, volume);
    if (csi_mem.find(key) != csi_mem.end()) {
        return csi_mem[key];
    }

    this->mb->tag_get_data(this->permeability_tag, &volume, 1, &k);
    this->mb->tag_get_data(this->centroid_tag, &volume, 1, &vol_centroid);

    this->mtu->get_bridge_adjacencies(face, 2, 0, face_nodes);
    this->mb->get_coords(face_nodes, face_nodes_coords);
    geoutils::normal_vector(face_nodes_coords, vol_centroid, n_i);

    std::copy(face_nodes_coords, face_nodes_coords + 9, sub_vol);
    std::copy(vol_centroid, vol_centroid + 3, sub_vol + 9);
    tetra_vol = geoutils::tetra_volume(sub_vol);

    csi = this->get_flux_term(n_i, k, n_i) / tetra_vol;

    csi_mem[key] = csi;

    return csi;
}

double LPEW3::get_neta (EntityHandle node, EntityHandle volume, EntityHandle face) {
    Range vol_nodes, face_nodes, ref_node;
    double vol_nodes_coords[12];
    double face_nodes_coords[9];
    double face_nodes_i_coords[9];
    double node_coords[9];
    double ref_node_coords[3];
    double k[9], n_out[3], n_i[3], tetra_vol = 0.0, neta = 0.0;

    std::tuple<EntityHandle, EntityHandle, EntityHandle> key (node, volume, face);
    if (neta_mem.find(key) != neta_mem.end()) {
        return neta_mem[key];
    }

    this->mtu->get_bridge_adjacencies(volume, 3, 0, vol_nodes);
    this->mtu->get_bridge_adjacencies(face, 2, 0, face_nodes);
    ref_node = subtract(vol_nodes, face_nodes);

    this->mb->get_coords(vol_nodes, &vol_nodes_coords[0]);
    this->mb->get_coords(face_nodes, &face_nodes_coords[0]);
    this->mb->get_coords(ref_node, &ref_node_coords[0]);
    this->mb->get_coords(&node, 1, &node_coords[0]);

    vol_nodes.erase(node);
    this->mb->get_coords(vol_nodes, &face_nodes_i_coords[0]);

    geoutils::normal_vector(face_nodes_i_coords, node_coords, n_out);
    geoutils::normal_vector(face_nodes_coords, ref_node_coords, n_i);
    tetra_vol = geoutils::tetra_volume(vol_nodes_coords);

    this->mb->tag_get_data(this->permeability_tag, &volume, 1, &k);
    neta = this->get_flux_term(n_out, k, n_i) / tetra_vol;
    neta_mem[key] = neta;

    return neta;
}

double LPEW3::get_lambda (EntityHandle node, EntityHandle aux_node, EntityHandle face) {
    Range adj_vols, face_nodes, ref_node, vol_nodes, ref_node_i;
    double face_nodes_coords[9], ref_node_coords[3], aux_node_coords[3],
        node_coords[3], ref_node_i_coords[3];
    double lambda_sum = 0.0, tetra_vol = 0.0, k[9], vol_centroid[3], sub_vol[12],
        n_int[3], n_i[3];

    std::tuple<EntityHandle, EntityHandle, EntityHandle> key (node, aux_node, face);
    if (lambda_mem.find(key) != lambda_mem.end()) {
        return lambda_mem[key];
    }

    this->mtu->get_bridge_adjacencies(face, 2, 3, adj_vols);
    this->mtu->get_bridge_adjacencies(face, 2, 0, face_nodes);

    // ref_node = face_nodes - (node U aux_node)
    ref_node = face_nodes;
    ref_node.erase(node);
    ref_node.erase(aux_node);

    this->mb->get_coords(face_nodes, &face_nodes_coords[0]);
    this->mb->get_coords(ref_node, &ref_node_coords[0]);
    this->mb->get_coords(&aux_node, 1, &aux_node_coords[0]);
    this->mb->get_coords(&node, 1, &node_coords[0]);

    for (Range::iterator it = adj_vols.begin(); it != adj_vols.end(); ++it) {
        this->mb->tag_get_data(this->permeability_tag, &(*it), 1, &k);
        this->mb->tag_get_data(this->centroid_tag, &(*it), 1, &vol_centroid);
        this->mtu->get_bridge_adjacencies(*it, 3, 0, vol_nodes);

        std::copy(face_nodes_coords, face_nodes_coords + 9, sub_vol);
        std::copy(vol_centroid, vol_centroid + 3, sub_vol + 9);
        tetra_vol = geoutils::tetra_volume(sub_vol);

        ref_node_i = subtract(vol_nodes, face_nodes);
        this->mb->get_coords(ref_node_i, &ref_node_i_coords[0]);

        geoutils::normal_vector(node_coords, aux_node_coords, vol_centroid, ref_node_coords, n_int);
        geoutils::normal_vector(face_nodes_coords, ref_node_i_coords, n_i);

        lambda_sum += this->get_flux_term(n_i, k, n_int) / tetra_vol;

        vol_nodes.clear();
        ref_node_i.clear();
    }
    lambda_mem[key] = lambda_sum;

    return lambda_sum;
}

double LPEW3::get_flux_term (double v1[3], double k[9], double v2[3]) {
    double flux_term = 0.0, temp[3] = {0.0, 0.0, 0.0};
    // <<v1, K>, v2> = trans(trans(v1)*K)*v2
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 1, 3, 3, 1.0, &v1[0], 1, &k[0], 3, 1.0, &temp[0], 3);
    flux_term = cblas_ddot(3, &temp[0], 1, &v2[0], 1);
    return flux_term;
}

void LPEW3::init_tags () {
    this->mb->tag_get_handle("PERMEABILITY", this->permeability_tag);
    this->mb->tag_get_handle("CENTROID", this->centroid_tag);
    this->mb->tag_get_handle("DIRICHLET", this->dirichlet_tag);
    this->mb->tag_get_handle("NEUMANN", this->neumann_tag);
}
