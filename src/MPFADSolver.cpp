#include "MPFADSolver.h"

using namespace std;
using namespace moab;

MPFADSolver::MPFADSolver () : mb(new Core()),
                            pcomm(new ParallelComm(mb, MPI_COMM_WORLD)),
                            mtu(new MeshTopoUtil(mb)) {}

MPFADSolver::MPFADSolver (Interface* moab_interface) : mb(moab_interface),
                                                pcomm(new ParallelComm(mb, MPI_COMM_WORLD)),
                                                mtu(new MeshTopoUtil(mb)) {}

void MPFADSolver::run () {
    /*
		Run solver for TPFA problem specificed at given moab::Core
		instance.

		Parameters
		----------
		None
	*/

    ErrorCode rval;
    clock_t ts;
    int rank = this->pcomm->proc_config().proc_rank();

    // Get all volumes in the mesh and exchange those shared with
    // others processors.
    Range volumes, faces, nodes;
    rval = this->mb->get_entities_by_dimension(0, 3, volumes, false);
    if (rval != MB_SUCCESS) {
        throw runtime_error("Could not retrieve volumes from the mesh.\n");
    }
    rval = this->mb->get_entities_by_dimension(0, 2, faces, false);
    if (rval != MB_SUCCESS) {
        throw runtime_error("Could not retrieve volumes from the mesh.\n");
    }
    rval = this->mb->get_entities_by_dimension(0, 0, nodes, false);
    if (rval != MB_SUCCESS) {
        throw runtime_error("Could not retrieve volumes from the mesh.\n");
    }

    // Exchange volumes sharing any vertex in an interface.
    rval = this->pcomm->exchange_ghost_cells(3, 0, 1, 0, true);
    if (rval != MB_SUCCESS) {
        throw runtime_error("exchange_ghost_cells failed\n");
    }
    rval = this->pcomm->exchange_ghost_cells(2, 0, 2, 0, true);
    if (rval != MB_SUCCESS) {
        throw runtime_error("exchange_ghost_cells failed\n");
    }
    rval = this->pcomm->exchange_ghost_cells(0, 0, 2, 0, true);
    if (rval != MB_SUCCESS) {
        throw runtime_error("exchange_ghost_cells failed\n");
    }

    // Calculate the total numbers of elements in the mesh.
    int num_local_elems = volumes.size(), num_global_elems = 0;
    MPI_Allreduce(&num_local_elems, &num_global_elems, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    printf("<%d> # global elems: %d\tlocal elems = %d\n", rank, num_global_elems, num_local_elems);

    this->init_tags();

    int* gids = (int*) calloc(volumes.size(), sizeof(int));
    if (gids == NULL) {
        printf("<%d> Error: Null pointer\n", rank);
        exit(EXIT_FAILURE);
    }
    rval = this->mb->tag_get_data(this->tags[global_id], volumes, (void*) gids);
    if (rval != MB_SUCCESS) {
        throw runtime_error("tag_get_data for gids failed\n");
    }

    // Setting up Epetra structures
    Epetra_MpiComm epetra_comm (MPI_COMM_WORLD);
    Epetra_Map row_map (num_global_elems, num_local_elems, gids, 0, epetra_comm);
    Epetra_CrsMatrix A (Copy, row_map, 0);
    Epetra_Vector b (row_map);
    Epetra_Vector X (row_map);

    printf("<%d> Matrix assembly...\n", rank);
    ts = clock();
    this->assemble_matrix(A, b, volumes, faces, nodes);
    ts = clock() - ts;
    printf("<%d> Done. Time elapsed: %f\n", rank, ((double) ts)/CLOCKS_PER_SEC);
    A.FillComplete();

    Epetra_LinearProblem linear_problem (&A, &X, &b);
    AztecOO solver (linear_problem);

    // Setting up solver preconditioning
    // Teuchos::ParameterList MLList;
    // ML_Epetra::MultiLevelPreconditioner * MLPrec = new ML_Epetra::MultiLevelPreconditioner(A, true);
    // MLList.set("max levels", 4);
    // MLList.set("repartition: enable", 1);
    // MLList.set("repartition: partitioner", "ParMetis");
    // MLList.set("coarse: type", "Chebyshev");
    // MLList.set("coarse: sweeps", 2);
    // MLList.set("smoother: type", "Chebyshev");
    // MLList.set("aggregation: type", "METIS");
    // ML_Epetra::SetDefaults("SA", MLList);
    // solver.SetPrecOperator(MLPrec);
    solver.SetAztecOption(AZ_kspace, 250);
    // solver.SetAztecOption(AZ_precond, AZ_none);
    solver.SetAztecOption(AZ_solver, AZ_gmres_condnum);
    solver.Iterate(1000, 1e-14);
    // delete MLPrec;

    printf("<%d> Setting pressure...\n", rank);
    ts = clock();
    this->set_pressure_tags(X, volumes);
    ts = clock() - ts;
    printf("<%d> Done. Time elapsed: %f\n", rank, ((double) ts)/CLOCKS_PER_SEC);

    free(gids);
}

void MPFADSolver::load_file (string fname) {
    string read_opts = "PARALLEL=READ_PART;PARTITION=PARALLEL_PARTITION;PARALLEL_RESOLVE_SHARED_ENTS";
    ErrorCode rval;
    rval = this->mb->load_file(fname.c_str(), 0, read_opts.c_str());
    if (rval != MB_SUCCESS) {
        throw runtime_error("load_file failed\n");
    }
}

void MPFADSolver::write_file (string fname) {
    string write_opts = "PARALLEL=WRITE_PART";
    EntityHandle volumes_meshset;
    Range volumes;
    ErrorCode rval;
    rval = this->mb->create_meshset(0, volumes_meshset);
    if (rval != MB_SUCCESS) {
        throw runtime_error("write_file failed while creating meshset\n");
    }
    rval = this->mb->get_entities_by_dimension(0, 3, volumes, false);
    if (rval != MB_SUCCESS) {
        throw runtime_error("write_file failed while retrieving volumes\n");
    }
    rval = this->mb->add_entities(volumes_meshset, volumes);
    if (rval != MB_SUCCESS) {
        throw runtime_error("write_file failed while adding volumes to meshset\n");
    }
    rval = this->mb->write_file(fname.c_str(), 0, write_opts.c_str(), &volumes_meshset, 1);
    if (rval != MB_SUCCESS) {
        throw runtime_error("write_file failed\n");
    }
}

void MPFADSolver::init_tags () {
    ErrorCode rval;
    Range empty_set;

    rval = this->mb->tag_get_handle("GLOBAL_ID", this->tags[global_id]);
    rval = this->mb->tag_get_handle("PERMEABILITY", this->tags[permeability]);
    rval = this->mb->tag_get_handle("CENTROID", this->tags[centroid]);
    rval = this->mb->tag_get_handle("DIRICHLET", this->tags[dirichlet]);
    rval = this->mb->tag_get_handle("NEUMANN", this->tags[neumann]);
    rval = this->mb->tag_get_handle("SOURCE", this->tags[source]);

    std::vector<Tag> tag_vector;
    for (int i = 0; i < 6; ++i) {
        tag_vector.push_back(this->tags[i]);
    }
    rval = this->pcomm->exchange_tags(tag_vector, tag_vector, empty_set);
    if (rval != MB_SUCCESS) {
        throw runtime_error("exchange_tags failed");
    }
}

void MPFADSolver::assemble_matrix (Epetra_CrsMatrix& A, Epetra_Vector& b, Range volumes, Range faces, Range nodes) {
    ErrorCode rval;

    // Retrieving Dirichlet faces and nodes.
    Range dirichlet_faces;
    rval = this->mb->get_entities_by_type_and_tag(0, MBTRI,
                    &this->tags[dirichlet], NULL, 1, dirichlet_faces);
    if (rval != MB_SUCCESS) {
        throw runtime_error("Unable to get dirichlet entities");
    }
    rval = this->mb->get_entities_by_type_and_tag(0, MBVERTEX,
                    &this->tags[dirichlet], NULL, 1, this->dirichlet_nodes);
    if (rval != MB_SUCCESS) {
        throw runtime_error("Unable to get dirichlet entities");
    }
    dirichlet_faces = intersect(dirichlet_faces, faces);
    this->dirichlet_nodes = intersect(this->dirichlet_nodes, nodes);

    // Retrieving Neumann faces and nodes. Notice that faces/nodes
    // that are also Dirichlet faces/nodes are filtered.
    Range neumann_faces;
    rval = this->mb->get_entities_by_type_and_tag(0, MBTRI,
                    &this->tags[neumann], NULL, 1, neumann_faces);
    if (rval != MB_SUCCESS) {
        throw runtime_error("Unable to get neumann entities");
    }
    rval = this->mb->get_entities_by_type_and_tag(0, MBVERTEX,
                    &this->tags[neumann], NULL, 1, this->neumann_nodes);
    if (rval != MB_SUCCESS) {
        throw runtime_error("Unable to get neumann entities");
    }
    neumann_faces = subtract(neumann_faces, dirichlet_faces);
    this->neumann_nodes = subtract(this->neumann_nodes, this->dirichlet_nodes);
    neumann_faces = intersect(neumann_faces, faces);
    this->neumann_nodes = intersect(this->neumann_nodes, nodes);

    // Get internal faces and nodes.
    Range internal_faces;
    rval = this->mb->get_entities_by_dimension(0, 2, internal_faces);
    if (rval != MB_SUCCESS) {
        throw runtime_error("Unable to get internal faces");
    }
    internal_faces = subtract(internal_faces, neumann_faces);
    internal_faces = subtract(internal_faces, dirichlet_faces);
    rval = this->mb->get_entities_by_dimension(0, 0, this->internal_nodes);
    if (rval != MB_SUCCESS) {
        throw runtime_error("Unable to get internal nodes");
    }
    this->internal_nodes = subtract(this->internal_nodes, this->neumann_nodes);
    this->internal_nodes = subtract(this->internal_nodes, this->dirichlet_nodes);
    internal_faces = intersect(internal_faces, faces);
    this->internal_nodes = intersect(this->internal_nodes, nodes);

    LPEW3 interpolation_method (this->mb);
    cout << "Initializing tags for LPEW3" << endl;
    interpolation_method.init_tags();

    cout << "Interpolating internal nodes" << endl;
    for (Range::iterator it = this->internal_nodes.begin(); it != this->internal_nodes.end(); ++it) {
        interpolation_method.interpolate(*it, false, this->weights[*it]);
    }
    cout << "Interpolating neumann nodes" << endl;
    for (Range::iterator it = this->neumann_nodes.begin(); it != this->neumann_nodes.end(); ++it) {
        interpolation_method.interpolate(*it, true, this->weights[*it]);
    }

    // Check source terms and assign their values straight to the
    // right hand vector.
    Range source_volumes;
    double source_term = 0.0;
    int volume_id = -1;
    rval = this->mb->get_entities_by_type_and_tag(0, MBTET,
                    &this->tags[source], NULL, 1, source_volumes);
    if (rval != MB_SUCCESS) {
        throw runtime_error("Unable to get source terms");
    }
    for (Range::iterator it = source_volumes.begin(); it != source_volumes.end(); ++it) {
        this->mb->tag_get_data(this->tags[source], &(*it), 1, &source_term);
        this->mb->tag_get_data(this->tags[global_id], &(*it), 1, &volume_id);
        b.SumIntoGlobalValues(1, &source_term, &volume_id);
    }

    this->visit_neumann_faces(A, b, neumann_faces);
    this->visit_dirichlet_faces(A, b, dirichlet_faces);
    this->visit_internal_faces(A, b, internal_faces);
}

void MPFADSolver::set_pressure_tags (Epetra_Vector& X, Range& volumes) {
    Tag pressure_tag;
    ErrorCode rval;
    rval = this->mb->tag_get_handle("PRESSURE", 1, MB_TYPE_DOUBLE, pressure_tag, MB_TAG_DENSE | MB_TAG_CREAT);
    if (rval != MB_SUCCESS) {
        throw runtime_error("Unable to create pressure tag");
    }
    rval = this->mb->tag_set_data(pressure_tag, volumes, &X[0]);
    if (rval != MB_SUCCESS) {
        throw runtime_error("Unable to set pressure data");
    }
}

double MPFADSolver::get_cross_diffusion_term (double tan[3], double vec[3], double s,
                                              double h1, double Kn1, double Kt1,
                                              double h2, double Kn2, double Kt2,
                                              bool boundary) {
    double mesh_anisotropy_term, physical_anisotropy_term, cross_diffusion_term;
    double dot_term, cdf_term;
    if (!boundary) {
        mesh_anisotropy_term = cblas_ddot(3, &tan[0], 1, &vec[0], 1) / pow(s, 2);
        physical_anisotropy_term = -(h1*(Kt1 / Kn1) + h2*(Kt2 / Kn2))/s;
        cross_diffusion_term = mesh_anisotropy_term + physical_anisotropy_term;
    }
    else {
        dot_term = -Kn1*cblas_ddot(3, &tan[0], 1, &vec[0], 1);
        cdf_term = h1*s*Kt1;
        cross_diffusion_term = (dot_term + cdf_term) / (2 * h1 * s);
    }

    return cross_diffusion_term;
}

void MPFADSolver::node_treatment (EntityHandle node, int id_left, int id_right,
                                    double k_eq, double d_JI, double d_JK,
                                    Epetra_CrsMatrix& A, Epetra_Vector& b) {
    double rhs = 0.5 * k_eq * (d_JK + d_JI);
    double col_value;
    int vol_id = -1;

    if (std::find(this->dirichlet_nodes.begin(), this->dirichlet_nodes.end(), node) != this->dirichlet_nodes.end()) {
        double pressure = 0.0, dirichlet_term;
        this->mb->tag_get_data(this->tags[dirichlet], &node, 1, &pressure);
        dirichlet_term = rhs*pressure;
        b.SumIntoGlobalValues(1, &dirichlet_term, &id_right);
        dirichlet_term = -rhs*pressure;
        b.SumIntoGlobalValues(1, &dirichlet_term, &id_left);
    }
    else if (std::find(this->neumann_nodes.begin(), this->neumann_nodes.end(), node) != this->neumann_nodes.end()) {
        double neu_term = this->weights[node][node], neumann_term;
        neumann_term = rhs*neu_term;
        b.SumIntoGlobalValues(1, &neumann_term, &id_left);
        neumann_term = -rhs*neu_term;
        b.SumIntoGlobalValues(1, &neumann_term, &id_right);

        for (std::map<EntityHandle, double>::iterator it = this->weights[node].begin(); it != this->weights[node].end(); ++it) {
            if (it->first == node) {
                continue;
            }
            this->mb->tag_get_data(this->tags[global_id], &(it->first), 1, &vol_id);
            col_value = it->second * rhs;
            A.InsertGlobalValues(id_right, 1, &col_value, &vol_id); col_value *= -1;
            A.InsertGlobalValues(id_left, 1, &col_value, &vol_id);
        }
    }
    else if (std::find(this->internal_nodes.begin(), this->internal_nodes.end(), node) != this->internal_nodes.end()) {
        for (std::map<EntityHandle, double>::iterator it = this->weights[node].begin(); it != this->weights[node].end(); ++it) {
            this->mb->tag_get_data(this->tags[global_id], &(it->first), 1, &vol_id);
            col_value = it->second * rhs;
            A.InsertGlobalValues(id_right, 1, &col_value, &vol_id); col_value *= -1;
            A.InsertGlobalValues(id_left, 1, &col_value, &vol_id);
        }
    }
}

void MPFADSolver::visit_neumann_faces (Epetra_CrsMatrix& A, Epetra_Vector& b, Range neumann_faces) {
    Range vols_sharing_face, face_vertices;
    int vol_id = -1, i = 0;
    double face_area = 0.0, n_IJK[3];

    double *vert_coords = (double*) calloc(9, sizeof(double));
    double *faces_flow = (double*) calloc(neumann_faces.size(), sizeof(double));
    this->mb->tag_get_data(this->tags[neumann], neumann_faces, faces_flow);

    for (Range::iterator it = neumann_faces.begin(); it != neumann_faces.end(); ++it, ++i) {
        this->mtu->get_bridge_adjacencies(*it, 2, 3, vols_sharing_face);
        this->mb->tag_get_data(this->tags[global_id], &(*vols_sharing_face.begin()), 1, &vol_id);
        this->mtu->get_bridge_adjacencies(*it, 2, 0, face_vertices);
        this->mb->get_coords(face_vertices, vert_coords);
        geoutils::normal_vector(vert_coords, n_IJK);
        face_area = geoutils::face_area(n_IJK);
        double rhs = -faces_flow[i]*face_area;
        b.SumIntoGlobalValues(1, &rhs, &vol_id);
        vols_sharing_face.clear();
        face_vertices.clear();
    }

    free(vert_coords);
    free(faces_flow);
}

void MPFADSolver::visit_dirichlet_faces (Epetra_CrsMatrix& A, Epetra_Vector& b, Range dirichlet_faces) {
    Range face_vertices, vols_sharing_face;
    int vol_id = -1;
    double face_area = 0.0, h_L = 0, k_n_L = 0, k_L_JI = 0, k_L_JK = 0,
            d_JK = 0, d_JI = 0, k_eq = 0, rhs = 0;
    double n_IJK[3], *tan_JI = NULL, *tan_JK = NULL, *vert_coords = NULL;
    double i[3], j[3], k[3], l[3], lj[3];
    double node_pressure[3], k_L[9], temp[3] = {0, 0, 0};

    tan_JI = (double*) calloc(3, sizeof(double));
    tan_JK = (double*) calloc(3, sizeof(double));

    vert_coords = (double*) calloc(9, sizeof(double));

    for (Range::iterator it = dirichlet_faces.begin(); it != dirichlet_faces.end(); ++it) {
        this->mtu->get_bridge_adjacencies(*it, 2, 0, face_vertices);
        this->mb->get_coords(face_vertices, vert_coords);
        this->mtu->get_bridge_adjacencies(*it, 2, 3, vols_sharing_face);
        this->mb->tag_get_data(this->tags[dirichlet], face_vertices, &node_pressure);

        // Dividing vertices coordinate array into three points.
        i[0] = vert_coords[0]; i[1] = vert_coords[1]; i[2] = vert_coords[2];
        j[0] = vert_coords[3]; j[1] = vert_coords[4]; j[2] = vert_coords[5];
        k[0] = vert_coords[6]; k[1] = vert_coords[7]; k[2] = vert_coords[8];

        // Retrieving left volume centroid.
        EntityHandle left_volume = vols_sharing_face[0];
        this->mb->tag_get_data(this->tags[centroid], &left_volume, 1, &l);

        geoutils::normal_vector(vert_coords, n_IJK);
        face_area = geoutils::face_area(n_IJK);

        cblas_dcopy(3, &j[0], 1, &lj[0], 1);    // LJ = J
        cblas_daxpy(3, -1, &l[0], 1, &lj[0], 1);    // LJ = J - L

        // Checking if the normal vector is oriented outward.
        double _test = cblas_ddot(3, &lj[0], 1, &n_IJK[0], 1);
        if (_test < 0.0) {
            std::swap(i, k);
            cblas_dscal(3, -1.0, &n_IJK[0], 1);
        }
        else {  // Why does get_bridge_adjacencies reverses its order in C++?
            std::swap(node_pressure[0], node_pressure[2]);
        }

        // Calculating tangential terms.
        // REVIEW: Investigate why those terms are swapped if it is implemented
        // as the original code sugests.
        cblas_dcopy(3, &k[0], 1, &tan_JI[0], 1);  // tan_JI = i
        cblas_daxpy(3, -1, &j[0], 1, &tan_JI[0], 1);  // tan_JI = -j + tan_JI = i - j
        geoutils::cross_product(n_IJK, tan_JI, temp);    // tan_JI = n_IJK x tan_JI = n_IJK x (i - j)
        cblas_dcopy(3, &temp[0], 1, &tan_JI[0], 1);

        cblas_dcopy(3, &i[0], 1, &tan_JK[0], 1);
        cblas_daxpy(3, -1, &j[0], 1, &tan_JK[0], 1);
        geoutils::cross_product(n_IJK, tan_JK, temp);
        cblas_dcopy(3, &temp[0], 1, &tan_JK[0], 1);

        cblas_dscal(3, 0.0, &temp[0], 1);

        // Calculating the distance between the normal vector to the face
        // and the vector from the face to the centroid.
        h_L = fabs(cblas_ddot(3, &n_IJK[0], 1, &lj[0], 1)) / face_area;

        this->mb->tag_get_data(this->tags[permeability], &left_volume, 1, &k_L);

        // Calculating <<N_IJK, K_L>, N_IJK> = trans(trans(N_IJK)*K_L)*N_IJK,
        // i.e., TPFA term.
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 1, 3, 3, 1.0, &n_IJK[0], 1, &k_L[0], 3, 1.0, &temp[0], 3);
        k_n_L = cblas_ddot(3, &temp[0], 1, &n_IJK[0], 1) / pow(face_area, 2);
        cblas_dscal(3, 0.0, &temp[0], 1);

        // Same as <<N_IJK, K_L>, tan_JI> = trans(trans(N_IJK)*K_L)*tan_JI
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 1, 3, 3, 1.0, &n_IJK[0], 1, &k_L[0], 3, 1.0, &temp[0], 3);
        k_L_JI = cblas_ddot(3, &temp[0], 1, &tan_JI[0], 1) / pow(face_area, 2);
        cblas_dscal(3, 0.0, &temp[0], 1);

        // Same as <<N_IJK, K_L>, tan_JK> = trans(trans(N_IJK)*K_L)*tan_JK
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 1, 3, 3, 1.0, &n_IJK[0], 1, &k_L[0], 3, 1.0, &temp[0], 3);
        k_L_JK = cblas_ddot(3, &temp[0], 1, &tan_JK[0], 1) / pow(face_area, 2);
        cblas_dscal(3, 0.0, &temp[0], 1);

        d_JK = this->get_cross_diffusion_term(tan_JK, lj, face_area, h_L, k_n_L, k_L_JK, 0, 0, 0, true);
        d_JI = this->get_cross_diffusion_term(tan_JI, lj, face_area, h_L, k_n_L, k_L_JI, 0, 0, 0, true);

        k_eq = (face_area*k_n_L) / h_L;
        rhs = d_JK*(node_pressure[0] - node_pressure[1]) - k_eq*node_pressure[1] + d_JI*(node_pressure[1] - node_pressure[2]);
        rhs *= -1.0;

        this->mb->tag_get_data(this->tags[global_id], &left_volume, 1, &vol_id);
        b.SumIntoGlobalValues(1, &rhs, &vol_id);
        A.InsertGlobalValues(vol_id, 1, &k_eq, &vol_id);

        face_vertices.clear();
        vols_sharing_face.clear();
    }
}

void MPFADSolver::visit_internal_faces (Epetra_CrsMatrix& A, Epetra_Vector& b, Range internal_faces) {
    Range face_vertices, vols_sharing_face;
    double face_area = 0, d_JK = 0, d_JI = 0, k_eq = 0;
    double h_L = 0, k_n_L = 0, k_L_JI = 0, k_L_JK = 0;
    double h_R = 0, k_n_R = 0, k_R_JI = 0, k_R_JK = 0;
    double n_IJK[3], *tan_JI = NULL, *tan_JK = NULL, *vert_coords = NULL;
    double i[3], j[3], k[3], l[3], r[3], lj[3], rj[3], dist_LR[3];
    double k_L[9], k_R[9], temp[3] = {0, 0, 0};
    int cols_ids[2];
    double right_cols_values[2], left_cols_values[2];

    vert_coords = (double*) calloc(9, sizeof(double));

    tan_JK = (double*) calloc(3, sizeof(double));
    tan_JI = (double*) calloc(3, sizeof(double));

    for (Range::iterator it = internal_faces.begin(); it != internal_faces.end(); ++it) {
        this->mb->get_adjacencies(&(*it), 1, 0, false, face_vertices);
        this->mb->get_coords(face_vertices, vert_coords);
        this->mtu->get_bridge_adjacencies(*it, 2, 3, vols_sharing_face);

        // Dividing vertices coordinate array into three points.
        i[0] = vert_coords[0]; i[1] = vert_coords[1]; i[2] = vert_coords[2];
        j[0] = vert_coords[3]; j[1] = vert_coords[4]; j[2] = vert_coords[5];
        k[0] = vert_coords[6]; k[1] = vert_coords[7]; k[2] = vert_coords[8];

        EntityHandle left_volume = vols_sharing_face[0], right_volume = vols_sharing_face[1];
        this->mb->tag_get_data(this->tags[centroid], &left_volume, 1, &l);
        this->mb->tag_get_data(this->tags[centroid], &right_volume, 1, &r);

        cblas_dcopy(3, &r[0], 1, &dist_LR[0], 1);  // dist_LR = R
        cblas_daxpy(3, -1, &l[0], 1, &dist_LR[0], 1);  // dist_LR = R - L

        // Calculating normal term.
        geoutils::normal_vector(vert_coords, n_IJK);
        face_area = geoutils::face_area(n_IJK);

        double _test = cblas_ddot(3, &dist_LR[0], 1, &n_IJK[0], 1);
        if (_test < 0.0) {
            EntityHandle temp_handle = left_volume;
            left_volume = right_volume;
            right_volume = temp_handle;
            std::swap(r, l);
            cblas_dscal(3, -1, &dist_LR[0], 1);
        }

        // Calculating tangential terms.
        cblas_dcopy(3, &i[0], 1, &tan_JI[0], 1);  // tan_JI = i
        cblas_daxpy(3, -1, &j[0], 1, &tan_JI[0], 1);  // tan_JI = -j + tan_JI = -j + i
        geoutils::cross_product(n_IJK, tan_JI, temp);    // tan_JI = n_IJK x tan_JI = n_IJK x (-j + i)
        cblas_dcopy(3, &temp[0], 1, &tan_JI[0], 1);
        cblas_dscal(3, 0.0, &temp[0], 1);

        cblas_dcopy(3, &k[0], 1, &tan_JK[0], 1);
        cblas_daxpy(3, -1, &j[0], 1, &tan_JK[0], 1);
        geoutils::cross_product(n_IJK, tan_JK, temp);
        cblas_dcopy(3, &temp[0], 1, &tan_JK[0], 1);
        cblas_dscal(3, 0.0, &temp[0], 1);

        /* RIGHT VOLUME PERMEABILITY TERMS */

        this->mb->tag_get_data(this->tags[permeability], &right_volume, 1, &k_R);
        cblas_dcopy(3, &j[0], 1, &rj[0], 1);  // RJ = J
        cblas_daxpy(3, -1, &r[0], 1, &rj[0], 1);  // RJ = J - R
        h_R = fabs(cblas_ddot(3, &n_IJK[0], 1, &rj[0], 1)) / face_area; // h_R = <N_IJK, RJ> / |N_IJK|

        // Calculating <<N_IJK, K_R>, N_IJK> = trans(trans(N_IJK)*K_R)*N_IJK,
        // i.e., TPFA term of the right volume.
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 1, 3, 3, 1.0, &n_IJK[0], 1, &k_R[0], 3, 1.0, &temp[0], 3);
        k_n_R = cblas_ddot(3, &temp[0], 1, &n_IJK[0], 1) / pow(face_area, 2);
        cblas_dscal(3, 0.0, &temp[0], 1);

        // Same as <<N_IJK, K_R>, tan_JI> = trans(trans(N_IJK)*K_R)*tan_JI
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 1, 3, 3, 1.0, &n_IJK[0], 1, &k_R[0], 3, 1.0, &temp[0], 3);
        k_R_JI = cblas_ddot(3, &temp[0], 1, &tan_JI[0], 1) / pow(face_area, 2);
        cblas_dscal(3, 0.0, &temp[0], 1);

        // Same as <<N_IJK, K_R>, tan_JK> = trans(trans(N_IJK)*K_R)*tan_JK
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 1, 3, 3, 1.0, &n_IJK[0], 1, &k_R[0], 3, 1.0, &temp[0], 3);
        k_R_JK = cblas_ddot(3, &temp[0], 1, &tan_JK[0], 1) / pow(face_area, 2);
        cblas_dscal(3, 0.0, &temp[0], 1);

        /* --------------- */

        /* LEFT VOLUME PERMEABILITY TERMS */

        this->mb->tag_get_data(this->tags[permeability], &left_volume, 1, &k_L);
        cblas_dcopy(3, &j[0], 1, &lj[0], 1);  // LJ = J
        cblas_daxpy(3, -1, &l[0], 1, &lj[0], 1);  // LJ = J - L
        h_L = fabs(cblas_ddot(3, &n_IJK[0], 1, &lj[0], 1)) / face_area; // h_L = <N_IJK, LJ> / |N_IJK|

        // Calculating <<N_IJK, K_L>, N_IJK> = trans(trans(N_IJK)*K_L)*N_IJK,
        // i.e., TPFA term of the left volume.
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 1, 3, 3, 1.0, &n_IJK[0], 1, &k_L[0], 3, 1.0, &temp[0], 3);
        k_n_L = cblas_ddot(3, &temp[0], 1, &n_IJK[0], 1) / pow(face_area, 2);
        cblas_dscal(3, 0.0, &temp[0], 1);

        // Same as <<N_IJK, K_L>, tan_JI> = trans(trans(N_IJK)*K_L)*tan_JI
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 1, 3, 3, 1.0, &n_IJK[0], 1, &k_L[0], 3, 1.0, &temp[0], 3);
        k_L_JI = cblas_ddot(3, &temp[0], 1, &tan_JI[0], 1) / pow(face_area, 2);
        cblas_dscal(3, 0.0, &temp[0], 1);

        // Same as <<N_IJK, K_L>, tan_JK> = trans(trans(N_IJK)*K_L)*tan_JK
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 1, 3, 3, 1.0, &n_IJK[0], 1, &k_L[0], 3, 1.0, &temp[0], 3);
        k_L_JK = cblas_ddot(3, &temp[0], 1, &tan_JK[0], 1) / pow(face_area, 2);
        cblas_dscal(3, 0.0, &temp[0], 1);

        /* --------------- */

        d_JI = this->get_cross_diffusion_term(tan_JI, dist_LR, face_area, h_L, k_n_L, k_L_JI, h_R, k_n_R, k_R_JI, false);
        d_JK = this->get_cross_diffusion_term(tan_JK, dist_LR, face_area, h_L, k_n_L, k_L_JK, h_R, k_n_R, k_R_JK, false);

        k_eq = (k_n_R * k_n_L / ((k_n_R * h_R) + (k_n_L * h_L))) * face_area;

        int id_left, id_right;
        this->mb->tag_get_data(this->tags[global_id], &left_volume, 1, &id_left);
        this->mb->tag_get_data(this->tags[global_id], &right_volume, 1, &id_right);

        this->node_treatment(face_vertices[0], id_left, id_right, k_eq, 0.0, d_JK, A, b);
        this->node_treatment(face_vertices[1], id_left, id_right, k_eq, d_JI, -d_JK, A, b);
        this->node_treatment(face_vertices[2], id_left, id_right, k_eq, -d_JI, 0.0, A, b);

        cols_ids[0] = id_right; cols_ids[1] = id_left;
        right_cols_values[0] = k_eq; right_cols_values[1] = -k_eq;
        left_cols_values[0] = -k_eq; left_cols_values[1] = k_eq;

        A.InsertGlobalValues(id_right, 2, &right_cols_values[0], &cols_ids[0]);
        A.InsertGlobalValues(id_left, 2, &left_cols_values[0], &cols_ids[0]);

        face_vertices.clear();
        vols_sharing_face.clear();
    }
}
