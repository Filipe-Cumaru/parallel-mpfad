#include "MPFADSolver.h"

using namespace std;
using namespace moab;

MPFADSolver::MPFADSolver () : mb(new Core()),
                            pcomm(new ParallelComm(mb, MPI_COMM_WORLD)),
                            topo_util(new MeshTopoUtil(mb)) {}

MPFADSolver::MPFADSolver (Interface* moab_interface) : mb(moab_interface),
                                                pcomm(new ParallelComm(mb, MPI_COMM_WORLD)),
                                                topo_util(new MeshTopoUtil(mb)) {}

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
    Range volumes;
    rval = this->mb->get_entities_by_dimension(0, 3, volumes, false);
    if (rval != MB_SUCCESS) {
        throw runtime_error("Could not retrieve volumes from the mesh.\n");
    }

    // Exchange volumes sharing any vertex in an interface.
    rval = this->pcomm->exchange_ghost_cells(3, 0, 1, 0, true);
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
    this->assemble_matrix(A, b, volumes);
    ts = clock() - ts;
    printf("<%d> Done. Time elapsed: %f\n", rank, ((double) ts)/CLOCKS_PER_SEC);

    Epetra_LinearProblem linear_problem (&A, &X, &b);
    AztecOO solver (linear_problem);

    // Setting up solver preconditioning
    Teuchos::ParameterList MLList;
    ML_Epetra::MultiLevelPreconditioner * MLPrec = new ML_Epetra::MultiLevelPreconditioner(A, true);
    MLList.set("max levels", 4);
    MLList.set("repartition: enable", 1);
    MLList.set("repartition: partitioner", "ParMetis");
    MLList.set("coarse: type", "Chebyshev");
    MLList.set("coarse: sweeps", 2);
    MLList.set("smoother: type", "Chebyshev");
    MLList.set("aggregation: type", "METIS");
    ML_Epetra::SetDefaults("SA", MLList);
    solver.SetPrecOperator(MLPrec);
    solver.SetAztecOption(AZ_kspace, 250);
    solver.SetAztecOption(AZ_solver, AZ_gmres_condnum);
    solver.Iterate(1000, 1e-14);
    delete MLPrec;

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
    std::vector<Tag> tag_vector (&this->tags[0], &this->tags[6]+1);

    rval = this->mb->tag_get_handle("GLOBAL_ID", this->tags[global_id]);
    rval = this->mb->tag_get_handle("PERMEABILITY", this->tags[permeability]);
    rval = this->mb->tag_get_handle("CENTROID", this->tags[centroid]);
    rval = this->mb->tag_get_handle("DIRICHLET", this->tags[dirichlet]);
    rval = this->mb->tag_get_handle("NEUMANN", this->tags[neumann]);
    rval = this->mb->tag_get_handle("SOURCE", this->tags[source]);

    rval = this->pcomm->exchange_tags(tag_vector, tag_vector, empty_set);
    if (rval != MB_SUCCESS) {
        throw runtime_error("exchange_tags failed");
    }
}

void MPFADSolver::assemble_matrix (Epetra_CrsMatrix& A, Epetra_Vector& b, Range volumes) {
    ErrorCode rval;

    // Retrieving Dirichlet faces and nodes.
    Range dirichlet_faces, dirichlet_nodes;
    rval = this->mb->get_entities_by_type_and_tag(0, MBTRI,
                    &this->tags[dirichlet], NULL, 1, dirichlet_faces);
    if (rval != MB_SUCCESS) {
        throw runtime_error("Unable to get dirichlet entities");
    }
    rval = this->mb->get_entities_by_type_and_tag(0, MBVERTEX,
                    &this->tags[dirichlet], NULL, 1, dirichlet_nodes);
    if (rval != MB_SUCCESS) {
        throw runtime_error("Unable to get dirichlet entities");
    }

    // Retrieving Neumann faces and nodes. Notice that faces/nodes
    // that are also Dirichlet faces/nodes are filtered.
    Range neumann_faces, neumann_nodes;
    rval = this->mb->get_entities_by_type_and_tag(0, MBTRI,
                    &this->tags[neumann], NULL, 1, neumann_faces);
    if (rval != MB_SUCCESS) {
        throw runtime_error("Unable to get neumann entities");
    }
    rval = this->mb->get_entities_by_type_and_tag(0, MBVERTEX,
                    &this->tags[neumann], NULL, 1, neumann_nodes);
    if (rval != MB_SUCCESS) {
        throw runtime_error("Unable to get neumann entities");
    }
    neumann_faces = subtract(neumann_faces, dirichlet_faces);
    neumann_nodes = subtract(neumann_nodes, dirichlet_nodes);

    // Get internal faces and nodes.
    Range internal_faces, internal_nodes;
    rval = this->mb->get_entities_by_dimension(0, 2, internal_faces);
    if (rval != MB_SUCCESS) {
        throw runtime_error("Unable to get internal faces");
    }
    internal_faces = subtract(internal_faces, neumann_faces);
    internal_faces = subtract(internal_faces, dirichlet_faces);
    rval = this->mb->get_entities_by_dimension(0, 0, internal_nodes);
    if (rval != MB_SUCCESS) {
        throw runtime_error("Unable to get internal nodes");
    }
    internal_nodes = subtract(internal_nodes, neumann_nodes);
    internal_nodes = subtract(internal_nodes, dirichlet_nodes);

    // Node interpolation here.

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
        b[volume_id] += source_term;
        A.InsertGlobalValues(volume_id, 1, &source_term, &volume_id);
    }

    this->visit_neumann_faces(A, b, neumann_faces);
    this->visit_dirichlet_faces(A, b, dirichlet_faces);
    this->visit_internal_faces(A, b, internal_faces);
}

void MPFADSolver::set_pressure_tags (Epetra_Vector& X, Range& volumes) {

}

void MPFADSolver::visit_neumann_faces (Epetra_CrsMatrix& A, Epetra_Vector& b, Range neumann_faces) {
    ErrorCode rval;
    Range vols_sharing_face, face_vertices;
    int vol_id = -1;

    double *vert_coords = (double*) calloc(9, sizeof(double));
    double *faces_flow = (double*) calloc(neumann_faces.size(), sizeof(double));
    rval = this->mb->tag_get_data(this->tags[neumann], neumann_faces, faces_flow);
    if (rval != MB_SUCCESS) {
        throw runtime_error("Unable to get Neumann BC");
    }

    for (Range::iterator it = neumann_faces.begin(); it != neumann_faces.end(); ++it) {
        // TODO: Add exception treatment.
        rval = this->topo_util->get_bridge_adjacencies(*it, 2, 3, vols_sharing_face);
        rval = this->mb->tag_get_data(this->tags[global_id], &(*vols_sharing_face.begin()), 1, &vol_id);
        rval = this->mb->get_adjacencies(&(*it), 1, 0, false, face_vertices);
        rval = this->mb->get_coords(face_vertices, vert_coords);
    }
}

void MPFADSolver::visit_dirichlet_faces (Epetra_CrsMatrix& A, Epetra_Vector& b, Range dirichlet_faces) {
    return;
}

void MPFADSolver::visit_internal_faces (Epetra_CrsMatrix& A, Epetra_Vector& b, Range internal_faces) {
    return;
}
