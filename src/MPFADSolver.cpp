#include "MPFADSolver.h"

using namespace std;
using namespace moab;

MPFADSolver::MPFADSolver () : mb(new Core()),
                            pcomm(mb, MPI_COMM_WORLD),
                            topo_util(new MeshTopoUtil(mb)) {}

MPFADSolver::MPFADSolver (Interface* moab_interface) : mb(moab_interface),
                                                pcomm(mb, MPI_COMM_WORLD),
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
    // NOTE: Check how many layers need to be exchanged in a MPFA-D scheme.
    rval = this->pcomm->exchange_ghost_cells(GHOST_DIM, BRIDGE_DIM, 1, 0, true);
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
    rval = this->mb->tag_get_data(tag_handles[global_id], volumes, (void*) gids);
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
    this->assemble_matrix(A, b, volumes, tag_handles);
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
    rval = solver->mb->write_file(fname.c_str(), 0, write_opts.c_str(), &volumes_meshset, 1);
    if (rval != MB_SUCCESS) {
        throw runtime_error("write_file failed\n");
    }
}

void MPFADSolver::assemble_matrix (Epetra_CrsMatrix& A, Epetra_Vector& b, Range volumes, Tag* tag_handles) {

}

void MPFADSolver::set_pressure_tags (Epetra_Vector& X, Range& volumes) {

}

void MPFADSolver::init_tags () {
    // TODO: Check how to exchange tags for nodes and faces. Review ghost cell
    // exchange in run method.
    ErrorCode rval;

    rval = this->mb->tag_get_handle("GLOBAL_ID", this->tags[global_id]);
    rval = this->mb->tag_get_handle("PERMEABILITY", this->tags[permeability]);
    rval = this->mb->tag_get_handle("CENTROID", this->tags[centroid]);
    rval = this->mb->tag_get_handle("DIRICHLET", this->tags[dirichlet]);
    rval = this->mb->tag_get_handle("NEUMANN", this->tags[neumann]);
    rval = this->mb->tag_get_handle("SOURCE", this->tags[source]);
}
