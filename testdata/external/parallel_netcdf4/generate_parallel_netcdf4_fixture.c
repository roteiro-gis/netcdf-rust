#include <mpi.h>
#include <netcdf.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK(call)                                                            \
    do {                                                                       \
        int status = (call);                                                   \
        if (status != NC_NOERR) {                                              \
            fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,                \
                    nc_strerror(status));                                      \
            MPI_Abort(MPI_COMM_WORLD, status);                                 \
        }                                                                      \
    } while (0)

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "parallel_nc4_compat.nc";
    int ncid, dimids[2], varid;
    float values[2][5] = {{0, 1, 2, 3, 4}, {10, 11, 12, 13, 14}};

    MPI_Init(&argc, &argv);
    CHECK(nc_create_par(path, NC_CLOBBER | NC_NETCDF4, MPI_COMM_WORLD,
                        MPI_INFO_NULL, &ncid));
    CHECK(nc_def_dim(ncid, "rank", 2, &dimids[0]));
    CHECK(nc_def_dim(ncid, "sample", 5, &dimids[1]));
    CHECK(nc_def_var(ncid, "values", NC_FLOAT, 2, dimids, &varid));
    CHECK(nc_put_att_text(ncid, varid, "units", 1, "1"));
    CHECK(nc_enddef(ncid));
    CHECK(nc_var_par_access(ncid, varid, NC_COLLECTIVE));
    CHECK(nc_put_var_float(ncid, varid, &values[0][0]));
    CHECK(nc_close(ncid));
    MPI_Finalize();
    return EXIT_SUCCESS;
}
