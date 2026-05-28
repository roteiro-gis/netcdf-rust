#include <mpi.h>
#include <pnetcdf.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK(call)                                                            \
    do {                                                                       \
        int status = (call);                                                   \
        if (status != NC_NOERR) {                                              \
            fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,                \
                    ncmpi_strerror(status));                                   \
            MPI_Abort(MPI_COMM_WORLD, status);                                 \
        }                                                                      \
    } while (0)

static void path_join(char *out, size_t out_len, const char *dir,
                      const char *name) {
    snprintf(out, out_len, "%s/%s", dir, name);
}

static void write_cdf1_fixed(const char *dir) {
    char path[4096];
    int ncid, dimids[2], varid;
    int data[3][4] = {{0, 1, 2, 3}, {10, 11, 12, 13}, {20, 21, 22, 23}};

    path_join(path, sizeof(path), dir, "pnetcdf_cdf1_fixed.nc");
    CHECK(ncmpi_create(MPI_COMM_WORLD, path, NC_CLOBBER, MPI_INFO_NULL, &ncid));
    CHECK(ncmpi_def_dim(ncid, "y", 3, &dimids[0]));
    CHECK(ncmpi_def_dim(ncid, "x", 4, &dimids[1]));
    CHECK(ncmpi_def_var(ncid, "fixed", NC_INT, 2, dimids, &varid));
    CHECK(ncmpi_enddef(ncid));
    CHECK(ncmpi_put_var_int_all(ncid, varid, &data[0][0]));
    CHECK(ncmpi_close(ncid));
}

static void write_cdf2_interleaved_records(const char *dir) {
    char path[4096];
    int ncid, time_dim, x_dim, temp_var, quality_var;
    int temp_dims[2], quality_dims[1];
    MPI_Offset temp_start[2], temp_count[2], quality_start[1], quality_count[1];
    float temp[3][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    int quality[3] = {10, 20, 30};

    path_join(path, sizeof(path), dir, "pnetcdf_cdf2_interleaved_records.nc");
    CHECK(ncmpi_create(MPI_COMM_WORLD, path, NC_CLOBBER | NC_64BIT_OFFSET,
                       MPI_INFO_NULL, &ncid));
    CHECK(ncmpi_def_dim(ncid, "time", NC_UNLIMITED, &time_dim));
    CHECK(ncmpi_def_dim(ncid, "x", 3, &x_dim));
    temp_dims[0] = time_dim;
    temp_dims[1] = x_dim;
    quality_dims[0] = time_dim;
    CHECK(ncmpi_def_var(ncid, "temp", NC_FLOAT, 2, temp_dims, &temp_var));
    CHECK(ncmpi_def_var(ncid, "quality", NC_INT, 1, quality_dims, &quality_var));
    CHECK(ncmpi_enddef(ncid));

    temp_start[0] = 0;
    temp_start[1] = 0;
    temp_count[0] = 3;
    temp_count[1] = 3;
    quality_start[0] = 0;
    quality_count[0] = 3;
    CHECK(ncmpi_put_vara_float_all(ncid, temp_var, temp_start, temp_count,
                                   &temp[0][0]));
    CHECK(ncmpi_put_vara_int_all(ncid, quality_var, quality_start, quality_count,
                                 quality));
    CHECK(ncmpi_close(ncid));
}

static void write_cdf5_unsigned_int64(const char *dir) {
    char path[4096];
    int ncid, dimid, flags_var, ids_var, int64_var, unsigned_var;
    int dimids[1];
    unsigned char flags[4] = {1, 2, 253, 254};
    unsigned int ids[4] = {1U, 4000000000U, 4000000001U, 4000000002U};
    long long int64_values[4] = {-5LL, -4LL, 9223372036854775806LL,
                                 9223372036854775807LL};
    unsigned long long unsigned_big[4] = {1ULL, 2ULL,
                                          18446744073709551614ULL,
                                          18446744073709551615ULL};

    path_join(path, sizeof(path), dir, "pnetcdf_cdf5_unsigned_int64.nc");
    CHECK(ncmpi_create(MPI_COMM_WORLD, path, NC_CLOBBER | NC_64BIT_DATA,
                       MPI_INFO_NULL, &ncid));
    CHECK(ncmpi_def_dim(ncid, "n", 4, &dimid));
    dimids[0] = dimid;
    CHECK(ncmpi_def_var(ncid, "flags", NC_UBYTE, 1, dimids, &flags_var));
    CHECK(ncmpi_def_var(ncid, "ids", NC_UINT, 1, dimids, &ids_var));
    CHECK(ncmpi_def_var(ncid, "int64_values", NC_INT64, 1, dimids, &int64_var));
    CHECK(ncmpi_def_var(ncid, "unsigned_big", NC_UINT64, 1, dimids, &unsigned_var));
    CHECK(ncmpi_enddef(ncid));
    CHECK(ncmpi_put_var_uchar_all(ncid, flags_var, flags));
    CHECK(ncmpi_put_var_uint_all(ncid, ids_var, ids));
    CHECK(ncmpi_put_var_longlong_all(ncid, int64_var, int64_values));
    CHECK(ncmpi_put_var_ulonglong_all(ncid, unsigned_var, unsigned_big));
    CHECK(ncmpi_close(ncid));
}

int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : ".";
    MPI_Init(&argc, &argv);
    write_cdf1_fixed(dir);
    write_cdf2_interleaved_records(dir);
    write_cdf5_unsigned_int64(dir);
    MPI_Finalize();
    return EXIT_SUCCESS;
}
