#!/usr/bin/env bash
set -euo pipefail

base_dir="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

if ! command -v ncgen >/dev/null 2>&1; then
  echo "SKIPPED: ncgen not found; committed compatibility fixtures are unchanged"
  exit 0
fi
if ! command -v nc-config >/dev/null 2>&1; then
  echo "SKIPPED: nc-config not found; committed compatibility fixtures are unchanged"
  exit 0
fi

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

mkdir -p "$base_dir/pnetcdf" "$base_dir/parallel"

cat >"$tmp_dir/pnetcdf_cdf1_fixed.cdl" <<'CDL'
netcdf pnetcdf_cdf1_fixed {
dimensions:
    y = 3 ;
    x = 4 ;
variables:
    int fixed(y, x) ;
data:
    fixed =
        0, 1, 2, 3,
        10, 11, 12, 13,
        20, 21, 22, 23 ;
}
CDL
ncgen -1 -o "$base_dir/pnetcdf/pnetcdf_cdf1_fixed.nc" "$tmp_dir/pnetcdf_cdf1_fixed.cdl"

cat >"$tmp_dir/pnetcdf_cdf2_interleaved_records.cdl" <<'CDL'
netcdf pnetcdf_cdf2_interleaved_records {
dimensions:
    time = UNLIMITED ;
    x = 3 ;
variables:
    float temp(time, x) ;
    int quality(time) ;
data:
    temp =
        1, 2, 3,
        4, 5, 6,
        7, 8, 9 ;
    quality = 10, 20, 30 ;
}
CDL
ncgen -6 -o "$base_dir/pnetcdf/pnetcdf_cdf2_interleaved_records.nc" "$tmp_dir/pnetcdf_cdf2_interleaved_records.cdl"

cat >"$tmp_dir/pnetcdf_cdf5_unsigned_int64.c" <<'C'
#include <netcdf.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK(call)                                                            \
    do {                                                                       \
        int status = (call);                                                   \
        if (status != NC_NOERR) {                                              \
            fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,                \
                    nc_strerror(status));                                      \
            return status;                                                     \
        }                                                                      \
    } while (0)

int main(int argc, char **argv) {
    int ncid, dimid, flags_var, ids_var, int64_var, uint64_var;
    int dimids[1];
    unsigned char flags[4] = {1, 2, 253, 254};
    unsigned int ids[4] = {1U, 4000000000U, 4000000001U, 4000000002U};
    long long int64_values[4] = {-5LL, -4LL, 9223372036854775806LL,
                                 9223372036854775807LL};
    unsigned long long uint64_values[4] = {1ULL, 2ULL,
                                           18446744073709551614ULL,
                                           18446744073709551615ULL};

    if (argc != 2) {
        return EXIT_FAILURE;
    }

    CHECK(nc_create(argv[1], NC_CLOBBER | NC_64BIT_DATA, &ncid));
    CHECK(nc_def_dim(ncid, "n", 4, &dimid));
    dimids[0] = dimid;
    CHECK(nc_def_var(ncid, "flags", NC_UBYTE, 1, dimids, &flags_var));
    CHECK(nc_def_var(ncid, "ids", NC_UINT, 1, dimids, &ids_var));
    CHECK(nc_def_var(ncid, "int64_values", NC_INT64, 1, dimids, &int64_var));
    CHECK(nc_def_var(ncid, "unsigned_big", NC_UINT64, 1, dimids, &uint64_var));
    CHECK(nc_enddef(ncid));
    CHECK(nc_put_var_uchar(ncid, flags_var, flags));
    CHECK(nc_put_var_uint(ncid, ids_var, ids));
    CHECK(nc_put_var_longlong(ncid, int64_var, int64_values));
    CHECK(nc_put_var_ulonglong(ncid, uint64_var, uint64_values));
    CHECK(nc_close(ncid));
    return EXIT_SUCCESS;
}
C
"$(nc-config --cc)" $(nc-config --cflags) "$tmp_dir/pnetcdf_cdf5_unsigned_int64.c" \
    -o "$tmp_dir/pnetcdf_cdf5_unsigned_int64" $(nc-config --libs)
"$tmp_dir/pnetcdf_cdf5_unsigned_int64" "$base_dir/pnetcdf/pnetcdf_cdf5_unsigned_int64.nc"

cat >"$tmp_dir/parallel_nc4_compat.cdl" <<'CDL'
netcdf parallel_nc4_compat {
dimensions:
    rank = 2 ;
    sample = 5 ;
variables:
    float values(rank, sample) ;
        values:units = "1" ;
data:
    values =
        0, 1, 2, 3, 4,
        10, 11, 12, 13, 14 ;
}
CDL
ncgen -4 -o "$base_dir/parallel/parallel_nc4_compat.nc" "$tmp_dir/parallel_nc4_compat.cdl"
