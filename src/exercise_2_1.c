#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

#include <mpi.h>

#include "array.h"
#include "multiply.h"

static unsigned int const seed = 1234;
static int const dimensions[] = {128*1, 128*2, 128*4, 128*8};
static int const n_dimensions = sizeof(dimensions)/sizeof(int);
static double const epsilon = 1e-10;

typedef void (*GEMM)(
    int const m, int const k, int const n,
    double const* const A, double const* const B, double* const C
);

static void populate_compatible_random_matrix_pairs(
    int const m, int const k, int const n,
    int const seed,
    double* const A, double* const B)
{
    set_initilize_rand_seed(seed);

    initialize_2d_double_blocked_rand(A, m, k);
    initialize_2d_double_blocked_rand(B, k, n);
}

static void initialize_problem_matrices(
    int const m, int const k, int const n,
    double** const A, double** const B, double** const C)
{
    *A = allocate_2d_double_blocked(m, k);
    *B = allocate_2d_double_blocked(k, n);
    *C = allocate_2d_double_blocked(m, n);
}

static void destroy_problem_matrices(double** const A, double** const B, double** const C)
{
    *A = free_2d_double_blocked(*A);
    *C = free_2d_double_blocked(*C);
    *C = free_2d_double_blocked(*C);
}

static bool test_muptiply(int const m, int const k, int const n, GEMM gemm, double const epsilon, unsigned int const seed)
{
    double* A = NULL;
    double* B = NULL;
    double* C = NULL;
    initialize_problem_matrices(m, k, n, &A, &B, &C);
    populate_compatible_random_matrix_pairs(m, k, n, seed, A, B);

    gemm(m, k, n, A, B, C);
    bool result_is_correct = is_product(m, k, n, A, B, C, epsilon);

    destroy_problem_matrices(&A, &B, &C);

    return result_is_correct;
}

// Implement a function "parallel_gemm" of type GEMM, that implements the
// matrix multiplication operation.
//
// void parallel_gemm(
//   int const m, int const k, int const n,
//   double const* const A, double const* const B, double* const C)
// {
// }
//
void parallel_gemm(
    int const m, int const k, int const n,
    double const* const A, double const* const B, double* const C)
{
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int cols_per_proc = n / world_size;          // n is multiple of p
    int local_n = cols_per_proc;

    double *local_B = malloc(k * local_n * sizeof(double));
    double *local_C = malloc(m * local_n * sizeof(double));

    // scatter B by contiguous column blocks
    MPI_Scatter(B, k * local_n, MPI_DOUBLE,
                local_B, k * local_n, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // local matrix multiply: A (m x k) * local_B (k x local_n) = local_C (m x local_n)
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < local_n; ++j) {
            double sum = 0.0;
            for (int t = 0; t < k; ++t) {
                sum += A[i*k + t] * local_B[t*local_n + j];
            }
            local_C[i*local_n + j] = sum;
        }
    }

    // gather column blocks of C
    MPI_Gather(local_C, m * local_n, MPI_DOUBLE,
               C,        m * local_n, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    free(local_B);
    free(local_C);
}

// Then set "tested_gemm" to the address of your funtion
//GEMM const tested_gemm = &multiply_matrices;
GEMM const tested_gemm = &parallel_gemm;

static bool generate_square_matrix_dimension(int* const m, int* const k, int* const n)
{
    int const max_dim = n_dimensions;
    static int dim = 0;

    if (dim >= max_dim) {
        return false;
    }

    *m = dimensions[dim];
    *k = dimensions[dim];
    *n = dimensions[dim];
    
    dim++;

    return true;
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    bool all_test_pass = true;

    int m = 0;
    int k = 0;
    int n = 0;

    while (generate_square_matrix_dimension(&m, &k, &n)) {
        bool const test_pass = test_muptiply(m, k, n, tested_gemm, epsilon, seed);
        if (!test_pass) {
            printf("Multiplication failed for: m=%d, k=%d, n=%d\n", m, k, n);
            all_test_pass = false;
        }
    }

    if (!all_test_pass) {
        return EXIT_FAILURE;
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
