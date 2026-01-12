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
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int local_n = n / size;  

    double *A_local = (double *) malloc(m * k * sizeof(double));
    if (!A_local) {
        fprintf(stderr, "Rank %d: malloc failed for A_local\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        memcpy(A_local, A, m * k * sizeof(double));
    }

    MPI_Bcast(A_local, m * k, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double *B_local = (double *) malloc(k * local_n * sizeof(double));
    if (!B_local) {
        fprintf(stderr, "Rank %d: malloc failed for B_local\n", rank);
        free(A_local);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Scatter(
        B,                  
        k * local_n,        
        MPI_DOUBLE,
        B_local,            
        k * local_n,
        MPI_DOUBLE,
        0,                  // root
        MPI_COMM_WORLD
    );

    double *C_local = (double *) calloc(m * local_n, sizeof(double));
    if (!C_local) {
        fprintf(stderr, "Rank %d: calloc failed for C_local\n", rank);
        free(A_local);
        free(B_local);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < local_n; j++) {
            double sum = 0.0;
            for (int p = 0; p < k; p++) {
                sum += A_local[i * k + p] * B_local[p * local_n + j];
            }
            C_local[i * local_n + j] = sum;
        }
    }


    MPI_Gather(
        C_local,            
        m * local_n,        
        MPI_DOUBLE,
        C,                  
        m * local_n,        
        MPI_DOUBLE,
        0,                  // root
        MPI_COMM_WORLD
    );

    free(A_local);
    free(B_local);
    free(C_local);
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
