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

    int bs = n / size;  

    double *A_local = (double *)malloc(bs * n * sizeof(double));
    double *B_local = (double *)malloc(bs * n * sizeof(double));
    double *C_local = (double *)calloc(bs * n, sizeof(double));  

    double *A_recv  = (double *)malloc(bs * n * sizeof(double)); 

    MPI_Scatter(A, bs * n, MPI_DOUBLE, A_local, bs * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(B, bs * n, MPI_DOUBLE, B_local, bs * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int t = 0; t < size; t++) {
        int b_block_idx = (rank + t) % size;

        for (int i = 0; i < bs; i++) {             
            for (int j = 0; j < n; j++) {           
                double sum = 0.0;
                for (int kk = 0; kk < n; kk++) {
                    sum += A_local[i * n + kk] * B_local[(b_block_idx * bs + kk) * n + j];
                }
                C_local[i * n + j] += sum;
            }
        }

        if (t < size - 1) {
            int left  = (rank - 1 + size) % size;
            int right = (rank + 1) % size;

            MPI_Sendrecv(
                A_local, bs * n, MPI_DOUBLE, left,  0xCANNON,
                A_recv,  bs * n, MPI_DOUBLE, right, 0xCANNON,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE
            );

            double *temp = A_local;
            A_local = A_recv;
            A_recv = temp;
        }
    }

    MPI_Gather(C_local, bs * n, MPI_DOUBLE,
               C,       bs * n, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    free(A_local);
    free(B_local);
    free(C_local);
    free(A_recv);
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
