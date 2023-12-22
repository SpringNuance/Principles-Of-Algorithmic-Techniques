/*************** A matrix multiplication test using cache-blocking and AVX2. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <immintrin.h> 
#include <omp.h>

/********************************************** Transposition (unoptimized). */

void tp(int len, int rank, const int *dim, const int *perm, long *below,
        const double *in, double *out) 
{
    // newdim[j] == dim[perm[j]] for j = 0,1,...,rank-1
    for(int d = 0; d < rank; d++) {
        int pd = perm[d];
        long b = 1;
        for(int j = 0; j < pd; j++)
            b *= dim[j];
        below[d] = b;
    }
    #pragma omp parallel for
    for(long u = 0; u < len; u++) {
        long uu = u;
        long v = 0;
        for(int d = 0; d < rank; d++) {
            v += (uu%dim[perm[d]])*below[d];
            uu /= dim[perm[d]];
        }       
        out[u] = in[v];
    }
}

/******************************************** The multiplication subroutine. */

void mm(int             size,
        double *        result,
        const double *  left,
        const double *  right)
{
    double wstart = omp_get_wtime();

    assert((size % 128) == 0);
    int len = size*size;
    int size_inner = 32;
    int size_mid = 128/32;     // size in inner blocks
    int size_outer = size/128; // size in mid blocks

    int len_inner = size_inner*size_inner;
    int len_mid = size_mid*size_mid*len_inner;

    double *result_a = (double *) 0;
    double *left_a   = (double *) 0;
    double *right_a  = (double *) 0;
    // page-align 4096 or AVX2-align 32
    posix_memalign((void**)&result_a, 32, sizeof(double)*len);
    posix_memalign((void**)&left_a,   32, sizeof(double)*len);
    posix_memalign((void**)&right_a,  32, sizeof(double)*len);

    /* Transpose inputs to aligned & cache-blocked form. */
    {       
        int rank = 6;
        long below[6];
        const int dim[] = {
            size_inner, size_mid, size_outer, size_inner, size_mid, size_outer 
        };                  
        const int perm[] = { 0, 3, 1, 4, 2, 5 };                    
        tp(len, rank, dim, perm, below, left, left_a);
        tp(len, rank, dim, perm, below, right, right_a);
    }

    #pragma omp parallel for
    for(long i = 0; i < len; i++)
        result_a[i] = 0.0;

    double wstart_inner = omp_get_wtime();

    double *c = result_a;
    double *a = left_a;
    double *b = right_a;

    #pragma omp parallel for
    for(int q = 0; q < size_outer*size_outer; q++) {
        int i_outer = q / size_outer;
        int k_outer = q % size_outer;
        double *c_outer = c + (i_outer*size_outer + k_outer)*len_mid;
        for(int j_outer = 0; j_outer < size_outer; j_outer++) {
            double *a_outer = a + (i_outer*size_outer + j_outer)*len_mid;
            double *b_outer = b + (j_outer*size_outer + k_outer)*len_mid;
            double *a_outer_p = a + (i_outer*size_outer + j_outer + 1)*len_mid;
            double *b_outer_p = b + ((j_outer+1)*size_outer + k_outer)*len_mid;
            for(int p = 0; p < size_mid*size_mid; p++) {
                int i_mid = p / size_mid;
                int k_mid = p % size_mid;
                double *c_mid = c_outer + (i_mid*size_mid + k_mid)*len_inner;
                for(int j_mid = 0; j_mid < size_mid; j_mid++) {
                    double *a_mid = a_outer + (i_mid*size_mid+j_mid)*len_inner;
                    double *b_mid = b_outer + (j_mid*size_mid+k_mid)*len_inner;
                    for(int i = 0; i < 30; i+=3) { 
                        _mm_prefetch(a_outer_p, _MM_HINT_NTA);
                        _mm_prefetch(b_outer_p, _MM_HINT_NTA);
                        a_outer_p += 8;
                        b_outer_p += 8;
                        _mm_prefetch(a_outer_p, _MM_HINT_NTA);
                        _mm_prefetch(b_outer_p, _MM_HINT_NTA);
                        a_outer_p += 8;
                        b_outer_p += 8;
                        _mm_prefetch(a_outer_p, _MM_HINT_NTA);
                        _mm_prefetch(b_outer_p, _MM_HINT_NTA);
                        a_outer_p += 8;
                        b_outer_p += 8;
                        for(int k = 0; k < size_inner; k += 16) {
                            double *a_inner = a_mid + i*size_inner;
                            double *b_inner = b_mid + k;
                            double *c_inner = c_mid + i*size_inner + k;
                            int iter = size_inner;
                            int width = size_inner;
                            __asm__ volatile
                            (  /* Assembler template */
                            "  mov            %[c], %%rbx                 \n\t"
                            "  mov            %[wd], %%rcx                \n\t"
                            "  shl            $0x3, %%rcx                 \n\t"
                            "  vmovapd        (%%rbx), %%ymm4             \n\t"
                            "  vmovapd        (%%rbx,%%rcx), %%ymm5       \n\t"
                            "  vmovapd        (%%rbx,%%rcx,2), %%ymm6     \n\t"
                            "  vmovapd        0x20(%%rbx), %%ymm7         \n\t"
                            "  vmovapd        0x20(%%rbx,%%rcx), %%ymm8   \n\t"
                            "  vmovapd        0x20(%%rbx,%%rcx,2), %%ymm9 \n\t"
                            "  vmovapd        0x40(%%rbx), %%ymm10        \n\t"
                            "  vmovapd        0x40(%%rbx,%%rcx), %%ymm11  \n\t"
                            "  vmovapd        0x40(%%rbx,%%rcx,2), %%ymm12\n\t"
                            "  vmovapd        0x60(%%rbx), %%ymm13        \n\t"
                            "  vmovapd        0x60(%%rbx,%%rcx), %%ymm14  \n\t"
                            "  vmovapd        0x60(%%rbx,%%rcx,2), %%ymm15\n\t"
                            "  mov            %[a], %%rdx                 \n\t"
                            "  mov            %[b], %%rax                 \n\t"
                            "  mov            %[iter], %%rbx              \n\t"
                            ".testinner10:                                \n\t"
                            "  vbroadcastsd   (%%rdx), %%ymm0             \n\t"
                            "  vbroadcastsd   (%%rdx,%%rcx), %%ymm1       \n\t"
                            "  vbroadcastsd   (%%rdx,%%rcx,2), %%ymm2     \n\t"
                            "  add            $0x08, %%rdx                \n\t"
                            "  vmovapd        (%%rax), %%ymm3             \n\t"
                            "  vfmadd231pd    %%ymm3, %%ymm0, %%ymm4      \n\t"
                            "  vfmadd231pd    %%ymm3, %%ymm1, %%ymm5      \n\t"
                            "  vfmadd231pd    %%ymm3, %%ymm2, %%ymm6      \n\t"
                            "  vmovapd        0x20(%%rax), %%ymm3         \n\t"
                            "  vfmadd231pd    %%ymm3, %%ymm0, %%ymm7      \n\t"
                            "  vfmadd231pd    %%ymm3, %%ymm1, %%ymm8      \n\t"
                            "  vfmadd231pd    %%ymm3, %%ymm2, %%ymm9      \n\t"
                            "  vmovapd        0x40(%%rax), %%ymm3         \n\t"
                            "  vfmadd231pd    %%ymm3, %%ymm0, %%ymm10     \n\t"
                            "  vfmadd231pd    %%ymm3, %%ymm1, %%ymm11     \n\t"
                            "  vfmadd231pd    %%ymm3, %%ymm2, %%ymm12     \n\t"
                            "  vmovapd        0x60(%%rax), %%ymm3         \n\t"
                            "  vfmadd231pd    %%ymm3, %%ymm0, %%ymm13     \n\t"
                            "  vfmadd231pd    %%ymm3, %%ymm1, %%ymm14     \n\t"
                            "  vfmadd231pd    %%ymm3, %%ymm2, %%ymm15     \n\t"
                            "  add            %%rcx, %%rax                \n\t"
                            "  dec            %%rbx                       \n\t"
                            "  jnz            .testinner10                \n\t"
                            "  mov            %[c], %%rbx                 \n\t"
                            "  vmovapd        %%ymm4, (%%rbx)             \n\t"
                            "  vmovapd        %%ymm5, (%%rbx,%%rcx)       \n\t"
                            "  vmovapd        %%ymm6, (%%rbx,%%rcx,2)     \n\t"
                            "  vmovapd        %%ymm7, 0x20(%%rbx)         \n\t"
                            "  vmovapd        %%ymm8, 0x20(%%rbx,%%rcx)   \n\t"
                            "  vmovapd        %%ymm9, 0x20(%%rbx,%%rcx,2) \n\t"
                            "  vmovapd        %%ymm10, 0x40(%%rbx)        \n\t"
                            "  vmovapd        %%ymm11, 0x40(%%rbx,%%rcx)  \n\t"
                            "  vmovapd        %%ymm12, 0x40(%%rbx,%%rcx,2)\n\t"
                            "  vmovapd        %%ymm13, 0x60(%%rbx)        \n\t"
                            "  vmovapd        %%ymm14, 0x60(%%rbx,%%rcx)  \n\t"
                            "  vmovapd        %%ymm15, 0x60(%%rbx,%%rcx,2)\n\t"
               /* 
                * Format for operands:
                *   [{asm symbolic name}] "{constraint}" ({C variable name})
                * Reference with "%[{asm symbolic name}]" in assembler template
                * Constraints:
                *   =  ~ overwrite               [for output operands]
                *   +  ~ both read and write     [for output operands]
                *   r  ~ register
                *   m  ~ memory
                */
                            : /* Output operands (comma-separated list) */
                            : /* Input operands (comma-separated list) */
                            [iter] "r" ((long) iter),
                            [wd]   "r" ((long) width),
                            [c]    "r" (c_inner),
                            [a]    "r" (a_inner),
                            [b]    "r" (b_inner)
                            : /* Clobbers 
                               * (comma-separted list of registers, 
                               *  e.g. "ymm12", "memory" for universal memory 
                               *  clobber) */
                            "rax",
                            "rbx",
                            "rcx",
                            "rdx",
                            "ymm0", 
                            "ymm1", 
                            "ymm2", 
                            "ymm3", 
                            "ymm4", 
                            "ymm5", 
                            "ymm6", 
                            "ymm7", 
                            "ymm8", 
                            "ymm9", 
                            "ymm10", 
                            "ymm11", 
                            "ymm12", 
                            "ymm13", 
                            "ymm14", 
                            "ymm15",
                            "memory"
                            );
                        }
                    }
                    for(int i = 30; i < 32; i+=3) {
                        _mm_prefetch(a_outer_p, _MM_HINT_NTA);
                        _mm_prefetch(b_outer_p, _MM_HINT_NTA);
                        a_outer_p += 8;
                        b_outer_p += 8;
                        _mm_prefetch(a_outer_p, _MM_HINT_NTA);
                        _mm_prefetch(b_outer_p, _MM_HINT_NTA);
                        a_outer_p += 8;
                        b_outer_p += 8;
                        for(int k = 0; k < size_inner; k += 16) {
                            double *a_inner = a_mid + i*size_inner;
                            double *b_inner = b_mid + k;
                            double *c_inner = c_mid + i*size_inner + k;
                            int iter = size_inner;
                            int width = size_inner;                         
                            __asm__ volatile
                            (  /* Assembler template */
                            "  mov            %[c], %%rbx                 \n\t"
                            "  mov            %[wd], %%rcx                \n\t"
                            "  shl            $0x3, %%rcx                 \n\t"
                            "  vmovapd        (%%rbx), %%ymm4             \n\t"
                            "  vmovapd        (%%rbx,%%rcx), %%ymm5       \n\t"
                            "  vmovapd        0x20(%%rbx), %%ymm7         \n\t"
                            "  vmovapd        0x20(%%rbx,%%rcx), %%ymm8   \n\t"
                            "  vmovapd        0x40(%%rbx), %%ymm10        \n\t"
                            "  vmovapd        0x40(%%rbx,%%rcx), %%ymm11  \n\t"
                            "  vmovapd        0x60(%%rbx), %%ymm13        \n\t"
                            "  vmovapd        0x60(%%rbx,%%rcx), %%ymm14  \n\t"
                            "  mov            %[a], %%rdx                 \n\t"
                            "  mov            %[b], %%rax                 \n\t"
                            "  mov            %[iter], %%rbx              \n\t"
                            ".testinner1:                                 \n\t"
                            "  vbroadcastsd   (%%rdx), %%ymm0             \n\t"
                            "  vbroadcastsd   (%%rdx,%%rcx), %%ymm1       \n\t"
                            "  add            $0x08, %%rdx                \n\t"
                            "  vmovapd        (%%rax), %%ymm3             \n\t"
                            "  vfmadd231pd    %%ymm3, %%ymm0, %%ymm4      \n\t"
                            "  vfmadd231pd    %%ymm3, %%ymm1, %%ymm5      \n\t"
                            "  vmovapd        0x20(%%rax), %%ymm3         \n\t"
                            "  vfmadd231pd    %%ymm3, %%ymm0, %%ymm7      \n\t"
                            "  vfmadd231pd    %%ymm3, %%ymm1, %%ymm8      \n\t"
                            "  vmovapd        0x40(%%rax), %%ymm3         \n\t"
                            "  vfmadd231pd    %%ymm3, %%ymm0, %%ymm10     \n\t"
                            "  vfmadd231pd    %%ymm3, %%ymm1, %%ymm11     \n\t"
                            "  vmovapd        0x60(%%rax), %%ymm3         \n\t"
                            "  vfmadd231pd    %%ymm3, %%ymm0, %%ymm13     \n\t"
                            "  vfmadd231pd    %%ymm3, %%ymm1, %%ymm14     \n\t"
                            "  add            %%rcx, %%rax                \n\t"
                            "  dec            %%rbx                       \n\t"
                            "  jnz            .testinner1                 \n\t"
                            "  mov            %[c], %%rbx                 \n\t"
                            "  vmovapd        %%ymm4, (%%rbx)             \n\t"
                            "  vmovapd        %%ymm5, (%%rbx,%%rcx)       \n\t"
                            "  vmovapd        %%ymm7, 0x20(%%rbx)         \n\t"
                            "  vmovapd        %%ymm8, 0x20(%%rbx,%%rcx)   \n\t"
                            "  vmovapd        %%ymm10, 0x40(%%rbx)        \n\t"
                            "  vmovapd        %%ymm11, 0x40(%%rbx,%%rcx)  \n\t"
                            "  vmovapd        %%ymm13, 0x60(%%rbx)        \n\t"
                            "  vmovapd        %%ymm14, 0x60(%%rbx,%%rcx)  \n\t"
               /* 
                * Format for operands:
                *   [{asm symbolic name}] "{constraint}" ({C variable name})
                * Reference with "%[{asm symbolic name}]" in assembler template
                * Constraints:
                *   =  ~ overwrite               [for output operands]
                *   +  ~ both read and write     [for output operands]
                *   r  ~ register
                *   m  ~ memory
                */
                            : /* Output operands (comma-separated list) */
                            : /* Input operands (comma-separated list) */
                            [iter] "r" ((long) iter),
                            [wd]   "r" ((long) width),
                            [c]    "r" (c_inner),
                            [a]    "r" (a_inner),
                            [b]    "r" (b_inner)
                            : /* Clobbers (comma-separated list of registers, 
                               *           e.g. "ymm12", "memory" for universal
                               *           mem clobber) */
                            "rax",
                            "rbx",
                            "rcx",
                            "rdx",
                            "ymm0", 
                            "ymm1", 
                            "ymm2", 
                            "ymm3", 
                            "ymm4", 
                            "ymm5", 
                            "ymm6", 
                            "ymm7", 
                            "ymm8", 
                            "ymm9", 
                            "ymm10", 
                            "ymm11", 
                            "ymm12", 
                            "ymm13", 
                            "ymm14", 
                            "ymm15",
                            "memory"
                            );
                        }
                    }
                }
            }
            assert(a_outer_p ==
                   a + (i_outer*size_outer + j_outer + 2)*len_mid);
            assert(b_outer_p ==
                   b + ((j_outer + 1)*size_outer + k_outer + 1)*len_mid);
        }
    }

    double wstop_inner = omp_get_wtime();
    double wtime_inner = (double) (1000.0*(wstop_inner-wstart_inner));

    /* Transpose result from cache-blocked to original form. */
    {       
        int rank = 6;
        long below[6];
        const int dim[] = {
            size_inner, size_inner, size_mid, size_mid, size_outer, size_outer 
        };
        const int perm[] = { 0, 2, 4, 1, 3, 5 };
        tp(len, rank, dim, perm, below, result_a, result);
    }

    free(result_a);
    free(left_a);
    free(right_a);  
   
    double wstop = omp_get_wtime();
    double flops = ((double) size)*size*size*2.0 - ((double) size)*size;
    double wtime = (double) (1000.0*(wstop-wstart));

    fprintf(stdout, 
            "n = %d\n"
            "wtime       = %6.0f ms,  "
            "perf       = %7.2f Gflop/s [ %6.2f Gflop/s/core]\n"
            "wtime_inner = %6.0f ms,  "
            "perf_inner = %7.2f Gflop/s [ %6.2f Gflop/s/core]\n",
            size,
            wtime,
            flops/(wtime*1e6),
            flops/(wtime*1e6)/omp_get_max_threads(),
            wtime_inner,
            flops/(wtime_inner*1e6),
            flops/(wtime_inner*1e6)/omp_get_max_threads());
    fflush(stdout);
}

/****************************************************** Program entry point. */

int main(int argc, char **argv) 
{
    for(int n = 3*128; n <= 16*3*128; n *= 2) {
        double *left   = (double *) malloc(sizeof(double)*n*n);
        double *right  = (double *) malloc(sizeof(double)*n*n);
        double *result = (double *) malloc(sizeof(double)*n*n);
        int al = 1;
        int ar = 2;
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < n; j++) {
                left[i*n + j]  = i == (j+al)%n ? 1.0 : 0.0;
                right[i*n + j] = i == (j+ar)%n ? 1.0 : 0.0;
            }
        }
        mm(n, result, left, right);
        for(int i = 0; i < n; i++)
            for(int j = 0; j < n; j++)
                assert(result[i*n + j] == (i == (j+al+ar)%n ? 1.0 : 0.0));
        free(result);
        free(right);
        free(left);
    }
    return 0;
}
