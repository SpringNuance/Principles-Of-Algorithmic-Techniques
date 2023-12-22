/******************************************* A simple memory bandwidth test. */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <string.h>
#include <sys/utsname.h>
#include <omp.h>

/****** A quick-and-dirty fast-forward pseudorandom number generator (PRNG). */

/* 
 * Idea: XOR two LFSRs, one clocked from left to right, 
 *                      the other from right to left. 
 */

/*
 * The primitive polynomial used to construct GF(2^{64})
 *
 *   :64  x^{64}+                   -
 * 63:60  x^{62}+x^{60}+            5
 * 59:56  x^{58}+x^{56}+            5
 * 55:52  x^{55}+x^{53}+x^{52}+     B
 * 51:48  x^{51}+x^{50}+x^{48}+     D
 * 47:44  x^{46}+x^{45}+x^{44}+     7
 * 43:40  x^{43}+x^{41}+x^{40}+     B
 * 39:36  x^{39}+                   8
 * 35:32  x^{35}+x^{34}+            C
 * 31:28  x^{31}+x^{30}+x^{29}+     E
 * 27:24  x^{27}+x^{25}+x^{24}+     B
 * 23:20  x^{22}+x^{21}+x^{20}+     7
 * 19:16  x^{19}+x^{17}+x^{16}+     B
 * 15:12  x^{15}+x^{12}+            9
 * 11:8   x^{11}+x^{9}+             A
 * 7:4    x^{7}+x^{6}+x^{5}+        E
 * 3:0    x^{3}+1                   9
 *
 * 0x55BD7B8CEB7B9AE9UL
 *
 * http://fchabaud.free.fr/English/default.php?COUNT=3&FILE0=Poly&FILE1=GF(2)&FILE2=Primitive
 *
 */

/*
    long int z = 0x55BD7B8CEB7B9AE9UL;
    long int zr = 0;
    for(long int j = 63; j >= 0; j--) {
        if((z >> j)&1) {
            fprintf(stdout, "%ld ", j);
            zr |= 1UL << (63-j);
        }
        fprintf(stdout, "%s", j % 4 == 0 ? "\n" : "");
    }
    fprintf(stdout, "0x%016lX\n", zr);
*/

#define BADPRNG_MODULUS         0x55BD7B8CEB7B9AE9UL

typedef unsigned long int badprng_scalar_t;

#define BADPRNG_SCALAR_MSB      0x8000000000000000L

typedef struct 
{
    badprng_scalar_t   state;
} badprng_t;

#define BADPRNG_INIT(ctx, origin)\
{                                                       \
    badprng_scalar_t t = (origin);                      \
    ctx.state = t^0x1234567890ABCDEFL;                  \
}

#define BADPRNG_RAND(out, ctx)\
{                                                                      \
    badprng_scalar_t tl = ctx.state;                                   \
    out = tl;                                                          \
    badprng_scalar_t fl = tl & 0x8000000000000000L;                    \
    tl = (tl << 1)^(((((signed long) fl) >> 63))&BADPRNG_MODULUS);     \
    ctx.state = tl;                                                    \
}

#define BADPRNG_MUL(target, left, right)\
    target = badprng_gf2_64_mul(left, right);

badprng_scalar_t badprng_gf2_64_mul(badprng_scalar_t x, badprng_scalar_t y)
{   
    badprng_scalar_t z = 0;
    for(int i = 0; i < 64; i++) {
        badprng_scalar_t f = x & BADPRNG_SCALAR_MSB;
        if(y & 1)
            z ^= x;
        y >>= 1;
        x <<= 1;
        if(f)
            x ^= BADPRNG_MODULUS;
    }
    return z;
}

badprng_scalar_t badprng_gf2pow(badprng_scalar_t g, long amount) 
{
    assert(amount >= 0);
    if(amount == 0)
        return 1;
    if(amount == 1)
        return g;
    badprng_scalar_t gg;
    BADPRNG_MUL(gg, g, g);
    if(amount & 1) {
        badprng_scalar_t t;
        t = badprng_gf2pow(gg, amount/2);
        badprng_scalar_t tt;
        BADPRNG_MUL(tt, g, t);
        return tt;
    } else {
        return badprng_gf2pow(gg, amount/2);
    }
}

#define BADPRNG_FWD(ctx, amount, base)                   \
{                                                        \
    badprng_scalar_t tt = badprng_gf2pow(0x02L, amount); \
    badprng_scalar_t tl = base.state;                    \
    badprng_scalar_t ttl;                                \
    BADPRNG_MUL(ttl, tt, tl);                            \
    ctx.state = ttl;                                     \
}


/************************************************** Some helper subroutines. */

#define TIME_STACK_CAPACITY 256

double wstart_stack[TIME_STACK_CAPACITY];
long wstart_stack_top = -1;

void push_wtime(void) 
{
    assert(wstart_stack_top + 1 < TIME_STACK_CAPACITY);
    wstart_stack[++wstart_stack_top] = omp_get_wtime();
}

double pop_wtime(void)
{
    double wstop = omp_get_wtime();
    assert(wstart_stack_top >= 0);
    double wstart = wstart_stack[wstart_stack_top--];
    return wstop-wstart;
}

double in_ms(double s)
{
    return s*1000.0;
}

double in_GiB(double bytes)
{
    return bytes / (double) (1 << 30);
}

int report(double wtime, double trans_bytes)
{
    FILE *out = stdout;
    static int par = 0;
    fprintf(out,
            "%8s %8.1f ms [%6.2lf GiB/s]%s",
            par % 2 != 0 ? "parallel" : "serial",
            in_ms(wtime),
            in_GiB(trans_bytes / wtime),
            par % 2 != 0 ? "\n" : "         ");
    fflush(out);
    par++;
    return 0;
}

/**************************************************************** The tests. */

#define NPAR 512
#define WORDS_IN_LINE 8

void perf_test_band()
{
    long n = (1UL << 25);
    int repeats = 3; 
      // actually runs one more repeat, the first repeat is unreported warmup
    
    double wtime;
    size_t array_size = sizeof(long)*n;  
    double trans_bytes;
    long *x = (long *) malloc(array_size);
    long *y = (long *) malloc(array_size);

    fprintf(stdout, "-- mem perf test [one word is %ld bytes (%ld bits), one line is %ld words (%ld bytes, %ld bits)]\n",
            (long) array_size/n,
            (long) 8*array_size/n,
            (long) WORDS_IN_LINE,
            (long) WORDS_IN_LINE*(array_size/n),
            (long) 8*WORDS_IN_LINE*(array_size/n));
    push_wtime();
            
    fprintf(stdout, 
            "-- linear word copy test (%4.2lf GiB)\n",
            in_GiB(array_size));
    trans_bytes = 2.0 * array_size;
    for(int r = 0; r <= repeats; r++) {
#pragma omp parallel for
        for(long i = 0; i < n; i++)
            x[i] = 123;

        push_wtime();
        for(long i = 0; i < n; i++)
            y[i] = x[i];    
        wtime = pop_wtime();
        for(long i = 0; i < n; i++)
            assert(y[i] == 123);
        r == 0 || report(wtime, trans_bytes);

#pragma omp parallel for
        for(long i = 0; i < n; i++)
            y[i] = 0;    

        push_wtime();
#pragma omp parallel for
        for(long i = 0; i < n; i++)
            y[i] = x[i];    
        wtime = pop_wtime();
        for(long i = 0; i < n; i++)
            assert(y[i] == 123);
        r == 0 || report(wtime, trans_bytes);
    }

    fprintf(stdout, 
            "-- linear word write test (%4.2lf GiB)\n",
            in_GiB(array_size));
    trans_bytes = array_size;
    for(int r = 0; r <= repeats; r++) {
        push_wtime();
        for(long i = 0; i < n; i++)
            x[i] = 2*i;    
        wtime = pop_wtime();
        for(long i = 0; i < n; i++)
            assert(x[i] == 2*i);
        r == 0 || report(wtime, trans_bytes);

        push_wtime();
#pragma omp parallel for
        for(long i = 0; i < n; i++)
            x[i] = 2*i;    
        wtime = pop_wtime();
        for(long i = 0; i < n; i++)
            assert(x[i] == 2*i);
        r == 0 || report(wtime, trans_bytes);
    }


    fprintf(stdout, 
            "-- linear word read test (%4.2lf GiB)\n", 
            in_GiB(array_size));
    trans_bytes = array_size;
    for(int r = 0; r <= repeats; r++) {
        push_wtime();
        long sum = 0;
        for(long i = 0; i < n; i++)
            sum += x[i];    
        wtime = pop_wtime();
        assert(sum == n*(n-1));
        r == 0 || report(wtime, trans_bytes);

        push_wtime();
        long sums[NPAR];
#pragma omp parallel for
        for(long t = 0; t < NPAR; t++) {
            sums[t] = 0;
            long start = (n/NPAR)*t;
            long stop = (n/NPAR)*(t+1)-1;
            long tsum = 0;
            for(long i = start; i <= stop; i++)
                tsum += x[i];
            sums[t] = tsum;
        }
        sum = 0;
        for(long t = 0; t < NPAR; t++)
            sum += sums[t];    
        wtime = pop_wtime();
        assert(sum == n*(n-1));
        r == 0 || report(wtime, trans_bytes);
    }

    fprintf(stdout, 
            "-- random word read test (%4.2lf GiB)\n", 
            in_GiB(array_size));
    trans_bytes = array_size;
    for(int r = 0; r <= repeats; r++) {
        badprng_t base, run;
        BADPRNG_INIT(base, 12345678901234567L);
        BADPRNG_FWD(run, 0, base);

        push_wtime();
        long sum = 0;
        long n_mask = n-1;
        for(long i = 0; i < n; i++) {
            badprng_scalar_t r;
            BADPRNG_RAND(r, run);
            long ri = r & n_mask;
            sum += x[ri];    
        }
        wtime = pop_wtime();
        r == 0 || report(wtime, trans_bytes);

        push_wtime();
        long psum = 0;
        long sums[NPAR];
#pragma omp parallel for
        for(long t = 0; t < NPAR; t++) {
            sums[t] = 0;
            long start = (n/NPAR)*t;
            long stop = (n/NPAR)*(t+1)-1;
            badprng_t fwd;
            BADPRNG_FWD(fwd, start, base);
            long tsum = 91828;
            long n_mask = n-1;
            for(long i = start; i <= stop; i++) {
                badprng_scalar_t r;
                BADPRNG_RAND(r, fwd);
                long ri = r & n_mask;
                tsum += x[ri];
            }
            sums[t] = tsum;
        }
        for(long t = 0; t < NPAR; t++)
            psum += sums[t];    
        wtime = pop_wtime();
        assert(sum == psum - NPAR*91828);
        r == 0 || report(wtime, trans_bytes);
    }


    fprintf(stdout, 
            "-- linear line read test [with prng overhead] (%4.2lf GiB)\n", 
            in_GiB(array_size));
    trans_bytes = array_size;
    long nl = n / WORDS_IN_LINE;
    for(int r = 0; r <= repeats; r++) {
        badprng_t base, run;
        BADPRNG_INIT(base, 12345678901234567L);
        BADPRNG_FWD(run, 0, base);

        push_wtime();
        long sum = 0;
        long nl_mask = nl-1;
        for(long i = 0; i < nl; i++) {
            badprng_scalar_t r;
            BADPRNG_RAND(r, run);
            long ii = i*WORDS_IN_LINE;
            long ri = (r & nl_mask)*WORDS_IN_LINE;
            for(long j = 0; j < WORDS_IN_LINE; j++)
                sum += x[ii+j]+ri;    
        }
        wtime = pop_wtime();
        r == 0 || report(wtime, trans_bytes);

        push_wtime();
        long psum = 0;
        long sums[NPAR];
#pragma omp parallel for
        for(long t = 0; t < NPAR; t++) {
            sums[t] = 0;
            long start = (nl/NPAR)*t;
            long stop = (nl/NPAR)*(t+1)-1;
            badprng_t fwd;
            BADPRNG_FWD(fwd, start, base);
            long tsum = 42557;
            long nl_mask = nl-1;
            for(long i = start; i <= stop; i++) {
                badprng_scalar_t r;
                BADPRNG_RAND(r, fwd);
                long ii = i*WORDS_IN_LINE;
                long ri = (r & nl_mask)*WORDS_IN_LINE;
                for(long j = 0; j < WORDS_IN_LINE; j++)
                    tsum += x[ii+j]+ri;    
            }
            sums[t] = tsum;
        }
        for(long t = 0; t < NPAR; t++)
            psum += sums[t];    
        wtime = pop_wtime();
        assert(sum == psum - NPAR*42557);
        r == 0 || report(wtime, trans_bytes);
    }


    fprintf(stdout, 
            "-- random line read test [line forward] (%4.2lf GiB)\n", 
            in_GiB(array_size));
    trans_bytes = array_size;
    nl = n / WORDS_IN_LINE;
    for(int r = 0; r <= repeats; r++) {
        badprng_t base, run;
        BADPRNG_INIT(base, 12345678901234567L);
        BADPRNG_FWD(run, 0, base);

        push_wtime();
        long sum = 0;
        long nl_mask = nl-1;
        for(long i = 0; i < nl; i++) {
            badprng_scalar_t r;
            BADPRNG_RAND(r, run);
            long ii = i*WORDS_IN_LINE;
            long ri = (r & nl_mask)*WORDS_IN_LINE;
            for(long j = 0; j < WORDS_IN_LINE; j++)
                sum += x[ri+j]+ii;    
        }
        wtime = pop_wtime();
        r == 0 || report(wtime, trans_bytes);

        push_wtime();
        long psum = 0;
        long sums[NPAR];
#pragma omp parallel for
        for(long t = 0; t < NPAR; t++) {
            sums[t] = 0;
            long start = (nl/NPAR)*t;
            long stop = (nl/NPAR)*(t+1)-1;
            badprng_t fwd;
            BADPRNG_FWD(fwd, start, base);
            long tsum = 36754;
            long nl_mask = nl-1;
            for(long i = start; i <= stop; i++) {
                badprng_scalar_t r;
                BADPRNG_RAND(r, fwd);
                long ii = i*WORDS_IN_LINE;
                long ri = (r & nl_mask)*WORDS_IN_LINE;
                for(long j = 0; j < WORDS_IN_LINE; j++)
                    tsum += x[ri+j]+ii;    
            }
            sums[t] = tsum;
        }
        for(long t = 0; t < NPAR; t++)
            psum += sums[t];    
        wtime = pop_wtime();
        assert(sum == psum - NPAR*36754);
        r == 0 || report(wtime, trans_bytes);
    }

    
    fprintf(stdout, 
            "-- random line read test [line backward] (%4.2lf GiB)\n", 
            in_GiB(array_size));
    trans_bytes = array_size;
    nl = n / WORDS_IN_LINE;
    for(int r = 0; r <= repeats; r++) {
        badprng_t base, run;
        BADPRNG_INIT(base, 12345678901234567L);
        BADPRNG_FWD(run, 0, base);

        push_wtime();
        long sum = 0;
        long nl_mask = nl-1;
        for(long i = 0; i < nl; i++) {
            badprng_scalar_t r;
            BADPRNG_RAND(r, run);
            long ri = (r & nl_mask)*WORDS_IN_LINE;
            long ii = i*WORDS_IN_LINE;
            for(long j = WORDS_IN_LINE-1; j >= 0; j--)
                sum += x[ri+j]+ii;    
        }
        wtime = pop_wtime();
        r == 0 || report(wtime, trans_bytes);

        push_wtime();
        long psum = 0;
        long sums[NPAR];
#pragma omp parallel for
        for(long t = 0; t < NPAR; t++) {
            sums[t] = 0;
            long start = (nl/NPAR)*t;
            long stop = (nl/NPAR)*(t+1)-1;
            badprng_t fwd;
            BADPRNG_FWD(fwd, start, base);
            long tsum = 0;
            long nl_mask = nl-1;
            for(long i = start; i <= stop; i++) {
                badprng_scalar_t r;
                BADPRNG_RAND(r, fwd);
                long ii = i*WORDS_IN_LINE;
                long ri = (r & nl_mask)*WORDS_IN_LINE;
                for(long j = WORDS_IN_LINE-1; j >= 0; j--)
                    tsum += x[ri+j]+ii;    
            }
            sums[t] = tsum;
        }
        for(long t = 0; t < NPAR; t++)
            psum += sums[t];    
        wtime = pop_wtime();
        assert(sum == psum);
        r == 0 || report(wtime, trans_bytes);
    }

    wtime = pop_wtime();
    fprintf(stdout,
            "-- mem perf test done [total wall-clock time %.1lf ms]\n",
            in_ms(wtime));

    free(x);
    free(y);
}

/****************************************************** Program entry point. */

int main(int argc, char **argv)
{
    perf_test_band();
    return 0;
}

