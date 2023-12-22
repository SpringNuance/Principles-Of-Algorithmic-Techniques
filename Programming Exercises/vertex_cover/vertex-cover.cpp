
/*  Auto-built C++ source file 'vertex-cover.cpp'.  */

/******************************************************************************

  Aalto University
  CS-E3190 Principles of Algorithmic Techniques
  Autumn 2021

  Exercise: Vertex cover

  Description:

  This exercise asks you to implement an algorithm that computes a low-cost
  vertex cover. More precisely, the input consists of an $n$-vertex, $m$-edge
  undirected graph with a positive integer cost associated to each vertex.
  A set of vertices is called a *vertex cover* if each edge has at least one
  end-vertex in the set. The *cost* of a vertex cover is the sum of the costs
  of its vertices. The *optimum cost* $\mathrm{OPT}$ of a vertex cover is the
  minimum cost of a vertex cover, where the minimum is taken over all possible
  vertex covers. The algorithm must output a vertex cover $S$ with cost at
  most $2\cdot\mathrm{OPT}$. For example, an implementation of the primal-dual
  vertex cover algorithm suffices for this purpose.
  
  **Your task** in this exercise is to complete the subroutine
                     
  ``void solver(int n, int m, const int *e, const int *c, int &k, int *s)``
  
  which should compute the size `k` and the elements `s` of a vertex cover as
  described in the previous paragraph from the given input consisting of
  positive integers `n` and `m`, as well as the arrays `c` and `e`, whose
  format is as follows.
  
  The array `e` concatenates the $m$ edges in the input graph.
  That is, the $m$ edges of the graph are
  $\{e[0],e[1]\},\{e[2],e[3]\},\ldots,\{e[2m-2],e[2m-1]\}$, with
  $\{e[2i],e[2i+1]\}\subseteq\{0,1,\ldots,n-1\}$ and $e[2i]<e[2i+1]$
  for all $i=0,1,\ldots,m-1$. The array `c` contains the cost $c[i]\geq 1$
  of each vertex $i=0,1,\ldots,n-1$.
                             
  The output of the subroutine should be as follows.
  To give as output a vertex cover $S=\{j_0,j_1,\ldots,j_{k-1}\}$,
  set `k` equal to $k$ and the element `s[i]` equal to $j_i$ for all
  $i=0,1,\ldots,k-1$. 
  You may assume that $2\leq n\leq 524288$, $1\leq m\leq 52428800$,
  $1\leq k\leq n$, and that the array $s$ has capacity for at least $n$
  elements. To locate the subroutine quickly, you can search for "`???`" in
  the source file.
  
  *Grading*. This exercise awards you up to 10 points in 
  the course grading. The number of points awarded is the maximum points times
  the number of tests passed over the total number of tests, rounded up. To
  successfully complete a test, your implementation must use no more than
  7 seconds of wall clock time and 512 MiB of memory. Each test
  will in general require the successful solution of one or more problem
  instances. In each batch of scaling tests, the first failed test will
  cause all subsequent tests in the batch to be skipped.

******************************************************************************/

#include <iostream>
#include <iomanip>
#include <sstream>
#include <memory>
#include <new>
#include <algorithm>
#include <stack>
#include <map>
#include <csignal>
#include <unistd.h>
#include <fcntl.h>
#include <sys/wait.h>
#include <random>
#include <getopt.h>
#include <omp.h>
#include <immintrin.h>
#include <utility>
#include <vector>

using namespace std;
/*************************************************** Simple error reporting. */

#define SDIE(s, msgcat) (errors(__FILE__,__LINE__) << msgcat << std::endl,\
                         exit(s),\
                         always_true())

#define DIE(msgcat) SDIE(EXIT_FAILURE, msgcat)

bool always_true(void)
{
    return true;
}

std::ostream &errors(const char *fn, int line) {
    std::cerr << "error [" << getpid() << ", " << fn
              << ", line " << line << "]: ";
    return std::cerr;
}

/********************************* Helper subroutines for resource tracking. */

typedef long long tick_t; // Introduce a global type for clock ticks. 

#define GIB_BYTES       ((size_t) (1L << 30))
#define MIB_BYTES       ((size_t) (1L << 20))
#define KIB_BYTES       ((size_t) (1L << 10))
#define TICKS_IN_SEC   ((tick_t)  1000000000)
#define TICKS_IN_MSEC   ((tick_t)    1000000)
#define TICKS_IN_MUSEC   ((tick_t)      1000)

namespace track {

    /* Custom allocator that bypasses the tracking interface. */
    template <class T>
    class AllocNoTrack
    {
    public:
        typedef size_t    size_type;
        typedef ptrdiff_t difference_type;
        typedef T *       pointer;
        typedef const T * const_pointer;
        typedef T &       reference;
        typedef const T & const_reference;
        typedef T         value_type;
        
        AllocNoTrack() {}
        AllocNoTrack(const AllocNoTrack&) {}
        
        T *allocate(size_t n, const void * = 0) {
            T *p = (T *) malloc(n*sizeof(T)); // resort to C malloc
            p != (T *) 0 || SDIE(123, "malloc failed");
            return p;
        }
        
        void deallocate(void *p, size_t) {
            if(p != (void *) 0)
                free(p); // resort to C free
        }
        
        T *address(T &x) const { return &x; }
        const T *address(const T &x) const { return &x; }
        AllocNoTrack<T> &operator=(const AllocNoTrack &) { return *this; }
        void construct(T *p, const T &val) { new ((T *) p) T(val); }
        void destroy(T *p) { p->~T(); }
        size_t max_size() const { return size_t(-1); }
    
        template <class U>
        struct rebind { typedef AllocNoTrack<U> other; };
        
        template <class U>
        AllocNoTrack(const AllocNoTrack<U> &) {}
    
        template <class U>
        AllocNoTrack &operator=(const AllocNoTrack<U> &) { return *this; }
    };

    /* Tracked memory allocation. */

    size_t alloc_current = 0;

    std::map<void *,
             size_t,
             std::less<void *>,
             AllocNoTrack<std::pair<void * const,size_t>>> alloc_map;

    std::stack<size_t,std::deque<size_t,AllocNoTrack<size_t>>> mem_stack;

    void *new_tracked(size_t size, std::align_val_t alignment)
    {
        void *p;
        posix_memalign(&p, (size_t) alignment, size) == 0 ||
            SDIE(123, "posix_memalign fails");
        alloc_map.erase(p) == 0 || DIE("duplicate pointer"); 
        alloc_map[p] = size;    
        alloc_current += size;
        if(mem_stack.size() > 0 && alloc_current > mem_stack.top())
            mem_stack.top() = alloc_current;
        return p;
    }

    void delete_tracked(void *p)
    {
        try {       
            size_t size = alloc_map.at(p);
            free(p);    
            alloc_current -= size;
            alloc_map.erase(p) == 1 || DIE("erase failed");
        } catch(const std::out_of_range &oor) {
            DIE("untracked pointer");
        }
    }

    /* Timing. */

    std::stack<tick_t,std::deque<tick_t,AllocNoTrack<tick_t>>> time_stack;
    
    /* Timing and sleep subroutines. */

    tick_t now(void)
    {
        tick_t t;
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts) == 0 || DIE("clock_gettime failed");
        t = (tick_t) ts.tv_sec * (tick_t) TICKS_IN_SEC + (tick_t) ts.tv_nsec;
        return t;   
    }

    void short_sleep(void)
    {
        struct timespec rgtp;
        rgtp.tv_sec  = 0;
        rgtp.tv_nsec = 10*TICKS_IN_MUSEC; // ten microsecond sleep
        nanosleep(&rgtp, (struct timespec *) 0);
    }

    /* Stack for tracking elapsed wall-clock time & peak memory usage. */

    void start(void)
    {
        time_stack.push(now());
        mem_stack.push(alloc_current);
    }
    
    void stop(long &ms, long &KiB)
    {
        time_stack.size() > 0 || DIE("pop on an empty stack");
        time_t current = now();
        time_t start   = time_stack.top();
        time_stack.pop();
        tick_t elapsed = current - start;       
        mem_stack.size() > 0 || DIE("pop on an empty stack");
        size_t peak = mem_stack.top();
        mem_stack.pop();
        if(mem_stack.size() > 0 && peak > mem_stack.top())
            mem_stack.top() = peak;
        ms  = (elapsed + TICKS_IN_MSEC-1)/TICKS_IN_MSEC;
        KiB = (peak + KIB_BYTES-1)/KIB_BYTES;
    }

    /* Check balances at exit. */

    void exit_balances(void)
    {
        if(alloc_map.size() != 0)
            std::cerr << getpid()
                      << ": warning -- memory leak detected at exit"
                      << std::endl;
        if(time_stack.size() != 0)
            std::cerr << getpid()
                      << ": warning -- started tracking not stopped at exit"
                      << std::endl;
    }    
}

/* Overload the global operators "new" and "delete" with tracked allocation. */

void *operator new(std::size_t size) {
    return track::new_tracked(size, (std::align_val_t) sizeof(void *));
}

void *operator new(std::size_t size, std::align_val_t alignment) {
    return track::new_tracked(size, alignment);
}

void operator delete(void *p) noexcept
{
    return track::delete_tracked(p);
}

void operator delete(void *p, std::size_t size) noexcept
{
    return track::delete_tracked(p);
}

void operator delete(void *p, std::align_val_t alignment) noexcept
{
    return track::delete_tracked(p);
}

void operator delete(void *p,
                     std::size_t size,
                     std::align_val_t alignment) noexcept
{
    return track::delete_tracked(p);
}

/**************************************************** Solver test interface. */

/* Forward declarations. */

class TestContext;

void solver(int n, int m, const int *e, const int *c, int &k, int *s);

void pipe_out(TestContext &ctx,
              int n, int m, const int *e, const int *c, int &k, int *s);

/* Testing context class. */

extern char **environ;  /* The environment for execve(). */

class TestContext
{
    bool                 good;
    bool                 boxed;
    std::ostringstream   reason;
    std::ostringstream   diag;
    bool                 started;
    bool                 ended;
    bool                 timeout;
    tick_t               stop;
    tick_t               atstart;
    tick_t               time_quota;
    tick_t               time_solve;
    long                 timingms;
    int                  in;
    int                  out;
    pid_t                pid;
    long                 num_solved;
public:
    static const char *boxed_binary;
    
    TestContext() {
        /* Configure for [boxed] non-timeouting piped I/O via stdin/stdout. */
        good       = true;
        boxed      = false;
        timeout    = false;
        started    = false;
        ended      = false;
        in         = STDIN_FILENO;
        out        = STDOUT_FILENO;
        num_solved = 0;
    }
            
    ~TestContext() {
        !started || ended || DIE("test started but not ended");
    }

    void start(bool box, tick_t time_allocation) {
        !started || DIE("already started");
        started    = true;
        boxed      = box;
        timeout    = true;
        atstart    = track::now();
        time_quota = time_allocation;
        time_solve = 0;

        /* Hard timeout at two times the time quota. */
        stop = atstart + 2*time_quota;

        if(boxed) {        
            /* Set up piping. */        
            int rw[2];
            pipe(rw) == 0 || DIE("pipe failed");
            int in_read  = rw[0];
            int in_write = rw[1];
            pipe(rw) == 0 || DIE("pipe failed");
            int out_read  = rw[0];
            int out_write = rw[1];

            /* Configure for non-blocking I/O on parent. */            
            fcntl(in_write, F_SETFL, O_NONBLOCK) != -1 || DIE("fcntl failed");
            fcntl(out_read, F_SETFL, O_NONBLOCK) != -1 || DIE("fcntl failed");
            
            /* Configure to ignore SIGPIPE. */            
            struct sigaction act;
            act.sa_handler = SIG_IGN;
            act.sa_flags   = 0;    
            sigprocmask(0, (const sigset_t *) 0, &act.sa_mask) == 0 ||
                DIE("sigprocmask failed");
            sigaction(SIGPIPE, &act, (struct sigaction *) 0) == 0 ||
                DIE("sigaction failed");

            /* Do fork. */            
            pid = fork();
            pid != -1 || DIE("fork failed");

            if(pid == 0) {            
                /* The child process. */
                
                /* Close the parent-side of the interface. */                
                close(in_write) == 0 || DIE("close failed");
                close(out_read) == 0 || DIE("close failed");
                
                /* Reconfigure standard input and output. */                
                close(STDIN_FILENO)  == 0 || DIE("close failed");
                close(STDOUT_FILENO) == 0 || DIE("close failed");
                dup2(in_read, STDIN_FILENO) == STDIN_FILENO ||
                    DIE("dup2 failed");
                dup2(out_write, STDOUT_FILENO) == STDOUT_FILENO ||
                    DIE("dup2 failed");
                close(in_read)  == 0  || DIE("close failed");
                close(out_write) == 0 || DIE("close failed");

                /* Execute a separate binary. */                
                const char *argv[] = { boxed_binary, "--piped", NULL };
                execve(boxed_binary, (char *const*) argv, environ);
                DIE("execve failed");            
                /* Execution never continues through here. */            
            } else {                
                /* The parent process. */
        
                /* Close the child-side of the interface. */            
                close(in_read)   == 0 || DIE("close failed");
                close(out_write) == 0 || DIE("close failed");

                /* Record the input-output file descriptors. */
                out = in_write;
                in  = out_read;

                /* Wait for ready signal from the child. */
                unsigned int ready_signal;
                recv(&ready_signal, sizeof(unsigned int), "ready");
                if(good && ready_signal != 0x0fe95617u) {
                    reason << "bad ready signal from child";
                    good = false;
                }
            }
        }
    }

    bool end(void) {            
        started || DIE("not started");
        !ended  || DIE("already ended");
        if(boxed) {
            if(good) {
                /* Request exit from child. */                
                int n = -1;
                send(&n, sizeof(int), "stop");            
            } else {
                /* Terminate. */            
                kill(pid, SIGKILL) == 0 || DIE("kill failed");
            }        

            /* Wait for termination. */            
            int status;
            bool good_exit = false;
            bool wait_done = false;
            bool fail_before_wait = !good;
            while(!wait_done) {
                pid_t rv = waitpid(pid, &status, WNOHANG);
                tick_t current = track::now();
                if(rv == 0) {
                    if(current > stop) {
                        if(!fail_before_wait)
                            reason << "timeout while waiting for child exit";
                        fail_before_wait = true;
                        kill(pid, SIGKILL) == 0 || DIE("kill failed");
                    } else {
                        track::short_sleep();
                    }
                } else {
                    rv == pid || DIE("waitpid failed");
                    wait_done = true;
                    if(!fail_before_wait) {                 
                        if(WIFEXITED(status)) {
                            int exit_status = WEXITSTATUS(status);
                            if(exit_status == 0)
                                good_exit = true;
                            else
                                reason << "child exit with status "
                                       << exit_status;
                        } else {
                            if(WIFSIGNALED(status)) {
                                good_exit = false;
                                int term_sig = WTERMSIG(status);
                                reason << "child terminated by signal "
                                       << term_sig;
                            } else {
                                DIE("unknown waitpid status");
                            }
                        }
                    }
                }
            }
            good = good_exit;                

            /* Close interface to child. */            
            close(in)  == 0 || DIE("close failed");
            close(out) == 0 || DIE("close failed");            
        }
        if(time_solve > time_quota) {
            reason.str(""); reason.clear();
            reason << "time limit exceeded";
            good = false;            
        }
        timingms = (time_solve+TICKS_IN_MSEC-1)/TICKS_IN_MSEC;
        ended = true;
        return good;
    }

    long timing_ms(void) {
        ended || DIE("not ended");
        return timingms;
    }

    void send(const void *data, size_t size, const char *id) {
        if(!good)
            return;
        !ended || DIE("already ended");
        tick_t current = track::now();
        while(size > 0) {
            ssize_t amount = write(out, data, size);
            current = track::now();
            if(amount < 0) {
                switch(errno) {
                case EAGAIN:
                    track::short_sleep();
                    break;
                default:
                    good = false;
                    reason << id << " write error";
                    return;
                }
            } else {
                if(amount == 0) {
                    good = false;
                    reason << id << " write -- no bytes written";
                    return;
                }
                size -= amount;
                data = ((const char *) data) + amount;
            }
            if(timeout && current > stop)
                break;
        }
        if(timeout && current > stop) {
            good = false;
            reason << "time limit exceeded";
            return;
        }
    }

    void recv(void *data, size_t size, const char *id) {
        if(!good)
            return;
        !ended || DIE("already ended");
        tick_t current = track::now();
        while(size > 0) {
            ssize_t amount = read(in, data, size);
            current = track::now();
            if(amount < 0) {
                switch(errno) {
                case EAGAIN:
                    track::short_sleep();
                    break;
                default:
                    good = false;
                    reason << id << " read error";
                    return;
                }
            } else {
                if(amount == 0) {
                    good = false;
                    reason << id << " read -- unexpected end of file";
                    return;
                }
                size -= amount;
                data = ((char *) data) + amount;
            }
            if(timeout && current > stop)
                break;
        }
        if(timeout && current > stop) {
            good = false;
            reason << "time limit exceeded";
            return;
        }
        return;
    }

    void guard(void) {
        if(!good) {
            std::cerr << reason.str() << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    bool solve(int n, int m, const int *e, const int *c, int &k, int *s) {
        started || DIE("not started");
        !ended  || DIE("already ended");
        good    || DIE("solve called on bad context");
        tick_t solve_start = track::now();
        if(!boxed) {
            /* Solve locally without boxing. */
            solver(n, m, e, c, k, s);        
        } else {
            /* Pipe to solver process. */            
            pipe_out(*this, n, m, e, c, k, s);
        }
        tick_t solve_end = track::now();
        time_solve += solve_end - solve_start;
        num_solved++;
        return good;
    }

    long count(void) const {
        return num_solved;
    }   

    bool bad(void) const {
        return !good;
    }

    std::ostringstream &fail(void) {
        good = false;
        return reason;
    }

    std::ostringstream &diagnostics(void) {
        !good || DIE("diagnostics called on good context");
        return diag;
    }
};

const char *TestContext::boxed_binary = NULL;

/************************ Solver and select helper subroutines + unit tests. */

bool check_instance(int n, int m, const int *e, const int *c)
{
    if(n <= 1 || m <= 0)
        return false;
    for(int j = 0; j < m; j++)
        if(e[2*j] >= e[2*j+1])
            return false;
    for(int i = 0; i < n; i++)
        if(c[i] <= 0)
            return false;
    return true;
}

void solver(int n, int m, const int *e, const int *c, int &k, int *s)
{
    // int *x = new int[n];
    // for(int i = 0; i < n; i++) {
    //     x[i] = 0;
    // }
    // int *y = new int[m];
    // for(int i = 0; i < m; i++) {
    //     y[i] = 0;
    // }
    // k = 0;
    // for(int i = 0; i < 2*m; i += 2) {
    //     if (x[e[i]] + x[e[i+1]] < 1) {
    //         int sum_of_y_u = 0;
    //         int sum_of_y_v = 0;
    //         // Check that whether updating y_u or y_v cost less, and choose the one that cost less
    //         for (int j = 0; j < 2*m; j += 2) {
    //             if (e[j] == e[i] || e[j+1] == e[i]) {
    //                 sum_of_y_u += y[j/2];
    //             }
    //             if(e[j] == e[i+1] || e[j+1] == e[i+1]) {
    //                 sum_of_y_v += y[j/2];
    //             }
    //         }
    //         if(c[e[i]] - sum_of_y_u < c[e[i+1]] - sum_of_y_v) {
    //             y[i/2] = c[e[i]];
    //             x[e[i]] = 1;
    //             s[k] = e[i];
    //             k++;
    //         } else {
    //             y[i/2] = c[e[i+1]];
    //             x[e[i+1]] = 1;
    //             s[k] = e[i+1];
    //             k++;
    //         }
    //     }
    // }
    // delete[] x;
    // delete[] y;
    // return;

    // JAVA implementation
	// int[] remainingWeights = Arrays.copyOf(weights, weights.length);
    int *remainingWeights = new int[n];
    for(int i = 0; i < n; i++) {
        remainingWeights[i] = c[i];
    }

    // ArrayList<String> vertexCoverNodes = new ArrayList<String>();
    //int totalWeight = 0;
    k = 0;
    for(int i = 0; i < m * 2; i += 2) {
        int fromVertex = e[i];
        int toVertex = e[i+1];
        if (remainingWeights[fromVertex] == 0 || remainingWeights[toVertex] == 0) {		//skip edges if either vertex is already tight
            continue;
        }

        if (remainingWeights[fromVertex] < remainingWeights[toVertex]){		//fromVertex weight is smaller
            int smallerWeight = remainingWeights[fromVertex];
            remainingWeights[fromVertex] = 0;	// 1 vertex becomes tight (greedy)
            remainingWeights[toVertex] -= smallerWeight;
            s[k] = fromVertex;
            k++;
        }
        else {		//toVertex weight is smaller or they're equal
            int smallerWeight = remainingWeights[toVertex];
            remainingWeights[toVertex] = 0;		//1 vertex becomes tight (greedy)
            remainingWeights[fromVertex] -= smallerWeight;
            s[k] = toVertex;
            k++;
        }
    }

    delete[] remainingWeights;
    return;
}

bool check_soln(int n, int m, const int *e, const int *c,                
                int k, const int *s, int optatmost)
{
    check_instance(n, m, e, c) || DIE("bad instance");
    if(k <= 0 || k > n)
        return false;
    bool good = true;
    int *b = new int[n];    
    for(int i = 0; i < n; i++)
        b[i] = false;
    int cost = 0;
    for(int i = 0; i < k; i++) {
        if(s[i] < 0 || s[i] >= n) {
            /* Vertex number out of bounds. */
            good = false;
            break;                  
        } else {
            if(b[s[i]]) {
                /* Repeat vertex. */
                good = false; 
                break;
            }
            b[s[i]] = true;
            cost += c[s[i]];
        }
    }
    for(int j = 0; j < m; j++) {
        if((!b[e[2*j]]) && (!b[e[2*j+1]])) {
            /* Edge not covered. */
            good = false;
            break;
        }
    }
    if(cost > 2*optatmost) {
        /* Not within approximation bound. */
        good = false;
    }
    delete[] b;
    return good;
}

void pipe_in(TestContext &ctx)
{
    while(true) {        
        /* Pipe-in solver loop. */

        /* Read preliminaries. */
        int n, m;
        ctx.recv(&n, sizeof(int), "preliminaries");
        ctx.guard();
        if(n < 0) 
            break; /* Stop. */
        ctx.recv(&m, sizeof(int), "preliminaries");
        ctx.guard();

        /* Allocate buffers. */
        int *e = new int [2*m];
        int *c = new int [n];
        int *s = new int [n];
        
        /* Read input. */
        ctx.recv(e, sizeof(int)*2*m, "e");
        ctx.guard();

        /* Read further input. */
        ctx.recv(c, sizeof(int)*n, "c");
        ctx.guard();
        
        /* Solve. */
        int k;
        solver(n, m, e, c, k, s);

        /* Write output. */
        ctx.send(&k, sizeof(int),  "k");
        ctx.guard();
        ctx.send(s, sizeof(int)*k, "s");
        ctx.guard();
        
        /* Release buffers. */
        delete[] s;
        delete[] c;
        delete[] e;
    }
}

void pipe_out(TestContext &ctx,
              int n, int m, const int *e, const int *c, int &k, int *s)
{
    /* Write preliminaries. */    
    ctx.send(&n, sizeof(int), "preliminaries");
    ctx.send(&m, sizeof(int), "preliminaries");
    
    /* Write input. */   
    ctx.send(e, sizeof(int)*2*m, "e");    
    ctx.send(c, sizeof(int)*n,   "c");
    
    /* Read solver output. */    
    ctx.recv(&k, sizeof(int),  "k");
    if(k < 0 || k > n)
        return;
    ctx.recv(s, sizeof(int)*k, "s");
}

void diagnostics(std::ostream &out,
                 int n, int m, const int *e, const int *c,
                 int k, const int *s,
                 int optatmost)
{
    check_instance(n, m, e, c) || DIE("bad input");
    out << std::endl;
    if(n <= 10 && m <= 20) {
        out << "  n = " << n << ", m = " << m << std::endl;
        for(int i = 0; i < m; i++) {
            out << "  { e[" << 2*i << "], e[" << 2*i+1 
                << "] } = {" << e[2*i] << ", " << e[2*i+1] << "}"
                << std::endl;
        }
        out << "  k = " << k << std::endl;
        if(k > 0) {
            int cost = 0;
            out << "  S = {";
            for(int j = 0; j < k && j < n; j++) {
                out << s[j];
                cost += c[s[j]];
                if(j < k-1 && j < n-1)
                    out << ", ";
                else
                    out << "}";
            }
            out << std::endl;
            out << "  cost      = " << cost << std::endl;
            out << "  optatmost = " << optatmost << std::endl;
        }
    } else {
        out << "  [instance with n = " << n << " and m = " << m << ", suppressing diagnostics]"
            << std::endl;
    }
    out << std::endl;
}

void trial_rand(TestContext &ctx, std::mt19937 &g,
                int n, int m, int k0, int cmax)
{
    (n > 0 && m > 0 && k0 > 0 && k0 <= n/2) || DIE("bad parameters");
    int *c = new int[n];
    int *e = new int[2*m];
    int *s = new int[n];
    bool *u = new bool[n];
    for(int i = 0; i < n; i++)
        u[i] = false;
    int k1 = 0;
    while(k1 < k0) {
        std::uniform_int_distribution<> d(0,n-1);   
        int i = d(g);
        if(!u[i]) {
            u[i] = true;
            k1++;
        }
    }
    int m1 = 0;
    while(m1 < m) {
        std::uniform_int_distribution<> d(0,n-1);   
        int i = d(g);
        int j = d(g);
        if(j < i) {
            int t = j;
            j = i;
            i = t;
        }
        if(i < j && ((!u[i]) || (!u[j]))) {
            e[2*m1]   = i;
            e[2*m1+1] = j;
            m1++;
        }           
    }
    int optatmost = 0;
    for(int i = 0; i < n; i++) {
        std::uniform_int_distribution<> d(1,cmax);
        c[i] = d(g);
        if(!u[i])
            optatmost += c[i];
    }

    check_instance(n, m, e, c) || DIE("bad instance");
    int k;
    if(ctx.solve(n, m, e, c, k, s)) {
        if(!check_soln(n, m, e, c, k, s, optatmost)) {
            ctx.fail() << "bad solution to instance " << ctx.count();
            diagnostics(ctx.diagnostics(), n, m, e, c, k, s, optatmost);
        }
    }

    delete[] u;
    delete[] s;
    delete[] e;
    delete[] c;
}

bool tests(bool do_boxed)
{
    std::mt19937 g(12345);

    track::start();
    bool good = true;
    for(int n = 2; n <= 30; n++) {
        int m = 100*n;
        int k0 = n/2;
        int cmax = 5;
        std::cerr << "test n = " << std::setw(3) << n
                  << ", m = " << std::setw(4) << m
                  << ", k0 = " << std::setw(3) << k0
                  << " ... ";
        TestContext ctx;
        ctx.start(do_boxed,
                  (tick_t) 7000*TICKS_IN_MSEC);
        int repeats = 1000;
        for(int r = 0; !ctx.bad() && r < repeats; r++)
            trial_rand(ctx, g, n, m, k0, cmax);
        ctx.end();
        if(ctx.bad()) {
            std::cerr << "FAIL [" << ctx.fail().str() << "]";
            good = false;
        } else {
            std::cerr << "OK   ["
                      << std::setw(5)
                      << ctx.timing_ms() << " ms]";
        }
        std::cerr << std::endl;
        if(ctx.bad())
            std::cerr << ctx.diagnostics().str();
    }
    for(int n = 1 << 10; n <= 1 << 19; n*=2) {
        int m = 100*n;
        int k0 = n/2;
        int cmax = 5;
        std::cerr << "scaling test n = " << std::setw(7) << n
                  << ", m = " << std::setw(9) << m
                  << ", k0 = " << std::setw(7) << k0
                  << " ... ";
        TestContext ctx;
        ctx.start(do_boxed,
                  (tick_t) 7000*TICKS_IN_MSEC);
        if(!ctx.bad())
            trial_rand(ctx, g, n, m, k0, cmax);
        ctx.end();
        if(ctx.bad()) {
            std::cerr << "FAIL [" << ctx.fail().str() << "]";
            good = false;
            if(ctx.bad())
                std::cerr << ctx.diagnostics().str();        
        } else {
            std::cerr << "OK   [";          
            std::cerr << std::setw(5) << ctx.timing_ms() << " ms";
            std::cerr << "]";
        }
        std::cerr << std::endl;
    }
    long ms, KiB;
    track::stop(ms, KiB);
    std::cerr << "tests done [" << ms << " ms, " << KiB << " KiB]"
              << std::endl;
    return good;
}

/****************************************************** Program entry point. */

int main(int argc, char * const *argv)
{
    bool run_unit_tests = true;   // Run the unit tests by default
    bool do_boxed       = false;  // Run the unit tests unboxed by default
    
    /* Parse the command line. */
    static struct option longopts[] = {
        { "piped", no_argument, NULL, 'p' },
        { "boxed", no_argument, NULL, 'b' }
    };
    int ch;
    while((ch = getopt_long(argc, argv, "pb", longopts, NULL)) != -1) {
        switch(ch) {
        case 'p':
            run_unit_tests = false;   // Run in piped-solver mode
            break;
        case 'b':
            do_boxed = true;          // Box the unit tests
            break;
        default:
            const char *ex =
                "Vertex cover";
            std::cout
              << "Template for exercise '"<<ex<<"', CS-E3190 Autumn 2021\n"
              << "\n"
              << "Usage: "<<argv[0]<<" [option]\n"
              << "\n"
              << "Options:\n"
              << "  -p   --piped    Run in piped-solver mode\n"
              << "  -b   --box      Box the unit tests\n"
              << std::endl;       
            return EXIT_FAILURE;
        }
    }

    /* Boxed tests execute self (in piped-solver mode). */
    TestContext::boxed_binary = argv[0];    

    /* Configure to check balances at exit to catch e.g. memory leaks. */
    atexit(&track::exit_balances) == 0 || DIE("atexit failed");

    /* Process and exit. */
    if(run_unit_tests) {
        /* Run the unit tests. */
        if(tests(do_boxed))        
            return 0;              /* Normal exit. */
        else         
            return EXIT_FAILURE;   /* At least one test failed. */
    } else {
        /* Start processing in piped-solver mode. */
        TestContext ctx;
        unsigned int ready_signal = 0x0fe95617;
        ctx.send(&ready_signal, sizeof(unsigned int), "ready");
        ctx.guard();
        pipe_in(ctx);          
        return 0;                  /* Normal exit. */
    }
}
