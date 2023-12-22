
/*  Auto-built C++ source file 'set-cover.cpp'.  */

/******************************************************************************

  Aalto University
  CS-E3190 Principles of Algorithmic Techniques
  Autumn 2021

  Exercise: Set cover

  Description:

  This exercise asks you to implement an algorithm that computes a low-cost
  set cover. More precisely, the input consists of a nonempty family
  $\mathcal{F}=\{S_0,S_1,\ldots,S_{m-1}\}$ of nonempty subsets
  $S_j\subseteq\{0,1,\ldots,n-1\}$ for $j=0,1,\ldots,m-1$.
  A set $R\subseteq\{0,1,\ldots,m-1\}$ is a *set cover* if
  for all $x\in\{0,1,\ldots,n-1\}$ there exists a $j\in R$ with $x\in S_j$.
  The *cost* of a set cover $R$ is $c(R)=|R|$.
  The *optimum cost* $\mathrm{OPT}$ of a set cover is the minimum cost of a
  set cover, where the minimum is taken over all possible set covers; if no
  set cover exists, the optimum is undefined. The algorithm must 
  either
  
    1. output a set cover $R$ with cost $c(R)\leq H_q\cdot\mathrm{OPT}$,
       where $q=\max_{j=0,1,\ldots,m-1}|S_j|$ and $H_q=1+2+\ldots+\frac{1}{q}$,
       or
    2. correctly assert that no set cover exists.
  
  For example, an implementation of the greedy set cover algorithm suffices
  for this purpose.
  
  **Your task** in this exercise is to complete the subroutine
                     
  ``void solver(int n, int m, const int *p, const int *f, int &k, int *r)``
  
  which should compute the size `k` and the elements `r` of a set cover as
  described in the previous paragraph from the given input consisting of
  positive integers `n` and `m`, as well as the arrays `p` and `f`, whose
  format is as follows.
  
  The array `f` concatenates the sets $S_0,S_1,\ldots,S_{m-1}$.
  That is, writing $S_{j}[0],S_{j}[1],\ldots,S_{j}[|S_j|-1]$ for the $|S_j|$
  distinct elements of the set $S_j$, we have
  $f=(S_{0}[0],S_{0}[1],\ldots,S_{0}[|S_0|-1],\ldots,S_{m-1}[0],S_{m-1}[1],\ldots,S_{m-1}[|S_{m-1}|-1])$.
  
  The array `p` satisfies $p[i]=\sum_{0\leq j<i}|S_j|$ for all $i=0,1,\ldots,m$.
  Thus, the array `p` provides an index to the array `f` with
  $S_j=\{f[p[j]],f[p[j]+1],\ldots,f[p[j+1]-1]\}$ for all $j=0,1,\ldots,m-1$. 
  
  For example, when $m=2$, $S_0=\{4,6,8\}$, and $S_1=\{7,9\}$,
  we have $f=(4,6,8,7,9)$ and $p=(0,3,5)$.
  
  The output of the subroutine should be as follows.
  To give as output a set cover $R=\{j_0,j_1,\ldots,j_{k-1}\}$,
  set `k` equal to $k$ and the element `r[i]` equal to $j_i$ for all
  $i=0,1,\ldots,k-1$. When no set cover exists, set `k` equal to $0$. 
  You may assume that $1\leq n\leq 1048576$, $0\leq m\leq 2097152$,
  $0\leq p[m]\leq 16777216$, $0\leq k\leq n$, $q\leq 10$, and that the array
  $r$ has capacity for at least $n$ elements. To locate the subroutine quickly,
  you can search for "`???`" in the source file.
  
  *Grading*. This exercise awards you up to 10 points in 
  the course grading. The number of points awarded is the maximum points times
  the number of tests passed over the total number of tests, rounded up. To
  successfully complete a test, your implementation must use no more than
  10 seconds of wall clock time and 1 GiB of memory. Each test
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
#include <bits/stdc++.h>
#include <vector>
#include <queue>
#include <utility>

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

void solver(int n, int m, const int *p, const int *f, int &k, int *r);

void pipe_out(TestContext &ctx,
              int n, int m, const int *p, const int *f, int &k, int *r);

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
                if(good && ready_signal != 0xfadf1b09u) {
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

    bool solve(int n, int m, const int *p, const int *f, int &k, int *r) {
        started || DIE("not started");
        !ended  || DIE("already ended");
        good    || DIE("solve called on bad context");
        tick_t solve_start = track::now();
        if(!boxed) {
            /* Solve locally without boxing. */
            solver(n, m, p, f, k, r);        
        } else {
            /* Pipe to solver process. */            
            pipe_out(*this, n, m, p, f, k, r);
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

bool check_instance(int n, int m, const int *p, const int *f)
{
    /* Must have that q <= 10 and q divides n; otherwise return false. */
    if(n <= 0 || m < 0 || p[0] != 0)
        return false;
    for(int i = 0; i < m; i++)
        if(p[i+1] - p[i] <= 0)
            return false;
    int *t = new int[p[m]];
    int i = 0;
    int q = 1;
    for(; i < m; i++) {
        int s = p[i];
        int k = p[i+1] - s;
        if(k > q) {
            q = k;
            if(q > 10)
                break;
        }
        for(int j = 0; j < k; j++)
            t[j] = f[s + j];
        std::sort(t, t + k);
        int j = 1;
        for(; j < k; j++)
            if(t[j-1] >= t[j])
                break;
        if(j < k)
            break;
        if(t[0] < 0 || t[k-1] >= n)
            break;
    }
    delete[] t;
    if(i < m)
        return false;
    if(n % q != 0)
        return false;
    return true;
}

// This function return the difference between the number of uncover elements and the number of uncover element of one set.
int uncover_diff(bool *uncover, pair<int, int> top, const int *p, const int *f)
{
    int uncover_elements = 0;
    for(int i = p[top.second]; i < p[top.second + 1]; i++) {
        if(uncover[f[i]]) {
            uncover_elements ++;
        }    
    }
    return (top.first - uncover_elements);
}

void solver(int n, int m, const int *p, const int *f, int &k, int *r)
{
    priority_queue<pair<int, int>> sets_queue; // The first element in the pair is the number of uncovered element, and the second number is the set index
    bool *uncover = new bool[n];               // uncover[i] is true if i is not covered yet
    int covered = 0;
    int adding = -1;                           // Keep track of how many newly covered elements in each iteration
    k = 0;
    for(int i = 0; i < n; i++) {
        uncover[i] = true;
    }
    for(int i = 0; i < m; i++) {
        sets_queue.push(make_pair((p[i+1] - p[i]), i));
    }
    while(covered != n) {
        if(adding == 0 || sets_queue.empty()) {
            k = 0;
            delete[] uncover;
            return;
        }
        auto curr_set = sets_queue.top(); // The best set
        sets_queue.pop();
        r[k] = curr_set.second;
        k ++;
        adding = 0;
        for(int i = p[curr_set.second]; i < p[curr_set.second + 1]; i++) { // Loop through all element in the chosen set. For each element, find whether that element is in the uncover vector, remove if it does
            if(uncover[f[i]]) {
                uncover[f[i]] = false;
                covered ++;
                adding ++;
            }
        }
        auto next_set = sets_queue.top();
        int diff = uncover_diff(uncover, next_set, p, f); 
        while(diff != 0) {                                                 // Loop while the first element in the queue is not up to date
            sets_queue.pop();
            sets_queue.push(make_pair((next_set.first - diff), next_set.second));
            next_set = sets_queue.top();
            diff = uncover_diff(uncover, next_set, p, f);
        }
    }
    delete[] uncover;
    return;
}

long fact(int q)
{
    q >= 0 || DIE("negative q");
    if(q == 0)
        return 1;
    else
        return q*fact(q-1);
}

bool check_soln(int n, int m, const int *p, const int *f, int k, const int *r)
{
    /* Assumes that the optimum solution has cost n/q. */
    check_instance(n, m, p, f) || DIE("bad instance");
    bool *c = new bool[n];
    bool good = true;
    for(int i = 0; i < n; i++)
        c[i] = false;
    /* Compute the largest set size q. */
    int q = 1;
    for(int i = 0; i < m; i++) {
        int d = p[i+1] - p[i];
        if(d > q)
            q = d;
    }
    (q >= 1 && q <= 10 && n%q == 0) || DIE("bad q");
    if(k > 0) {
        if(k <= n) {
            /* Check that we have a set cover. */
            for(int j = 0; j < k; j++) {
                int i = r[j];
                int s = p[i];
                int d = p[i+1] - p[i];
                for(int j = 0; j < d; j++)
                    c[f[s + j]] = true;
            }
            for(int i = 0; i < n; i++) {
                if(!c[i]) {
                    good = false;
                    break;
                }
            }
            /* Check that we are within the approximation factor. */
            long fs = 0;
            long fq = fact(q);
                /* Will not overflow when n, q are within bounds. */
            for(int i = 1; i <= q; i++)
                fs += fq/i;    /* Exact division. */
            if(fq*k > fs*(n/q))
                good = false;
                    /* k not within factor H_q = 1+1/2+...+1/q of n/q. */
        } else {
            good = false;
        }
    } else {
        /* Check that no set cover exists. */
        for(int j = 0; j < p[m]; j++)
            c[f[j]] = true;
        int i = 0;
        for(; i < n; i++) {
            if(!c[i])
                break;
        }
        if(i == n)
            good = false;
    }
    delete[] c;
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
        int *p = new int [m+1];
        int *r = new int [n];
        
        /* Read input. */
        ctx.recv(p, sizeof(int)*(m+1), "p");
        ctx.guard();

        /* Allocate further buffers. */
        int *f = new int [p[m]];

        /* Read further input. */
        ctx.recv(f, sizeof(int)*p[m], "f");
        ctx.guard();
        
        /* Solve. */
        int k;
        solver(n, m, p, f, k, r);

        /* Write output. */
        ctx.send(&k, sizeof(int), "k");
        ctx.guard();
        ctx.send(r, sizeof(int)*k, "r");
        ctx.guard();
        
        /* Release buffers. */
        delete[] r;
        delete[] f;
        delete[] p;
    }
}

void pipe_out(TestContext &ctx,
              int n, int m, const int *p, const int *f, int &k, int *r)
{
    /* Write preliminaries. */    
    ctx.send(&n, sizeof(int), "preliminaries");
    ctx.send(&m, sizeof(int), "preliminaries");
    
    /* Write input. */   
    ctx.send(p, sizeof(int)*(m+1), "p");    
    ctx.send(f, sizeof(int)*p[m],  "f");
    
    /* Read solver output. */    
    ctx.recv(&k, sizeof(int),  "k");
    if(k < 0 || k > n)
        return;
    ctx.recv(r, sizeof(int)*k, "r");      
}

void perm_rand(std::mt19937 &g, int n, int *p)
{
    n >= 0 || DIE("bad input");
    for(int i = 0; i < n; i++)
        p[i] = i;
    for(int i = 0; i < n-1; i++) {
        std::uniform_int_distribution<> d(0,n-i-1);       
        int x = i + d(g);
        int t = p[x];
        p[x] = p[i];
        p[i] = t;
    }
}

void diagnostics(std::ostream &out,
                 int n, int m, const int *p, const int *f, int k, const int *r)
{
    check_instance(n, m, p, f) || DIE("bad input");
    int i = 0;
    out << std::endl;
    if(n <= 10 && m <= 20) {
        out << "  n = " << n << ", m = " << m << std::endl;
        for(; i < m; i++) {
            out << "  S_{" << i << "} = {";
            int s = p[i];
            int d = p[i+1] - s;
            for(int j = 0; j < d; j++) {
                out << f[s+j];
                if(j < d-1)
                    out << ", ";
                else
                    out << "}";
            }
            out << std::endl;
        }
        out << "  k = " << k << std::endl;
        if(k > 0) {
            out << "  R = {";
            for(int j = 0; j < k && j < n; j++) {
                out << r[j];
                if(j < k-1 && j < n-1)
                    out << ", ";
                else
                    out << "}";
            }
            out << std::endl;
        }
    } else {
        out << "  [instance with n = " << n << " and m = " << m << ", suppressing diagnostics]"
            << std::endl;
    }
    out << std::endl;
}

void trial_rand(TestContext &ctx, std::mt19937 &g,
                int n, int q, int c, bool have)
{
    (n > 0 && c > 0 && q > 0 && n % q == 0) || DIE("bad parameters");
    int opt = n/q;
    int m = c*(opt - (have ? 0 : 1));
    int *rn = new int[n];
    int *rm = new int[m];
    perm_rand(g, m, rm);
    int *p = new int[m+1];
    for(int i = 0; i <= m; i++)
        p[i] = q*i;
    int *f = new int[p[m]];
    int k;
    int *r = new int[n];
    std::uniform_int_distribution<> d(0,n-1);   
    int hole = d(g);

    int mp = 0;
    for(int u = 0; u < c; u++) {
        perm_rand(g, n, rn);
        for(int i = 0; i < opt; i++) {
            int j = 0;
            for(; j < q; j++)
                if(rn[q*i + j] == hole)
                    break;
            if(have || j == q) {
                for(j = 0; j < q; j++)
                    f[p[rm[mp]]+j] = rn[q*i + j];
                std::sort(f + p[rm[mp]], f + p[rm[mp]+1]);
                mp++;
            }
        }
    }
    check_instance(n, m, p, f) || DIE("bad instance");
    if(ctx.solve(n, m, p, f, k, r)) {
        if(!check_soln(n, m, p, f, k, r)) {
            ctx.fail() << "bad solution to instance " << ctx.count();
            diagnostics(ctx.diagnostics(), n, m, p, f, k, r);
        }
    }

    delete[] rm;
    delete[] rn;    
    delete[] r;
    delete[] f;
    delete[] p;
}

bool tests(bool do_boxed)
{
    std::mt19937 g(12345);

    track::start();
    bool good = true;
    for(int opt = 1; opt <= 30; opt++) {
        int q = 3;
        int n = q*opt;
        int c = 5;
        std::cerr << "test n = " << std::setw(3) << n
                  << ", m = " << std::setw(3) << opt*c
                  << ", p[m] = " << std::setw(3) << opt*c*q
                  << " ... ";
        TestContext ctx;
        ctx.start(do_boxed,
                  (tick_t) 10000*TICKS_IN_MSEC);
        int repeats = 1000;
        for(int r = 0; !ctx.bad() && r < repeats; r++)
            trial_rand(ctx, g, n, q, c, r % 2 == 0);
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
    for(int opt = 1; opt <= 1 << 17; opt*=2) {
        int q = 5;
        int n = q*opt;
        int c = 5;
        std::cerr << "scaling test n = " << std::setw(7) << n
                  << ", m = " << std::setw(7) << opt*c
                  << ", p[m] = " << std::setw(8) << opt*c*q
                  << " ... ";
        TestContext ctx;
        ctx.start(do_boxed,
                  (tick_t) 10000*TICKS_IN_MSEC);
        if(!ctx.bad())
            trial_rand(ctx, g, n, q, c, true);
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
                "Set cover";
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
        unsigned int ready_signal = 0xfadf1b09;
        ctx.send(&ready_signal, sizeof(unsigned int), "ready");
        ctx.guard();
        pipe_in(ctx);          
        return 0;                  /* Normal exit. */
    }
}
