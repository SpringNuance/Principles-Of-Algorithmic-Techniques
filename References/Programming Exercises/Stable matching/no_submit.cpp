
/*  Auto-built C++ source file 'stable-matching.cpp'.  */

/******************************************************************************

  Aalto University
  CS-E3190 Principles of Algorithmic Techniques
  Autumn 2021

  Exercise: Stable matching

  Description:

  This exercise asks you to implement an algorithm that computes a stable
  matching between $n$ applicants and $n$ positions. Each applicant
  $i=0,1,\ldots,n-1$ has preferences for each position $j=0,1,\ldots,n-1$,
  and vice versa. These preferences are given as input in two $n^2$-element
  arrays `a_pref` and `b_pref` so that applicant $i=0,1,\ldots,n-1$ prefers
  the $n$ available positions in order
  
  ``a_pref[i*n+0] > a_pref[i*n+1] > ... > a_pref[i*n+n-1]``,
  
  where `a_pref[i*n+0]` is the most preferred position, `a_pref[i*n+1]` is
  the next most preferred position, and so forth until the least preferred
  position `a_pref[i*n+n-1]`. Dually, position $j=0,1,\ldots,n-1$ prefers
  the applicants in order
  
  ``b_pref[j*n+0] > b_pref[j*n+1] > ... > b_pref[j*n+n-1]``,
  
  where `b_pref[j*n+0]` is the most preferred applicant, `b_pref[i*n+1]` is
  the next most preferred applicant, and so forth until the least preferred
  applicant `b_pref[i*n+n-1]`.
  
  The stable matching is output via the array `s` such that each applicant
  $i=0,1,\ldots,n-1$ is matched to the position `s[i]`. Accordingly, 
  the entries `s[0],s[1],...,s[n-1]` must form a permutation of the
  integers $0,1,\ldots,n-1$. Furthermore, for all $i,j=0,1,\ldots,n-1$ such
  that $j\neq s[i]$ it must be that applicant $i$ prefers position $s[i]$
  over position $j$ or position $j$ prefers applicant $s^{-1}[j]$ over
  applicant $i$; indeed, otherwise the pair $(i,j)$ is unstable. Here we
  write $s^{-1}$ for the inverse permutation of $s$.
      
  **Your task** in this exercise is to complete the subroutine
                     
  ``void solver(int n, const int *a_pref, const int *b_pref, int *s)``
  
  which should compute the array `s` from the given inputs `n`, `a_pref`, and
  `b_pref`. To locate the subroutine quickly, you can search for "`???`" in
  the source file. You may assume that $0\leq n\leq 2048$. The source file
  contains many subroutines that you may find useful in preparing your solution.
  For example, the subroutine `perm_inv` computes the inverse of a permutation. 
  
  *Grading*. This exercise awards you up to 10 points in 
  the course grading. The number of points awarded is the maximum points times
  the number of tests passed over the total number of tests, rounded up. To
  successfully complete a test, your implementation must use no more than
  3 seconds of wall clock time and 100 MiB of memory. Each test
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
#include <queue>
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

void solver(int n, const int *a_pref, const int *b_pref, int *s);

void pipe_out(TestContext &ctx,
              int n, const int *a_pref, const int *b_pref, int *s);

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
                if(good && ready_signal != 0xd8b9c04bu) {
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

    bool solve(int n, const int *a_pref, const int *b_pref, int *s) {
        started || DIE("not started");
        !ended  || DIE("already ended");
        good    || DIE("solve called on bad context");
        tick_t solve_start = track::now();
        if(!boxed) {
            /* Solve locally without boxing. */
            solver(n, a_pref, b_pref, s);        
        } else {
            /* Pipe to solver process. */            
            pipe_out(*this, n, a_pref, b_pref, s);
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

bool check_perm(int n, const int *p)
{
    if(n < 0)
        return false;
    int *q = new int [n];
    for(int i = 0; i < n; i++)
        q[i] = p[i];
    std::sort(q, q + n);
    bool good = true;
    for(int i = 0; i < n; i++)
        if(q[i] != i)
            good = false;
    delete[] q;
    return good;  
}

bool check_instance(int n, const int *a_pref, const int *b_pref)
{
    if(n < 0)
        return false;
    for(int i = 0; i < n; i++)
        if(!(check_perm(n, a_pref + i*n) && check_perm(n, b_pref + i*n)))
            return false;
    return true;
}

void perm_inv(int n, const int *p, int *pi)
{
    check_perm(n, p) || DIE("bad permutation");    
    for(int i = 0; i < n; i++)
        pi[p[i]] = i;
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

void part_rand(std::mt19937 &g, int n, int k, int *p)
{
    (n >= 0 && k > 0 && k <= n) || DIE("bad input");
    int *q = new int [n];
    perm_rand(g, n, q);
    for(int i = 0; i < n; i++)
        p[q[i]] = i % k;
    delete[] q;
}

bool betterPositionOrNot ( int oldPos, int currentPos, int n, const int *a_pref, int prefApplicant){
    int betterPosition = 0;
    for ( int i = prefApplicant * n  ; i <  prefApplicant * n + n; i++){
        if (a_pref[i] == oldPos){
            betterPosition = oldPos;
            break;
        } else if ( a_pref[i] == currentPos){
            betterPosition = currentPos;
            break;
        }
    }
    if (currentPos == betterPosition){
        return true;
    } else {
        return false;
    }
}

void solver(int n, const int *a_pref, const int *b_pref, int *s)
{
    vector<int> result;
    for (int i = 0; i < n; i++){
        result.push_back(-1);
    }

    queue<int> positions;
    for (int i = 0; i < n; i++){
        positions.push(i);
    }

    vector<queue<int>> companiesPref;

    for (int i = 0; i < n ; i++){
        queue<int> companyPref;
        for (int j = 0; j < n; j ++){
            companyPref.push(b_pref[i * n + j]);
        }
        companiesPref.push_back(companyPref);
    }


    while (!positions.empty()){
        int currentPosition = positions.front();
        int preferredApplicant = companiesPref[currentPosition].front();
        positions.pop();
        companiesPref[currentPosition].pop();
        int oldPosOfPrefApplicant = result[preferredApplicant];
        if (oldPosOfPrefApplicant == -1){
            result[preferredApplicant] = currentPosition;
        } else if (betterPositionOrNot(oldPosOfPrefApplicant,currentPosition, n, a_pref,preferredApplicant)){
            result[preferredApplicant] = currentPosition;
            positions.push(oldPosOfPrefApplicant);

        }else {
            positions.push(currentPosition);
        }
        }


    for (int i= 0; i < n; i++){
        int position = result[i];
        int applicant = i;
        s[applicant] = position;
    }


}

bool check_soln(int n, const int *a_pref, const int *b_pref, int *s)
{
    check_instance(n, a_pref, b_pref) || DIE("bad instance");

    if(!check_perm(n, s))
        return false;

    int *si = new int [n];
    int *a_rk = new int [n*n];
    int *b_rk = new int [n*n];

    perm_inv(n, s, si);
    for(int i = 0; i < n; i++) {
        perm_inv(n, a_pref + i*n, a_rk + i*n);
        perm_inv(n, b_pref + i*n, b_rk + i*n);
    }
    bool good = true;
    for(int i = 0; good && i < n; i++)
        for(int j = 0; good && j < n; j++)
            if(j != s[i] &&
               a_rk[i*n + j] < a_rk[i*n + s[i]] &&
               b_rk[j*n + i] < b_rk[j*n + si[j]])
                good = false;
    
    delete[] b_rk;
    delete[] a_rk;
    delete[] si;
               
    return good;
}

void pipe_in(TestContext &ctx)
{
    while(true) {        
        /* Pipe-in solver loop. */

        /* Read preliminaries. */
        int n;
        ctx.recv(&n, sizeof(int), "preliminaries");
        ctx.guard();
        if(n < 0) 
            break; /* Stop. */

        /* Allocate buffers. */
        int *a_pref = new int [n*n];
        int *b_pref = new int [n*n];
        int *s = new int [n];
        
        /* Read input. */
        ctx.recv(a_pref, sizeof(int)*n*n, "a_pref");
        ctx.recv(b_pref, sizeof(int)*n*n, "b_pref");
        ctx.guard();

        /* Solve. */
        solver(n, a_pref, b_pref, s);

        /* Write output. */
        ctx.send(s, sizeof(int)*n, "output");
        ctx.guard();
        
        /* Release buffers. */
        delete[] s;
        delete[] b_pref;
        delete[] a_pref;
    }
}

void pipe_out(TestContext &ctx,
              int n, const int *a_pref, const int *b_pref, int *s)
{
    /* Interact with solver process via pipes. */
    
    /* Write preliminaries. */    
    ctx.send(&n, sizeof(int), "preliminaries");
    
    /* Write input. */   
    ctx.send(a_pref, sizeof(int)*n*n, "a_pref");
    ctx.send(b_pref, sizeof(int)*n*n, "b_pref");
    
    /* Read solver output. */    
    ctx.recv(s, sizeof(int)*n, "output");
}

void inst_rand(std::mt19937 &g, int n, int ka, int kb, int *a_pref, int *b_pref)
{
    int *pa = new int [ka*n];
    int *pb = new int [kb*n];
    int *qa = new int [n];
    int *qb = new int [n];
    part_rand(g, n, ka, qa);
    part_rand(g, n, kb, qb);   
    for(int i = 0; i < ka; i++)     
        perm_rand(g, n, pa + i*n);
    for(int i = 0; i < kb; i++)     
        perm_rand(g, n, pb + i*n);
    for(int i = 0; i < n; i++)     
        for(int j = 0; j < n; j++)
            a_pref[i*n + j] = pa[qa[i]*n + j];
    for(int i = 0; i < n; i++)     
        for(int j = 0; j < n; j++)
            b_pref[i*n + j] = pb[qb[i]*n + j];  
    delete[] qb;
    delete[] qa;
    delete[] pb;
    delete[] pa;
}

void inst_long1(int n, int *a_pref, int *b_pref)
{
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            a_pref[i*n + j] = (i == n-1) ? j : (j == n-1) ? n-1 : (i+j)%(n-1);
            b_pref[i*n + j] = (i + j + 1) % n;
        }
    }
}

void inst_long2(int n, int *a_pref, int *b_pref)
{
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            a_pref[i*n + j] = (i == n-1) ? j : (j == n-1) ? n-1 : (i+j)%(n-1);
            b_pref[i*n + j] = (n + i - j + 1) % n;
        }
    }
}

void diagnostics(std::ostream &out,
                 int n, const int *a_pref, const int *b_pref, const int *s)
{
    n >= 0 || DIE("bad input");
    out << std::endl;
    if(n > 10) {
        out << "  [instance with n = " << n << ", suppressing diagnostics]"
            << std::endl;
    } else {        
        out << "  n = " << n << std::endl << std::endl;
        for(int i = 0; i < n; i++) {
            out << "  a" << i << ": ";
            for(int j = 0; j < n; j++) {
                if(j > 0)
                    out << " > ";
                out << "b" << a_pref[i*n + j];
            }
            out << std::endl;
        }
        out << std::endl;
        for(int i = 0; i < n; i++) {
            out << "  b" << i << ": ";
            for(int j = 0; j < n; j++) {
                if(j > 0)
                    out << " > ";
                out << "a" << b_pref[i*n + j];
            }
            out << std::endl;
        }
        out << std::endl;
        for(int i = 0; i < n; i++)
            out << "  s[" << i << "] = " << s[i] << std::endl;
    }
    out << std::endl;
}

void trial_rand(TestContext &ctx, std::mt19937 &g, int n, int ka, int kb)
{
    int *a_pref = new int [n*n];
    int *b_pref = new int [n*n];
    int *s = new int [n];
    inst_rand(g, n, ka, kb, a_pref, b_pref);
    check_instance(n, a_pref, b_pref) || DIE("bad instance");
    if(ctx.solve(n, a_pref, b_pref, s)) {
        if(!check_soln(n, a_pref, b_pref, s)) {
            ctx.fail() << "bad solution to instance " << ctx.count();
            diagnostics(ctx.diagnostics(), n, a_pref, b_pref, s);
        }
    }
    delete[] s;
    delete[] b_pref;
    delete[] a_pref;
}

void trial_long(TestContext &ctx, std::mt19937 &g, int n, int choice)
{
    int *a_pref = new int [n*n];
    int *b_pref = new int [n*n];
    int *s = new int [n];
    switch(choice) {
    case 0:
        inst_long1(n, a_pref, b_pref);
        break;
    case 1:
        inst_long1(n, b_pref, a_pref);
        break;
    case 2:
        inst_long2(n, a_pref, b_pref);
        break;
    case 3:
        inst_long2(n, b_pref, a_pref);
        break;
    default:
        DIE("unsupported choice");
    }
    check_instance(n, a_pref, b_pref) || DIE("bad instance");
    if(ctx.solve(n, a_pref, b_pref, s)) {
        if(!check_soln(n, a_pref, b_pref, s)) {
            ctx.fail() << "bad solution to instance " << ctx.count();
            diagnostics(ctx.diagnostics(), n, a_pref, b_pref, s);
        }
    }
    delete[] s;
    delete[] b_pref;
    delete[] a_pref;
}

bool tests(bool do_boxed)
{
    std::mt19937 g(12345);

    track::start();
    bool good = true;
    for(int n = 1; n <= 5; n++) {
        for(int ka = 1; ka <= n; ka++) {                
            for(int kb = 1; kb <= n; kb++) {
                std::cerr << "test n = " << std::setw(2) << n
                          << ", ka = " << std::setw(2) << ka
                          << ", kb = " << std::setw(2) << kb
                          << " ... ";
                TestContext ctx;
                ctx.start(do_boxed,
                          (tick_t) 3000*TICKS_IN_MSEC);
                int repeats = 100;
                for(int r = 0; !ctx.bad() && r < repeats; r++)
                    trial_rand(ctx, g, n, ka, kb);
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
        }
    }
    for(int n = 1; n <= 2048; n*=2) {
        std::cerr << "scaling test n = " << std::setw(5) << n
                  << " ... ";
        const int repeats = 4;
        long timing[repeats];
        bool testgood = true;
        for(int r = 0; r < repeats; r++) {        
            TestContext ctx;
            ctx.start(do_boxed,
                      (tick_t) 3000*TICKS_IN_MSEC);
            if(!ctx.bad())
                trial_long(ctx, g, n, r);
            ctx.end();
            if(ctx.bad()) {
                std::cerr << "FAIL [" << ctx.fail().str() << "]";
                good = false;
                std::cerr << std::endl;
                if(ctx.bad())
                    std::cerr << ctx.diagnostics().str();        
                testgood = false;
                break;
            } else {
                timing[r] = ctx.timing_ms();
            }
        }
        if(testgood) {
            std::cerr << "OK   [";          
            for(int r = 0; r < repeats; r++) {
                std::cerr << std::setw(5) << timing[r] << " ms";
                if(r == repeats-1)
                    std::cerr << "]";
                else
                    std::cerr << ", ";              
            }
            std::cerr << std::endl;
        }
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
                "Stable matching";
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
        unsigned int ready_signal = 0xd8b9c04b;
        ctx.send(&ready_signal, sizeof(unsigned int), "ready");
        ctx.guard();
        pipe_in(ctx);          
        return 0;                  /* Normal exit. */
    }
}
