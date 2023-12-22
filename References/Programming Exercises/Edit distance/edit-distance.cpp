
/*  Auto-built C++ source file 'edit-distance.cpp'.  */

/******************************************************************************

  Aalto University
  CS-E3190 Principles of Algorithmic Techniques
  Autumn 2021

  Exercise: Edit distance

  Description:

  This exercise asks you to implement an algorithm that computes the edit
  distance (Levenshtein distance) between two given strings. More precisely,
  for two strings $a$ and $b$, the *Levenshtein distance* $D(a,b)$ is the
  minimum length of a sequence of edit operations that transforms the string
  $a$ into the string $b$, where each edit operation in the sequence is one of
  the following:
  
      1. deletion of one character from the string,
      2. insertion of one character into the string, or
      3. changing of one character in the string into another character.
  
  For example, the Levenshtein distance satisfies $D(a,a)=0$ for all strings
  $a$, since a sequence of zero edit operations suffices to transform $a$
  into $a$. Similarly, one can show that $D(a,b)=D(b,a)$ and
  $D(a,c)\leq D(a,b)+D(b,c)$ for all strings $a,b,c$. That is, $D$ is a metric
  in the space of all strings.
   
  **Your task** in this exercise is to complete the subroutine
                     
  ``void solver(int n, int m, const char *a, const char *b, int &d)`` ,
  
  which takes as input a string `a` of length `n` and a string `b` of
  length `m`. The subroutine should compute the Levenshtein distance $D(a,b)$
  and store it into `d`. To locate the subroutine quickly, you can search for
  "`???`" in the source file. You may assume that $n,m\geq 0$ and
  $nm\leq 4294967296$. Furthermore, you may assume that $a[n]=b[m]=0$.
  
  *Grading*. This exercise awards you up to 10 points in 
  the course grading. The number of points awarded is the maximum points times
  the number of tests passed over the total number of tests, rounded up. To
  successfully complete a test, your implementation must use no more than
  15 seconds of wall clock time and 10 MiB of memory. Each test
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

void solver(int n, int m, const char *a, const char *b, int &d);

void pipe_out(TestContext &ctx,
              int n, int m, const char *a, const char *b, int &d);

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
                if(good && ready_signal != 0x2f141dceu) {
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

    bool solve(int n, int m, const char *a, const char *b, int &d) {
        started || DIE("not started");
        !ended  || DIE("already ended");
        good    || DIE("solve called on bad context");
        tick_t solve_start = track::now();
        if(!boxed) {
            /* Solve locally without boxing. */
            solver(n, m, a, b, d);        
        } else {
            /* Pipe to solver process. */            
            pipe_out(*this, n, m, a, b, d);
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

bool check_instance(int n, int m, const char *a, const char *b)
{
    if(n < 0 || m < 0)
        return false;
    return true;
}

void solver(int n, int m, const char *a, const char *b, int &d)
{
    int **res = new int*[2];
    for(int k = 0; k < 2; k++) {
        res[k] = new int[m+1];
    }
    for(int i = 0; i <= n; i++) {
        for(int j = 0; j <= m; j ++) {
            if(i == 0) {
                res[i][j] = j;
            } else if(j == 0) {
                res[i%2][j] = i;
            } else if(a[i-1] == b[j-1]) {
                res[i%2][j] = res[(i-1)%2][j-1];
            } else {
                int rmv = res[(i-1)%2][j];
                int add = res[i%2][j-1];
                int sub = res[(i-1)%2][j-1];
                int mid;
                if(rmv+1 < add+1) {
                    mid = rmv+1;
                } else {
                    mid = add+1;
                }
                if(mid < sub+1) {
                    res[i%2][j] = mid;
                } else {
                    res[i%2][j] = sub+1;
                }
            }
        }
    }
    d = res[n%2][m];
    for(int k = 0; k < 2; k++) {
        delete[] res[k];
    }
    delete[] res;
    return;
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
        char *a = new char [n+1];
        char *b = new char [m+1];
        
        /* Read input. */
        ctx.recv(a, sizeof(char)*(n+1), "a");
        ctx.guard();
        ctx.recv(b, sizeof(char)*(m+1), "b");
        ctx.guard();
        
        /* Solve. */
        int d;
        solver(n, m, a, b, d);

        /* Write output. */
        ctx.send(&d, sizeof(int), "d");
        ctx.guard();
        
        /* Release buffers. */
        delete[] b;
        delete[] a;
    }
}

void pipe_out(TestContext &ctx,
              int n, int m, const char *a, const char *b, int &d)
{
    /* Write preliminaries. */    
    ctx.send(&n, sizeof(int), "preliminaries");
    ctx.send(&m, sizeof(int), "preliminaries");
    
    /* Write input. */   
    ctx.send(a, sizeof(char)*(n+1), "a");    
    ctx.send(b, sizeof(char)*(m+1), "b");
    
    /* Read solver output. */    
    ctx.recv(&d, sizeof(int),  "d");
}

void diagnostics(std::ostream &out,
                 int n, int m, const char *a, const char *b, int d, int dt)
{
    check_instance(n, m, a, b) || DIE("bad input");
    out << std::endl;
    if(n <= 10 && m <= 10) {
        out << "  a = \"" << a << "\"" << std::endl;
        out << "  b = \"" << b << "\"" << std::endl;
        out << "  d = " << d << ", dt = " << dt << std::endl;
    } else {
        out << "  [instance with n = " << n << " and m = " << m << ", suppressing diagnostics]"
            << std::endl;
    }
    out << std::endl;
}

void trial_rand(TestContext &ctx, int seed, int n, int m, int dt)
{
    (n >= 0 && m >= 0 && dt >= 0) || DIE("bad parameters");

    std::mt19937 g(seed);
    
    char *a = new char [n+1];
    char *b = new char [m+1];
    a[n] = 0;
    b[m] = 0;
    check_instance(n, m, a, b) || DIE("bad instance");

    const char *alphabet = "agct";
    int alphabet_len = 4;
    std::uniform_int_distribution<> da(0,alphabet_len-1);   
    for(int i = 0; i < n; i++)
        a[i] = alphabet[da(g)];
    for(int i = 0; i < m; i++)
        b[i] = alphabet[da(g)];
    
    int d;
    if(ctx.solve(n, m, a, b, d)) {
        if(d != dt) {
            ctx.fail() << "bad solution to instance " << ctx.count();
            diagnostics(ctx.diagnostics(), n, m, a, b, d, dt);
        }
    }
    delete[] b;
    delete[] a;
}

const int baseline_tests[][4] = { { 992670690, 0, 0, 0 }, { 823185381, 0, 0, 0 }, { 358822685, 0, 0, 0 }, { 561383553, 0, 0, 0 }, { 789925284, 0, 0, 0 }, { 170765737, 0, 0, 0 }, { 878579710, 0, 0, 0 }, { 549516158, 0, 0, 0 }, { 438360421, 0, 0, 0 }, { 285257250, 0, 0, 0 }, { 557845021, 0, 1, 1 }, { 107320065, 0, 1, 1 }, { 142558326, 0, 1, 1 }, { 983958385, 0, 1, 1 }, { 805374267, 0, 1, 1 }, { 967425166, 0, 1, 1 }, { 216529513, 0, 1, 1 }, { 605979227, 0, 1, 1 }, { 807061239, 0, 1, 1 }, { 665605494, 0, 1, 1 }, { 211410640, 0, 2, 2 }, { 832587122, 0, 2, 2 }, { 128781001, 0, 2, 2 }, { 115061003, 0, 2, 2 }, { 36027469, 0, 2, 2 }, { 251993226, 0, 2, 2 }, { 457175121, 0, 2, 2 }, { 712592594, 0, 2, 2 }, { 282922662, 0, 2, 2 }, { 467278599, 0, 2, 2 }, { 819264555, 0, 3, 3 }, { 693349607, 0, 3, 3 }, { 478118423, 0, 3, 3 }, { 899507741, 0, 3, 3 }, { 745967032, 0, 3, 3 }, { 389708215, 0, 3, 3 }, { 143129887, 0, 3, 3 }, { 607425725, 0, 3, 3 }, { 108204897, 0, 3, 3 }, { 216844123, 0, 3, 3 }, { 759410519, 0, 4, 4 }, { 462752292, 0, 4, 4 }, { 81439808, 0, 4, 4 }, { 997822959, 0, 4, 4 }, { 8322435, 0, 4, 4 }, { 563495164, 0, 4, 4 }, { 398375548, 0, 4, 4 }, { 967598725, 0, 4, 4 }, { 888259215, 0, 4, 4 }, { 555401847, 0, 4, 4 }, { 133990731, 0, 5, 5 }, { 576360846, 0, 5, 5 }, { 269260147, 0, 5, 5 }, { 367318865, 0, 5, 5 }, { 907150443, 0, 5, 5 }, { 166259589, 0, 5, 5 }, { 396556834, 0, 5, 5 }, { 563106101, 0, 5, 5 }, { 734071155, 0, 5, 5 }, { 562109000, 0, 5, 5 }, { 115316741, 1, 0, 1 }, { 414372578, 1, 0, 1 }, { 437564012, 1, 0, 1 }, { 999953669, 1, 0, 1 }, { 881458747, 1, 0, 1 }, { 529943123, 1, 0, 1 }, { 105983500, 1, 0, 1 }, { 863176590, 1, 0, 1 }, { 112038651, 1, 0, 1 }, { 572980304, 1, 0, 1 }, { 260248731, 1, 1, 0 }, { 908238630, 1, 1, 1 }, { 561372504, 1, 1, 1 }, { 187221984, 1, 1, 1 }, { 223155947, 1, 1, 1 }, { 818012165, 1, 1, 1 }, { 844380234, 1, 1, 1 }, { 487468620, 1, 1, 0 }, { 127879446, 1, 1, 1 }, { 441282060, 1, 1, 1 }, { 514786552, 1, 2, 2 }, { 20176958, 1, 2, 2 }, { 148440361, 1, 2, 2 }, { 629275282, 1, 2, 1 }, { 479737010, 1, 2, 2 }, { 3195986, 1, 2, 1 }, { 412181689, 1, 2, 1 }, { 82823982, 1, 2, 2 }, { 940383290, 1, 2, 1 }, { 783365911, 1, 2, 2 }, { 111189909, 1, 3, 3 }, { 743031559, 1, 3, 2 }, { 10498897, 1, 3, 2 }, { 261125695, 1, 3, 3 }, { 972992891, 1, 3, 2 }, { 394542592, 1, 3, 2 }, { 47321306, 1, 3, 2 }, { 978368172, 1, 3, 2 }, { 764731833, 1, 3, 2 }, { 922418062, 1, 3, 2 }, { 282559898, 1, 4, 3 }, { 105711275, 1, 4, 3 }, { 720447390, 1, 4, 3 }, { 596512483, 1, 4, 4 }, { 302030624, 1, 4, 3 }, { 645853955, 1, 4, 3 }, { 986462144, 1, 4, 3 }, { 283211782, 1, 4, 4 }, { 617755330, 1, 4, 3 }, { 27045809, 1, 4, 3 }, { 645033211, 1, 5, 4 }, { 879294844, 1, 5, 4 }, { 102930668, 1, 5, 4 }, { 169416485, 1, 5, 4 }, { 620684232, 1, 5, 4 }, { 613878827, 1, 5, 5 }, { 644715326, 1, 5, 4 }, { 118490326, 1, 5, 4 }, { 913132821, 1, 5, 4 }, { 60299841, 1, 5, 4 }, { 648617696, 2, 0, 2 }, { 71322885, 2, 0, 2 }, { 355044603, 2, 0, 2 }, { 490934504, 2, 0, 2 }, { 441234290, 2, 0, 2 }, { 421743225, 2, 0, 2 }, { 806672318, 2, 0, 2 }, { 30394580, 2, 0, 2 }, { 540485319, 2, 0, 2 }, { 431739001, 2, 0, 2 }, { 953201477, 2, 1, 2 }, { 158530349, 2, 1, 1 }, { 434284690, 2, 1, 2 }, { 193765492, 2, 1, 2 }, { 463320776, 2, 1, 1 }, { 31756235, 2, 1, 1 }, { 179990930, 2, 1, 1 }, { 87798149, 2, 1, 1 }, { 723242812, 2, 1, 1 }, { 985074806, 2, 1, 1 }, { 815742245, 2, 2, 2 }, { 761069449, 2, 2, 2 }, { 127790188, 2, 2, 2 }, { 452106877, 2, 2, 2 }, { 510181478, 2, 2, 0 }, { 303295297, 2, 2, 0 }, { 67596078, 2, 2, 2 }, { 437303193, 2, 2, 2 }, { 164359083, 2, 2, 1 }, { 355136501, 2, 2, 2 }, { 969106112, 2, 3, 1 }, { 859278157, 2, 3, 3 }, { 52157192, 2, 3, 2 }, { 381577096, 2, 3, 2 }, { 405053074, 2, 3, 2 }, { 81491055, 2, 3, 2 }, { 126749327, 2, 3, 1 }, { 906923354, 2, 3, 1 }, { 930297376, 2, 3, 2 }, { 225056748, 2, 3, 1 }, { 60341106, 2, 4, 2 }, { 825651808, 2, 4, 3 }, { 290214998, 2, 4, 3 }, { 549183045, 2, 4, 3 }, { 84447443, 2, 4, 4 }, { 624884815, 2, 4, 2 }, { 96078946, 2, 4, 4 }, { 790147868, 2, 4, 3 }, { 578137479, 2, 4, 4 }, { 238958340, 2, 4, 2 }, { 141678759, 2, 5, 4 }, { 336121074, 2, 5, 4 }, { 973132419, 2, 5, 5 }, { 394720342, 2, 5, 4 }, { 518552864, 2, 5, 4 }, { 457899765, 2, 5, 3 }, { 795387149, 2, 5, 4 }, { 280493396, 2, 5, 3 }, { 344036545, 2, 5, 5 }, { 963444377, 2, 5, 3 }, { 301693612, 3, 0, 3 }, { 653405445, 3, 0, 3 }, { 357964785, 3, 0, 3 }, { 923115646, 3, 0, 3 }, { 661070850, 3, 0, 3 }, { 540989267, 3, 0, 3 }, { 79814104, 3, 0, 3 }, { 105344688, 3, 0, 3 }, { 460948393, 3, 0, 3 }, { 881056675, 3, 0, 3 }, { 911535621, 3, 1, 3 }, { 471018803, 3, 1, 2 }, { 443345944, 3, 1, 2 }, { 351050861, 3, 1, 2 }, { 549276905, 3, 1, 2 }, { 506381902, 3, 1, 2 }, { 857312851, 3, 1, 2 }, { 269083709, 3, 1, 2 }, { 183460861, 3, 1, 3 }, { 424562602, 3, 1, 2 }, { 840933102, 3, 2, 2 }, { 591640, 3, 2, 1 }, { 264873566, 3, 2, 2 }, { 400498956, 3, 2, 2 }, { 692943993, 3, 2, 3 }, { 832386187, 3, 2, 2 }, { 370325431, 3, 2, 3 }, { 162565860, 3, 2, 3 }, { 613944172, 3, 2, 2 }, { 384905689, 3, 2, 2 }, { 215458024, 3, 3, 3 }, { 947011609, 3, 3, 2 }, { 960698464, 3, 3, 3 }, { 357495009, 3, 3, 2 }, { 679179868, 3, 3, 3 }, { 302265918, 3, 3, 2 }, { 780397254, 3, 3, 1 }, { 416262262, 3, 3, 2 }, { 498033211, 3, 3, 2 }, { 825085170, 3, 3, 3 }, { 54219744, 3, 4, 4 }, { 598603981, 3, 4, 3 }, { 84122149, 3, 4, 4 }, { 264936599, 3, 4, 3 }, { 866156392, 3, 4, 3 }, { 380419980, 3, 4, 3 }, { 899636946, 3, 4, 4 }, { 450756277, 3, 4, 4 }, { 88136285, 3, 4, 2 }, { 984748534, 3, 4, 3 }, { 121712641, 3, 5, 3 }, { 75188761, 3, 5, 4 }, { 801286848, 3, 5, 4 }, { 480792763, 3, 5, 4 }, { 204204058, 3, 5, 4 }, { 798679864, 3, 5, 4 }, { 930698362, 3, 5, 4 }, { 851303308, 3, 5, 4 }, { 102559962, 3, 5, 3 }, { 271258391, 3, 5, 5 }, { 979244988, 4, 0, 4 }, { 438136182, 4, 0, 4 }, { 215686751, 4, 0, 4 }, { 44286797, 4, 0, 4 }, { 309924345, 4, 0, 4 }, { 629852533, 4, 0, 4 }, { 437429573, 4, 0, 4 }, { 741286320, 4, 0, 4 }, { 642449344, 4, 0, 4 }, { 232172618, 4, 0, 4 }, { 43910559, 4, 1, 3 }, { 935029058, 4, 1, 3 }, { 783972010, 4, 1, 3 }, { 286127166, 4, 1, 4 }, { 135761043, 4, 1, 4 }, { 490647935, 4, 1, 4 }, { 418300465, 4, 1, 3 }, { 624026361, 4, 1, 3 }, { 86685050, 4, 1, 3 }, { 673158835, 4, 1, 3 }, { 995711804, 4, 2, 4 }, { 174722575, 4, 2, 2 }, { 127815285, 4, 2, 3 }, { 922949730, 4, 2, 3 }, { 981898987, 4, 2, 3 }, { 285367569, 4, 2, 3 }, { 79264396, 4, 2, 3 }, { 224025742, 4, 2, 2 }, { 134516205, 4, 2, 3 }, { 440608000, 4, 2, 3 }, { 779624091, 4, 3, 2 }, { 522567719, 4, 3, 2 }, { 64849027, 4, 3, 3 }, { 764656126, 4, 3, 3 }, { 904018577, 4, 3, 2 }, { 915402732, 4, 3, 4 }, { 390355801, 4, 3, 3 }, { 327813294, 4, 3, 2 }, { 709525036, 4, 3, 2 }, { 655279818, 4, 3, 3 }, { 343405474, 4, 4, 2 }, { 341484136, 4, 4, 4 }, { 623990096, 4, 4, 2 }, { 776871244, 4, 4, 3 }, { 565473870, 4, 4, 4 }, { 693524886, 4, 4, 3 }, { 670180617, 4, 4, 1 }, { 964042652, 4, 4, 1 }, { 761514080, 4, 4, 2 }, { 15372752, 4, 4, 4 }, { 727557757, 4, 5, 2 }, { 447148988, 4, 5, 3 }, { 245947128, 4, 5, 4 }, { 303665279, 4, 5, 4 }, { 513327347, 4, 5, 3 }, { 51211437, 4, 5, 1 }, { 949893127, 4, 5, 3 }, { 299440537, 4, 5, 2 }, { 765996621, 4, 5, 4 }, { 21431675, 4, 5, 3 }, { 790886053, 5, 0, 5 }, { 806098091, 5, 0, 5 }, { 975158269, 5, 0, 5 }, { 921913314, 5, 0, 5 }, { 199469928, 5, 0, 5 }, { 967701765, 5, 0, 5 }, { 276768688, 5, 0, 5 }, { 810257892, 5, 0, 5 }, { 853081926, 5, 0, 5 }, { 712372936, 5, 0, 5 }, { 1546918, 5, 1, 4 }, { 905528427, 5, 1, 4 }, { 669678305, 5, 1, 4 }, { 607335649, 5, 1, 5 }, { 100554462, 5, 1, 4 }, { 931829821, 5, 1, 4 }, { 224426417, 5, 1, 4 }, { 927413056, 5, 1, 4 }, { 488811917, 5, 1, 5 }, { 757744872, 5, 1, 5 }, { 63715761, 5, 2, 3 }, { 954033906, 5, 2, 4 }, { 93239338, 5, 2, 4 }, { 587586224, 5, 2, 4 }, { 604687648, 5, 2, 4 }, { 336059541, 5, 2, 4 }, { 778510733, 5, 2, 3 }, { 166874833, 5, 2, 4 }, { 593761813, 5, 2, 3 }, { 688097933, 5, 2, 4 }, { 885138786, 5, 3, 3 }, { 176904786, 5, 3, 3 }, { 602900751, 5, 3, 3 }, { 760762381, 5, 3, 3 }, { 348039576, 5, 3, 4 }, { 564155471, 5, 3, 5 }, { 720997873, 5, 3, 4 }, { 171584228, 5, 3, 2 }, { 412477070, 5, 3, 3 }, { 145508789, 5, 3, 3 }, { 835939704, 5, 4, 4 }, { 277631532, 5, 4, 3 }, { 127384629, 5, 4, 4 }, { 996479102, 5, 4, 3 }, { 248568786, 5, 4, 3 }, { 756759023, 5, 4, 3 }, { 664926103, 5, 4, 2 }, { 498917949, 5, 4, 2 }, { 141670444, 5, 4, 3 }, { 154475618, 5, 4, 2 }, { 527142745, 5, 5, 4 }, { 301321155, 5, 5, 4 }, { 793699473, 5, 5, 2 }, { 615703048, 5, 5, 4 }, { 727855917, 5, 5, 2 }, { 720511021, 5, 5, 4 }, { 104985614, 5, 5, 3 }, { 686911368, 5, 5, 4 }, { 460640, 5, 5, 4 }, { 119831120, 5, 5, 3 } };

const int scaling_tests[][4] = { { 992670690, 16, 16, 11 }, { 823185381, 32, 32, 20 }, { 358822685, 64, 64, 36 }, { 561383553, 128, 128, 73 }, { 789925284, 256, 256, 141 }, { 170765737, 512, 512, 280 }, { 878579710, 1024, 1024, 538 }, { 549516158, 2048, 2048, 1065 }, { 438360421, 4096, 4096, 2124 }, { 285257250, 8192, 8192, 4254 }, { 557845021, 16384, 16384, 8489 }, { 107320065, 32768, 32768, 17005 }, { 142558326, 65536, 65536, 33880 }, { 983958385, 131072, 131072, 67734 },
{ 805374267, 262144, 262144, 135518 } };
    
bool tests(bool do_boxed)
{
    track::start();
    bool good = true;
    int tno = 0;
    for(int n = 0; n <= 5; n++) {
        for(int m = 0; m <= 5; m++) {
            std::cerr << "test n = " << std::setw(3) << n
                      << ", m = " << std::setw(3) << m
                      << " ... ";
            TestContext ctx;
            ctx.start(do_boxed,
                      (tick_t) 15000*TICKS_IN_MSEC);
            int repeats = 10;
            for(int r = 0; r < repeats; r++) {
                if(!ctx.bad()) {
                    int seed = baseline_tests[tno][0];
                    (baseline_tests[tno][1] == n &&
                     baseline_tests[tno][2] == m) || DIE("bad test");
                    int dt = baseline_tests[tno][3];
                    trial_rand(ctx, seed, n, m, dt);
                }
                tno++;
            }
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
    tno = 0;
    for(int n = 16; n <= 1 << 16; n*=2) {
        int m = n;
        std::cerr << "scaling test n = " << std::setw(7) << n
                  << ", m = " << std::setw(7) << m
                  << " ... ";
        TestContext ctx;
        ctx.start(do_boxed,
                  (tick_t) 15000*TICKS_IN_MSEC);
        if(!ctx.bad()) {
            int seed = scaling_tests[tno][0];
            (scaling_tests[tno][1] == n &&
             scaling_tests[tno][2] == m) || DIE("bad test");
            int dt = scaling_tests[tno][3];
            trial_rand(ctx, seed, n, m, dt);
        }
        tno++;
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
                "Edit distance";
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
        unsigned int ready_signal = 0x2f141dce;
        ctx.send(&ready_signal, sizeof(unsigned int), "ready");
        ctx.guard();
        pipe_in(ctx);          
        return 0;                  /* Normal exit. */
    }
}
