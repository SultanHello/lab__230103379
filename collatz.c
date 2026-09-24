// Parallel-computing practicum: empirical OpenMP measurements.
// Build: gcc-16 -O2 -fopenmp -std=c11 collatz.c -o collatz
// Usage: ./collatz <mode> <threads> [chunk]
// Modes: seq, static, static_chunk, dynamic, guided, false_share, reduction
#include <inttypes.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N 13379000ULL
#define MOD 1000000007ULL
#define MAX_THREADS 64

typedef struct {
    uint32_t max_steps;
    uint64_t checksum;
    uint64_t hits;
} Result;

static inline uint32_t collatz_steps(uint64_t n) {
    uint32_t steps = 0;
    while (n > 1) {
        n = (n & 1) ? 3 * n + 1 : n >> 1;
        ++steps;
    }
    return steps;
}

static Result run_kernel(const char *mode, int threads, int chunk, double *elapsed) {
    uint32_t max_steps = 0;
    uint64_t checksum = 0, hits = 0;
    uint64_t shared_hits[MAX_THREADS] = {0};
    omp_set_num_threads(threads);
    double t0 = omp_get_wtime();

    if (!strcmp(mode, "seq")) {
        for (uint64_t i = 1; i <= N; ++i) {
            uint32_t s = collatz_steps(i);
            if (s > max_steps) max_steps = s;
            checksum = (checksum + s) % MOD;
            if (s > 100) ++hits;
        }
    } else if (!strcmp(mode, "false_share")) {
        #pragma omp parallel for reduction(max:max_steps) reduction(+:checksum) schedule(static)
        for (uint64_t i = 1; i <= N; ++i) {
            uint32_t s = collatz_steps(i);
            if (s > 100) ++shared_hits[omp_get_thread_num()];
            if (s > max_steps) max_steps = s;
            checksum += s;
        }
        for (int t = 0; t < threads; ++t) hits += shared_hits[t];
        checksum %= MOD;
    } else if (!strcmp(mode, "reduction")) {
        #pragma omp parallel for reduction(max:max_steps) reduction(+:checksum,hits) schedule(static)
        for (uint64_t i = 1; i <= N; ++i) {
            uint32_t s = collatz_steps(i);
            if (s > 100) ++hits;
            if (s > max_steps) max_steps = s;
            checksum += s;
        }
        checksum %= MOD;
    } else if (!strcmp(mode, "static_chunk")) {
        #pragma omp parallel for reduction(max:max_steps) reduction(+:checksum,hits) schedule(static, chunk)
        for (uint64_t i = 1; i <= N; ++i) {
            uint32_t s = collatz_steps(i);
            if (s > 100) ++hits;
            if (s > max_steps) max_steps = s;
            checksum += s;
        }
        checksum %= MOD;
    } else if (!strcmp(mode, "dynamic")) {
        #pragma omp parallel for reduction(max:max_steps) reduction(+:checksum,hits) schedule(dynamic, chunk)
        for (uint64_t i = 1; i <= N; ++i) {
            uint32_t s = collatz_steps(i);
            if (s > 100) ++hits;
            if (s > max_steps) max_steps = s;
            checksum += s;
        }
        checksum %= MOD;
    } else if (!strcmp(mode, "guided")) {
        #pragma omp parallel for reduction(max:max_steps) reduction(+:checksum,hits) schedule(guided)
        for (uint64_t i = 1; i <= N; ++i) {
            uint32_t s = collatz_steps(i);
            if (s > 100) ++hits;
            if (s > max_steps) max_steps = s;
            checksum += s;
        }
        checksum %= MOD;
    } else { // static baseline
        #pragma omp parallel for reduction(max:max_steps) reduction(+:checksum,hits) schedule(static)
        for (uint64_t i = 1; i <= N; ++i) {
            uint32_t s = collatz_steps(i);
            if (s > 100) ++hits;
            if (s > max_steps) max_steps = s;
            checksum += s;
        }
        checksum %= MOD;
    }
    *elapsed = omp_get_wtime() - t0;
    return (Result){max_steps, checksum, hits};
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <mode> <threads> [chunk]\n", argv[0]);
        return 2;
    }
    int threads = atoi(argv[2]);
    int chunk = argc > 3 ? atoi(argv[3]) : 0;
    if (threads < 1 || threads > MAX_THREADS || chunk < 0) return 2;
    double seconds;
    Result r = run_kernel(argv[1], threads, chunk, &seconds);
    printf("mode=%s,threads=%d,chunk=%d,seconds=%.9f,max_steps=%u,checksum=%" PRIu64 ",hits=%" PRIu64 "\n",
           argv[1], threads, chunk, seconds, r.max_steps, r.checksum, r.hits);
    return 0;
}
