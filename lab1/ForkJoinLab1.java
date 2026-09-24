import java.util.concurrent.ForkJoinPool;
import java.util.Locale;
import java.util.stream.IntStream;

/**
 * Lab 1 starter: fork-join team emulation, timing sweep, and optional CPU load.
 * Run with: java ForkJoinLab1 [workIterationsPerTask]
 * Use 0 for the team-creation sweep; use a larger value for the CPU-load observation.
 */
public class ForkJoinLab1 {
    private static volatile double sink;
    private static final int[] TEAM_SIZES = {1, 2, 4, 8, 16, 32, 64};
    private static final int WARMUP_RUNS = 2;
    private static final int MEASURED_RUNS = 5;

    private static void runTeam(int teamSize, long workIterations, boolean printWorkers) {
        ForkJoinPool pool = new ForkJoinPool(teamSize);
        double[] partialResults = new double[teamSize];
        try {
            pool.submit(() -> IntStream.range(0, teamSize).parallel().forEach(rank -> {
                if (printWorkers) {
                    Thread t = Thread.currentThread();
                    System.err.printf("rank=%d/%d thread=%s id=%d%n",
                            rank, teamSize, t.getName(), t.getId());
                }
                double local = rank + 1.0;
                for (long i = 0; i < workIterations; i++) {
                    local = Math.sqrt(local + (i & 7) + 1.0);
                }
                partialResults[rank] = local;
            })).join();
        } finally {
            pool.shutdown(); // join above is the end-of-region synchronization point
        }
        double total = 0.0;
        for (double value : partialResults) total += value;
        sink = total; // Keeps the computation observable; not a benchmark result.
    }

    private static long timeOneRun(int teamSize, long workIterations) {
        long start = System.nanoTime();
        runTeam(teamSize, workIterations, false);
        return System.nanoTime() - start;
    }

    public static void main(String[] args) {
        long workIterations = args.length == 0 ? 0 : Long.parseLong(args[0]);
        System.out.println("team_size,trial,work_iterations,elapsed_ms");
        for (int p : TEAM_SIZES) {
            for (int i = 0; i < WARMUP_RUNS; i++) timeOneRun(p, workIterations);
            for (int trial = 1; trial <= MEASURED_RUNS; trial++) {
                long nanos = timeOneRun(p, workIterations);
                System.out.printf(Locale.US, "%d,%d,%d,%.3f%n", p, trial, workIterations, nanos / 1_000_000.0);
            }
        }

        System.err.println("\nThread-order observation run (P=4):");
        runTeam(4, 0, true);
        System.err.println("\nRepeat the program ten times and save stdout if documenting run-to-run ordering.");
    }
}
