# Lab 1 — Fork-Join Model (Java)

## Requirements

- OpenJDK 17 or newer
- Terminal

## Compile and collect timing sweep

From this folder:

```bash
javac ForkJoinLab1.java
java ForkJoinLab1 0 > lab1-results.csv 2> lab1-threads.txt
```

`lab1-results.csv` contains five measurements for each team size P = 1, 2, 4, 8, 16, 32, 64, after two warm-up runs. Use these as raw observations and calculate the mean for each P in your report.

## Repeat the thread-order observation ten times

```bash
for i in {1..10}; do
  java ForkJoinLab1 0
 done > lab1-10-runs.txt 2>&1
```

Check whether observed worker order differs between runs. The program uses a Java ForkJoinPool, so the displayed rank is a logical work item, not a guaranteed permanent OpenMP-style thread ID.

## CPU load observation

Run a larger workload, then observe Activity Monitor → CPU while the process is active:

```bash
java ForkJoinLab1 10000000
```

Record the workload argument and what you observe. CPU usage depends on the machine and current system load. Do not copy example or other-machine readings.

## Submission

Include the Java source, this README, your own raw CSV/log files, and the completed report. The report template is `Lab1-Report.md`. Convert the completed report to PDF/DOCX if your course platform requires that format.
