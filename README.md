# CSE-307 Term Paper: Disk Scheduling with a Learned Scheduler Selector

**Selected Track:** Track 2 — Disk Scheduling: Learned Scheduler Selector

The project compares classical disk scheduling algorithms with a lightweight learned scheduler selector. The main goal is to observe how different scheduling algorithms behave when the disk request pattern changes over time.
]
## Project Overview

Disk scheduling is an important Operating Systems concept because the order of serving disk I/O requests affects total head movement and overall performance. A scheduler that performs well for one workload may not perform well for another workload.

In this project, four classical disk scheduling algorithms are implemented and tested:

- FCFS
- SCAN
- C-SCAN
- SSTF

A simple machine-learning classifier is then used as a scheduler selector. Instead of replacing the classical algorithms, the learned component predicts which scheduler is likely to perform best for a given request window.

---
## Implemented Algorithms

| Algorithm | Description |
|---|---|
| FCFS | Serves requests in the same order in which they arrive. |
| SCAN | Moves the disk head in one direction and reverses after reaching the disk end. |
| C-SCAN | Moves in one direction only and wraps around after reaching the disk end. |
| SSTF | Selects the pending request closest to the current disk head position. |

The performance metric used in this project is **total seek distance**. Lower total seek distance means better performance.

---

## Learned Scheduler Selector

The learned component is a `DecisionTreeClassifier` from scikit-learn.

For each request window, the classifier uses statistical features such as:

- Initial head position
- Mean request position
- Standard deviation of requests
- Minimum and maximum requested cylinder
- Request range
- Mean absolute difference between consecutive requests
- Locality of requests
- Monotonicity of the request sequence
- Unique request ratio
- Distance between head position and mean request position

The classifier is trained using labels generated from the classical algorithms. For each training window, all four schedulers are executed, and the scheduler with the lowest seek distance becomes the label for that sample.

---

## Workload Design

The simulator generates synthetic disk request windows using different workload patterns:

1. **Sequential workload  
 Requests follow a mostly ordered pattern and show locality.

2. **Random workload  
   Requests are distributed randomly across the disk.

3. **Bursty workload  
   Requests are clustered around a few disk regions.

4. **Mixed workload 
   A combination of sequential behavior and random or bursty behavior.

The final evaluation timeline includes a deliberate workload shift:

```text
Sequential phase  →  Random phase  →  Bursty phase
```

This shift is used to test whether a single fixed scheduler remains effective when the request pattern changes.

---

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── src/
│   └── simulate.py
├── results/
   ├── training_dataset.csv
   ├── shifted_timeline_results.csv
   ├── summary.md
  └── figures/
       ├── total_seek_comparison.png
       ├── timeline_seek.png
       └── confidence_timeline.png

```

---

## Setup Instructions

Install the required Python packages:

```bash
python3 -m pip install -r requirements.txt
```

Required packages:

- numpy
- pandas
- matplotlib
- scikit-learn

---

## How to Run the Experiment

From the repository root, run:

```bash
python3 src/simulate.py --out results
```

This command will:

1. Generate synthetic disk request workloads.
2. Run FCFS, SCAN, C-SCAN, and SSTF.
3. Train the decision-tree scheduler selector.
4. Evaluate all methods on the shifted workload timeline.
5. Save raw results and summary files inside the `results/` folder.

A full command with optional parameters is shown below:

```bash
python3 src/simulate.py \
  --samples-per-kind 350 \
  --requests-per-window 60 \
  --timeline-windows 18 \
  --seed 307 \
  --out results
```

---

## Result Summary

The included experiment run produced the following classifier results:

| Metric | Value |

| Held-out classifier accuracy | 0.729 |
| Mean confidence when prediction was correct | 0.849 |
| Mean confidence when prediction was wrong | 0.652 |

Total seek distance on the shifted workload timeline:

| Method | Total seek distance |
|---|---:|
| FCFS | 43,594 |
| SCAN | 4,974 |
| C-SCAN | 6,383 |
| SSTF | 4,096 |
| Learned selector | 3,862 |
| Oracle best | 3,856 |

The learned selector achieved the lowest seek distance among the practical methods and performed very close to the oracle best selector.

---

## Key Findings

- FCFS performed poorly because it follows arrival order without considering seek distance.
- SSTF performed well for clustered and locality-heavy requests.
- SCAN and C-SCAN reduced unnecessary back-and-forth movement but sometimes added extra travel to the disk boundary.
- The learned selector performed well because it adapted its scheduler choice according to the request-window characteristics.
- The confidence score was generally higher when the classifier prediction was correct, but it was not perfectly calibrated.

---

## Notes on Preparation

For the coding part, the assignment brief was followed closely. Standard documentation and publicly available programming references, including GitHub examples of common Python project structure and standard algorithm implementation style, were consulted only for guidance. The final implementation, experiments, outputs, and analysis in this repository are specific to this project.
