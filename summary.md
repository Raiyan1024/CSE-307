# Experiment Summary
Held-out classifier accuracy: 0.729
Mean confidence when correct: 0.849
Mean confidence when wrong: 0.652

## Total seek distance on shifted timeline
- FCFS: 43594
- SCAN: 4974
- C-SCAN: 6383
- SSTF: 4096
- Learned selector: 3862
- Oracle best: 3856

## Decision tree

```text
|--- max <= 198.50
|   |--- min <= 70.50
|   |   |--- head <= 1.50
|   |   |   |--- class: SCAN
|   |   |--- head >  1.50
|   |   |   |--- head <= 102.50
|   |   |   |   |--- min <= 31.50
|   |   |   |   |   |--- class: SSTF
|   |   |   |   |--- min >  31.50
|   |   |   |   |   |--- class: SCAN
|   |   |   |--- head >  102.50
|   |   |   |   |--- range <= 162.50
|   |   |   |   |   |--- class: SSTF
|   |   |   |   |--- range >  162.50
|   |   |   |   |   |--- class: SCAN
|   |--- min >  70.50
|   |   |--- head <= 99.50
|   |   |   |--- max <= 152.50
|   |   |   |   |--- class: SCAN
|   |   |   |--- max >  152.50
|   |   |   |   |--- class: SCAN
|   |   |--- head >  99.50
|   |   |   |--- max <= 180.00
|   |   |   |   |--- unique_ratio <= 0.56
|   |   |   |   |   |--- class: SSTF
|   |   |   |   |--- unique_ratio >  0.56
|   |   |   |   |   |--- class: SSTF
|   |   |   |--- max >  180.00
|   |   |   |   |--- distance_head_to_mean <= 25.99
|   |   |   |   |   |--- class: SSTF
|   |   |   |   |--- distance_head_to_mean >  25.99
|   |   |   |   |   |--- class: SCAN
|--- max >  198.50
|   |--- head <= 100.50
|   |   |--- std <= 59.84
|   |   |   |--- head <= 42.50
|   |   |   |   |--- head <= 24.00
|   |   |   |   |   |--- class: SCAN
|   |   |   |   |--- head >  24.00
|   |   |   |   |   |--- class: SSTF
|   |   |   |--- head >  42.50
|   |   |   |   |--- distance_head_to_mean <= 38.71
|   |   |   |   |   |--- class: SCAN
|   |   |   |   |--- distance_head_to_mean >  38.71
|   |   |   |   |   |--- class: SCAN
|   |   |--- std >  59.84
|   |   |   |--- unique_ratio <= 0.89
|   |   |   |   |--- class: SSTF
|   |   |   |--- unique_ratio >  0.89
|   |   |   |   |--- class: SSTF
|   |--- head >  100.50
|   |   |--- distance_head_to_mean <= 8.62
|   |   |   |--- class: SCAN
|   |   |--- distance_head_to_mean >  8.62
|   |   |   |--- class: SCAN

```

## Classification report

```text
              precision    recall  f1-score   support

        FCFS       0.00      0.00      0.00         1
        SCAN       0.61      0.81      0.70       134
        SSTF       0.85      0.68      0.75       215

    accuracy                           0.73       350
   macro avg       0.49      0.50      0.48       350
weighted avg       0.76      0.73      0.73       350

```
