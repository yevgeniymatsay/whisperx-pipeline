# exp_v8d_boundary1: Truth vs Predicted Call Boundaries (side-by-side)

Bucket (pred): `rezora-whisperx-us-east-1-864981718771`
Predictions: `s3://rezora-whisperx-us-east-1-864981718771/call_segmenter/predictions/exp_v8d_boundary1_20260206_004443_thr70_off60_gap30_gmean_p20_min10/`
Labels: `s3://rezora-whisperx-us-east-1-864981718771/labeling/corrected_boundaries/v1/`

Model git sha: `fff31b7b6f0e`
Decode: `threshold` (thr_on=0.7, thr_off=0.6, gap_merge_s=30.0, gap_stat=mean, gap_min_p=0.2, min_seg_s=10.0)

Format:
- Truth boundaries are your manual call labels.
- Pred boundaries are model outputs (post-processed) with mean probability `p` across the segment.

## Top 10 by Absolute Call-Count Delta

| video_id | truth_calls | pred_calls | delta | time_f1 | seg_f1 |
|---|---:|---:|---:|---:|---:|
| `RtOncKiOT48` | 2 | 12 | +10 | 0.705 | 0.000 |
| `Qa-ppZFUp0g` | 27 | 18 | -9 | 0.952 | 0.711 |
| `DEt3IRqqUVs` | 12 | 5 | -7 | 0.978 | 0.235 |
| `FkgGv2iMjEo` | 8 | 15 | +7 | 0.990 | 0.435 |
| `9g2gwfWIbUk` | 2 | 8 | +6 | 0.985 | 0.000 |
| `pCcAlwOYYx8` | 1 | 7 | +6 | 0.867 | 0.000 |
| `RL6Y5qig0Wg` | 7 | 12 | +5 | 0.984 | 0.737 |
| `pgYGE9jmNPA` | 1 | 6 | +5 | 0.956 | 0.000 |
| `4ipwbOJRMck` | 1 | 5 | +4 | 0.971 | 0.000 |
| `Krnsw9WZtRA` | 8 | 4 | -4 | 0.895 | 0.667 |

## Top 10 Worst by Segment IoU-F1

| video_id | truth_calls | pred_calls | time_f1 | seg_f1 | unmatched_pred | unmatched_truth |
|---|---:|---:|---:|---:|---:|---:|
| `-POaWp9_UaM` | 4 | 2 | 0.991 | 0.000 | 2 | 4 |
| `4ipwbOJRMck` | 1 | 5 | 0.971 | 0.000 | 5 | 1 |
| `9g2gwfWIbUk` | 2 | 8 | 0.985 | 0.000 | 8 | 2 |
| `PnTbJdhNbPk` | 1 | 2 | 0.711 | 0.000 | 2 | 1 |
| `RtOncKiOT48` | 2 | 12 | 0.705 | 0.000 | 12 | 2 |
| `ZjhxWa0N4II` | 3 | 1 | 0.945 | 0.000 | 1 | 3 |
| `bCxvXMhbFls` | 1 | 1 | 0.055 | 0.000 | 1 | 1 |
| `pCcAlwOYYx8` | 1 | 7 | 0.867 | 0.000 | 7 | 1 |
| `pgYGE9jmNPA` | 1 | 6 | 0.956 | 0.000 | 6 | 1 |
| `DEt3IRqqUVs` | 12 | 5 | 0.978 | 0.235 | 3 | 10 |

## Counts Summary (all videos)

| video_id | truth_calls | pred_calls |
|---|---:|---:|
| `-POaWp9_UaM` | 4 | 2 |
| `-UzXRV8nHn4` | 1 | 1 |
| `15QzruVINDc` | 0 | 0 |
| `4OweikRF7bg` | 0 | 0 |
| `4ipwbOJRMck` | 1 | 5 |
| `6L-8f1eYOOY` | 1 | 1 |
| `6UWzVyNtbS0` | 1 | 1 |
| `9g2gwfWIbUk` | 2 | 8 |
| `CfMJ01KP_ns` | 6 | 8 |
| `D4uiHjHW4AU` | 19 | 16 |
| `DEt3IRqqUVs` | 12 | 5 |
| `FkgGv2iMjEo` | 8 | 15 |
| `FotSodGIVM8` | 2 | 2 |
| `GcjtU_V-mW4` | 2 | 3 |
| `JT1XSpK1pkM` | 2 | 2 |
| `Krnsw9WZtRA` | 8 | 4 |
| `MPHdNIiB_8E` | 0 | 0 |
| `Ox_LARRQmlw` | 1 | 1 |
| `PnTbJdhNbPk` | 1 | 2 |
| `Qa-ppZFUp0g` | 27 | 18 |
| `RL6Y5qig0Wg` | 7 | 12 |
| `RtOncKiOT48` | 2 | 12 |
| `S8yFUyD_JXU` | 1 | 2 |
| `UWLJ81ezcpU` | 7 | 10 |
| `Uytq1t3zAz8` | 1 | 1 |
| `VXOSVmvwV0c` | 3 | 4 |
| `VZwqygti-nc` | 4 | 3 |
| `ZjhxWa0N4II` | 3 | 1 |
| `aW8jAYnvqyI` | 0 | 0 |
| `bCxvXMhbFls` | 1 | 1 |
| `bPv8JzD_bMI` | 4 | 7 |
| `ci-FdcWiJiA` | 7 | 9 |
| `dZWKwNqDXIo` | 5 | 9 |
| `dyFaLq3kfHs` | 1 | 2 |
| `gDdoZC9Nhgk` | 7 | 11 |
| `hYjbrxLXJIU` | 0 | 0 |
| `hs44BGJtOwg` | 0 | 0 |
| `jvFLW5EClgk` | 0 | 0 |
| `lioFo9pz9x8` | 1 | 2 |
| `m2z1tgfv8Kc` | 0 | 0 |
| `mxCrbMSfup4` | 3 | 7 |
| `nNA0XQO_7jQ` | 0 | 0 |
| `pCcAlwOYYx8` | 1 | 7 |
| `pgYGE9jmNPA` | 1 | 6 |
| `pjf5uhOMcTc` | 0 | 0 |

## Detailed Boundaries

### -POaWp9_UaM
Truth calls: **4**; Pred calls: **2**; time_f1=0.991; seg_f1=0.000

Truth boundaries:
- 68.724 → 214.650
- 216.061 → 352.478
- 352.592 → 404.942
- 406.142 → 683.371

Pred boundaries (mean_p):
- 76.000 → 647.500 (p=0.6917)
- 647.500 → 683.000 (p=0.5302)

### -UzXRV8nHn4
Truth calls: **1**; Pred calls: **1**; time_f1=0.997; seg_f1=1.000

Truth boundaries:
- 181.119 → 430.160

Pred boundaries (mean_p):
- 182.500 → 430.500 (p=0.9099)

### 15QzruVINDc
Truth calls: **0**; Pred calls: **0**; time_f1=1.000; seg_f1=1.000

Truth boundaries:
- (none)

Pred boundaries (mean_p):
- (none)

### 4OweikRF7bg
Truth calls: **0**; Pred calls: **0**; time_f1=1.000; seg_f1=1.000

Truth boundaries:
- (none)

Pred boundaries (mean_p):
- (none)

### 4ipwbOJRMck
Truth calls: **1**; Pred calls: **5**; time_f1=0.971; seg_f1=0.000

Truth boundaries:
- 75.044 → 2535.511

Pred boundaries (mean_p):
- 75.000 → 548.500 (p=0.8309)
- 548.500 → 1651.500 (p=0.8200)
- 1694.500 → 2536.000 (p=0.8199)
- 2576.500 → 2628.000 (p=0.3704)
- 2767.500 → 2814.500 (p=0.3236)

### 6L-8f1eYOOY
Truth calls: **1**; Pred calls: **1**; time_f1=1.000; seg_f1=1.000

Truth boundaries:
- 0.000 → 1044.431

Pred boundaries (mean_p):
- 0.500 → 1044.500 (p=0.7404)

### 6UWzVyNtbS0
Truth calls: **1**; Pred calls: **1**; time_f1=0.969; seg_f1=1.000

Truth boundaries:
- 68.833 → 199.138

Pred boundaries (mean_p):
- 72.000 → 194.500 (p=0.4666)

### 9g2gwfWIbUk
Truth calls: **2**; Pred calls: **8**; time_f1=0.985; seg_f1=0.000

Truth boundaries:
- 0.000 → 934.003
- 937.520 → 1955.949

Pred boundaries (mean_p):
- 2.500 → 293.500 (p=0.6997)
- 294.000 → 472.000 (p=0.7204)
- 474.500 → 1054.500 (p=0.7245)
- 1055.000 → 1530.500 (p=0.6997)
- 1531.000 → 1673.500 (p=0.6948)
- 1676.000 → 1704.000 (p=0.5097)
- 1704.500 → 1975.500 (p=0.6049)
- 1979.500 → 2008.000 (p=0.3022)

### CfMJ01KP_ns
Truth calls: **6**; Pred calls: **8**; time_f1=0.976; seg_f1=0.714

Truth boundaries:
- 44.818 → 198.874
- 198.931 → 635.358
- 729.783 → 967.727
- 970.201 → 1538.415
- 1539.348 → 2231.910
- 2232.687 → 2654.263

Pred boundaries (mean_p):
- 51.500 → 260.000 (p=0.7334)
- 261.000 → 635.500 (p=0.7707)
- 734.000 → 1136.500 (p=0.7103)
- 1144.000 → 1212.500 (p=0.5388)
- 1272.500 → 1681.000 (p=0.7300)
- 1716.000 → 1762.500 (p=0.7472)
- 1763.500 → 2163.000 (p=0.7457)
- 2163.000 → 2654.500 (p=0.7146)

### D4uiHjHW4AU
Truth calls: **19**; Pred calls: **16**; time_f1=0.908; seg_f1=0.457

Truth boundaries:
- 261.795 → 347.176
- 449.234 → 611.234
- 718.610 → 749.031
- 778.726 → 799.352
- 827.465 → 973.185
- 1024.933 → 1049.625
- 1067.032 → 1262.681
- 1266.524 → 1767.914
- 1862.660 → 1974.017
- 1997.116 → 2051.211
- 2066.759 → 2329.216
- 2366.712 → 2396.533
- 2490.841 → 2527.035
- 2527.105 → 2599.352
- 2633.427 → 2671.749
- 2687.597 → 2715.656
- 2790.308 → 2817.925
- 2835.464 → 2883.021
- 2884.539 → 2976.891

Pred boundaries (mean_p):
- 268.000 → 338.500 (p=0.3714)
- 339.500 → 363.000 (p=0.3668)
- 433.500 → 599.500 (p=0.5457)
- 605.500 → 628.000 (p=0.3395)
- 718.000 → 975.000 (p=0.5498)
- 1016.000 → 1142.500 (p=0.5676)
- 1147.000 → 1272.000 (p=0.6859)
- 1272.000 → 1473.000 (p=0.6141)
- 1488.000 → 1768.000 (p=0.6606)
- 1867.500 → 1905.000 (p=0.4985)
- 1910.500 → 2329.000 (p=0.5680)
- 2372.000 → 2446.500 (p=0.3810)
- 2494.000 → 2544.000 (p=0.5205)
- 2544.000 → 2604.500 (p=0.4494)
- 2605.000 → 2721.500 (p=0.5299)
- 2759.500 → 2980.000 (p=0.5982)

### DEt3IRqqUVs
Truth calls: **12**; Pred calls: **5**; time_f1=0.978; seg_f1=0.235

Truth boundaries:
- 0.000 → 48.254
- 48.254 → 125.780
- 125.780 → 172.309
- 172.309 → 174.474
- 186.376 → 194.548
- 199.846 → 242.228
- 252.229 → 332.666
- 335.175 → 434.636
- 437.733 → 573.780
- 577.975 → 598.110
- 602.404 → 808.000
- 810.275 → 821.340

Pred boundaries (mean_p):
- 5.000 → 175.500 (p=0.7532)
- 187.500 → 242.500 (p=0.8564)
- 252.000 → 599.000 (p=0.7332)
- 608.000 → 684.000 (p=0.6433)
- 685.000 → 820.000 (p=0.5639)

### FkgGv2iMjEo
Truth calls: **8**; Pred calls: **15**; time_f1=0.990; seg_f1=0.435

Truth boundaries:
- 144.431 → 251.563
- 317.795 → 559.822
- 652.932 → 811.384
- 868.416 → 943.995
- 1035.025 → 1050.799
- 1063.044 → 1097.752
- 1113.517 → 1147.411
- 1157.475 → 3623.703

Pred boundaries (mean_p):
- 145.000 → 252.000 (p=0.5950)
- 317.500 → 559.500 (p=0.6762)
- 654.000 → 811.000 (p=0.6530)
- 868.000 → 884.000 (p=0.6965)
- 884.000 → 943.500 (p=0.7346)
- 1037.500 → 1098.000 (p=0.6671)
- 1115.000 → 1585.000 (p=0.5913)
- 1588.000 → 2357.000 (p=0.6828)
- 2361.000 → 2723.500 (p=0.6866)
- 2723.500 → 2773.000 (p=0.5899)
- 2775.000 → 2789.500 (p=0.6988)
- 2799.000 → 2844.500 (p=0.5955)
- 2845.500 → 3473.000 (p=0.6503)
- 3477.500 → 3553.000 (p=0.5616)
- 3561.500 → 3623.500 (p=0.7374)

### FotSodGIVM8
Truth calls: **2**; Pred calls: **2**; time_f1=0.787; seg_f1=0.500

Truth boundaries:
- 33.524 → 366.326
- 461.912 → 633.082

Pred boundaries (mean_p):
- 38.500 → 336.500 (p=0.7055)
- 337.000 → 366.500 (p=0.4835)

### GcjtU_V-mW4
Truth calls: **2**; Pred calls: **3**; time_f1=0.969; seg_f1=0.800

Truth boundaries:
- 142.589 → 238.604
- 513.423 → 604.887

Pred boundaries (mean_p):
- 145.000 → 239.000 (p=0.5770)
- 513.500 → 532.000 (p=0.3683)
- 540.000 → 604.500 (p=0.5338)

### JT1XSpK1pkM
Truth calls: **2**; Pred calls: **2**; time_f1=0.908; seg_f1=1.000

Truth boundaries:
- 40.397 → 64.726
- 335.183 → 357.848

Pred boundaries (mean_p):
- 47.000 → 65.500 (p=0.6923)
- 335.000 → 358.500 (p=0.7731)

### Krnsw9WZtRA
Truth calls: **8**; Pred calls: **4**; time_f1=0.895; seg_f1=0.667

Truth boundaries:
- 29.249 → 59.838
- 106.285 → 179.954
- 248.622 → 260.551
- 274.427 → 347.140
- 352.796 → 361.676
- 438.124 → 456.197
- 487.946 → 512.572
- 521.384 → 559.848

Pred boundaries (mean_p):
- 29.000 → 61.000 (p=0.6890)
- 108.000 → 179.500 (p=0.5880)
- 248.500 → 344.000 (p=0.4885)
- 487.500 → 559.000 (p=0.6397)

### MPHdNIiB_8E
Truth calls: **0**; Pred calls: **0**; time_f1=1.000; seg_f1=1.000

Truth boundaries:
- (none)

Pred boundaries (mean_p):
- (none)

### Ox_LARRQmlw
Truth calls: **1**; Pred calls: **1**; time_f1=0.999; seg_f1=1.000

Truth boundaries:
- 80.995 → 716.291

Pred boundaries (mean_p):
- 82.000 → 715.500 (p=0.7019)

### PnTbJdhNbPk
Truth calls: **1**; Pred calls: **2**; time_f1=0.711; seg_f1=0.000

Truth boundaries:
- 15.854 → 1384.158

Pred boundaries (mean_p):
- 22.500 → 598.500 (p=0.5747)
- 1204.500 → 1384.500 (p=0.5233)

### Qa-ppZFUp0g
Truth calls: **27**; Pred calls: **18**; time_f1=0.952; seg_f1=0.711

Truth boundaries:
- 62.001 → 100.014
- 102.180 → 204.668
- 219.622 → 319.923
- 343.677 → 488.370
- 525.988 → 605.610
- 676.982 → 712.688
- 744.738 → 791.100
- 801.040 → 820.142
- 820.553 → 843.042
- 851.698 → 870.707
- 871.636 → 884.019
- 884.309 → 912.601
- 916.407 → 1012.770
- 1033.230 → 1062.078
- 1120.383 → 1148.581
- 1186.632 → 1253.272
- 1272.536 → 1287.211
- 1287.211 → 1308.208
- 1331.494 → 1370.050
- 1370.458 → 1384.199
- 1406.510 → 1421.897
- 1424.612 → 1461.092
- 1461.281 → 1471.516
- 1471.629 → 1485.060
- 1575.740 → 1698.958
- 1706.062 → 1770.691
- 1784.437 → 1831.639

Pred boundaries (mean_p):
- 64.500 → 317.000 (p=0.5742)
- 344.000 → 489.000 (p=0.6893)
- 527.500 → 600.500 (p=0.5165)
- 679.000 → 710.500 (p=0.6489)
- 751.500 → 787.500 (p=0.5076)
- 801.000 → 842.500 (p=0.6659)
- 853.000 → 1010.500 (p=0.5167)
- 1042.000 → 1061.500 (p=0.4519)
- 1122.500 → 1149.000 (p=0.6111)
- 1189.500 → 1251.500 (p=0.5978)
- 1278.000 → 1307.000 (p=0.7203)
- 1331.500 → 1385.000 (p=0.6894)
- 1396.000 → 1421.500 (p=0.3609)
- 1424.500 → 1483.500 (p=0.6512)
- 1577.500 → 1699.000 (p=0.6888)
- 1710.000 → 1769.500 (p=0.7419)
- 1785.000 → 1828.500 (p=0.5318)
- 1925.500 → 1943.500 (p=0.3024)

### RL6Y5qig0Wg
Truth calls: **7**; Pred calls: **12**; time_f1=0.984; seg_f1=0.737

Truth boundaries:
- 88.710 → 222.038
- 455.360 → 559.472
- 638.357 → 675.494
- 766.126 → 779.070
- 893.717 → 1059.770
- 1157.827 → 1225.158
- 1434.781 → 1602.009

Pred boundaries (mean_p):
- 89.000 → 104.500 (p=0.6835)
- 105.500 → 177.500 (p=0.6618)
- 182.000 → 220.500 (p=0.5618)
- 458.000 → 560.500 (p=0.6399)
- 638.000 → 675.500 (p=0.7725)
- 765.500 → 777.500 (p=0.8276)
- 896.000 → 1009.000 (p=0.7323)
- 1010.500 → 1060.000 (p=0.7006)
- 1159.000 → 1225.000 (p=0.5958)
- 1434.500 → 1448.500 (p=0.7601)
- 1450.000 → 1540.000 (p=0.6314)
- 1540.500 → 1602.500 (p=0.5124)

### RtOncKiOT48
Truth calls: **2**; Pred calls: **12**; time_f1=0.705; seg_f1=0.000

Truth boundaries:
- 98.156 → 484.037
- 563.603 → 1483.472

Pred boundaries (mean_p):
- 1.500 → 28.000 (p=0.5524)
- 105.500 → 194.000 (p=0.4974)
- 234.500 → 287.000 (p=0.5147)
- 301.000 → 359.000 (p=0.4641)
- 362.000 → 604.500 (p=0.5889)
- 604.500 → 620.000 (p=0.7526)
- 620.500 → 816.000 (p=0.6665)
- 861.500 → 952.000 (p=0.4836)
- 952.500 → 979.500 (p=0.6334)
- 980.000 → 996.000 (p=0.8500)
- 996.500 → 1028.500 (p=0.6962)
- 1037.000 → 1066.500 (p=0.7504)

### S8yFUyD_JXU
Truth calls: **1**; Pred calls: **2**; time_f1=0.835; seg_f1=0.667

Truth boundaries:
- 77.611 → 443.224

Pred boundaries (mean_p):
- 28.500 → 90.500 (p=0.3904)
- 144.500 → 463.500 (p=0.4883)

### UWLJ81ezcpU
Truth calls: **7**; Pred calls: **10**; time_f1=0.887; seg_f1=0.706

Truth boundaries:
- 387.932 → 527.672
- 657.982 → 692.946
- 736.642 → 760.148
- 795.336 → 889.925
- 1033.707 → 1139.590
- 1199.640 → 1270.695
- 1313.627 → 1427.264

Pred boundaries (mean_p):
- 390.500 → 469.000 (p=0.4468)
- 503.500 → 541.500 (p=0.4937)
- 662.000 → 690.500 (p=0.5505)
- 745.500 → 757.500 (p=0.6799)
- 797.500 → 889.500 (p=0.5951)
- 1033.000 → 1080.000 (p=0.4795)
- 1080.000 → 1146.500 (p=0.6332)
- 1200.500 → 1211.000 (p=0.4734)
- 1248.000 → 1268.500 (p=0.6362)
- 1314.000 → 1423.500 (p=0.5341)

### Uytq1t3zAz8
Truth calls: **1**; Pred calls: **1**; time_f1=0.999; seg_f1=1.000

Truth boundaries:
- 8.884 → 1098.453

Pred boundaries (mean_p):
- 9.500 → 1099.500 (p=0.6080)

### VXOSVmvwV0c
Truth calls: **3**; Pred calls: **4**; time_f1=0.966; seg_f1=0.857

Truth boundaries:
- 37.446 → 173.066
- 181.256 → 267.866
- 274.943 → 393.007

Pred boundaries (mean_p):
- 43.500 → 150.500 (p=0.5000)
- 150.500 → 168.000 (p=0.6024)
- 168.500 → 233.500 (p=0.4881)
- 234.500 → 392.500 (p=0.5158)

### VZwqygti-nc
Truth calls: **4**; Pred calls: **3**; time_f1=0.924; seg_f1=0.571

Truth boundaries:
- 40.401 → 83.021
- 93.259 → 140.640
- 167.200 → 197.157
- 201.827 → 332.638

Pred boundaries (mean_p):
- 43.500 → 81.500 (p=0.5450)
- 92.500 → 257.000 (p=0.5289)
- 259.000 → 331.500 (p=0.5270)

### ZjhxWa0N4II
Truth calls: **3**; Pred calls: **1**; time_f1=0.945; seg_f1=0.000

Truth boundaries:
- 10.255 → 66.691
- 72.149 → 144.985
- 150.111 → 305.392

Pred boundaries (mean_p):
- 11.500 → 326.500 (p=0.7752)

### aW8jAYnvqyI
Truth calls: **0**; Pred calls: **0**; time_f1=1.000; seg_f1=1.000

Truth boundaries:
- (none)

Pred boundaries (mean_p):
- (none)

### bCxvXMhbFls
Truth calls: **1**; Pred calls: **1**; time_f1=0.055; seg_f1=0.000

Truth boundaries:
- 43.068 → 713.823

Pred boundaries (mean_p):
- 43.000 → 62.000 (p=0.4410)

### bPv8JzD_bMI
Truth calls: **4**; Pred calls: **7**; time_f1=0.788; seg_f1=0.364

Truth boundaries:
- 86.956 → 111.865
- 530.643 → 569.233
- 728.712 → 847.820
- 926.350 → 1034.779

Pred boundaries (mean_p):
- 87.000 → 97.500 (p=0.4584)
- 100.000 → 136.000 (p=0.5091)
- 485.500 → 565.000 (p=0.3842)
- 614.500 → 648.500 (p=0.8291)
- 739.000 → 767.000 (p=0.3766)
- 781.500 → 845.000 (p=0.5700)
- 926.000 → 1034.500 (p=0.6935)

### ci-FdcWiJiA
Truth calls: **7**; Pred calls: **9**; time_f1=0.806; seg_f1=0.500

Truth boundaries:
- 0.000 → 82.192
- 257.803 → 276.974
- 389.934 → 435.465
- 595.341 → 616.391
- 1020.462 → 1124.976
- 1752.450 → 1772.900
- 2034.918 → 2171.015

Pred boundaries (mean_p):
- 6.000 → 73.500 (p=0.5457)
- 75.000 → 86.000 (p=0.3998)
- 257.500 → 299.000 (p=0.3307)
- 300.000 → 319.000 (p=0.4540)
- 389.500 → 413.500 (p=0.3913)
- 892.000 → 902.000 (p=0.4744)
- 1026.500 → 1094.500 (p=0.5198)
- 1095.500 → 1132.500 (p=0.3542)
- 2054.000 → 2171.000 (p=0.5224)

### dZWKwNqDXIo
Truth calls: **5**; Pred calls: **9**; time_f1=0.998; seg_f1=0.714

Truth boundaries:
- 60.993 → 158.577
- 231.489 → 498.542
- 553.184 → 807.714
- 840.897 → 1031.735
- 1080.048 → 1727.524

Pred boundaries (mean_p):
- 61.500 → 159.000 (p=0.7341)
- 232.000 → 499.000 (p=0.8565)
- 553.000 → 808.000 (p=0.7122)
- 841.500 → 854.500 (p=0.4830)
- 856.000 → 875.000 (p=0.5674)
- 875.000 → 1030.500 (p=0.6834)
- 1080.000 → 1222.500 (p=0.6605)
- 1223.000 → 1682.500 (p=0.7773)
- 1682.500 → 1727.500 (p=0.8534)

### dyFaLq3kfHs
Truth calls: **1**; Pred calls: **2**; time_f1=0.959; seg_f1=0.667

Truth boundaries:
- 73.850 → 499.583

Pred boundaries (mean_p):
- 74.000 → 94.000 (p=0.6483)
- 127.000 → 500.000 (p=0.7206)

### gDdoZC9Nhgk
Truth calls: **7**; Pred calls: **11**; time_f1=0.960; seg_f1=0.333

Truth boundaries:
- 96.069 → 150.692
- 257.311 → 466.293
- 468.043 → 507.117
- 512.702 → 549.139
- 697.043 → 902.660
- 908.561 → 964.391
- 978.786 → 1248.417

Pred boundaries (mean_p):
- 97.000 → 150.500 (p=0.7369)
- 257.000 → 279.500 (p=0.5117)
- 280.000 → 371.000 (p=0.4070)
- 377.500 → 549.500 (p=0.5925)
- 696.500 → 769.000 (p=0.6510)
- 780.000 → 817.000 (p=0.3778)
- 821.000 → 895.000 (p=0.5948)
- 908.500 → 932.000 (p=0.6670)
- 932.000 → 965.000 (p=0.7322)
- 979.000 → 1123.500 (p=0.5445)
- 1150.500 → 1248.500 (p=0.6414)

### hYjbrxLXJIU
Truth calls: **0**; Pred calls: **0**; time_f1=1.000; seg_f1=1.000

Truth boundaries:
- (none)

Pred boundaries (mean_p):
- (none)

### hs44BGJtOwg
Truth calls: **0**; Pred calls: **0**; time_f1=1.000; seg_f1=1.000

Truth boundaries:
- (none)

Pred boundaries (mean_p):
- (none)

### jvFLW5EClgk
Truth calls: **0**; Pred calls: **0**; time_f1=1.000; seg_f1=1.000

Truth boundaries:
- (none)

Pred boundaries (mean_p):
- (none)

### lioFo9pz9x8
Truth calls: **1**; Pred calls: **2**; time_f1=0.981; seg_f1=0.667

Truth boundaries:
- 55.469 → 1045.232

Pred boundaries (mean_p):
- 58.500 → 318.500 (p=0.6658)
- 351.000 → 1046.000 (p=0.6850)

### m2z1tgfv8Kc
Truth calls: **0**; Pred calls: **0**; time_f1=1.000; seg_f1=1.000

Truth boundaries:
- (none)

Pred boundaries (mean_p):
- (none)

### mxCrbMSfup4
Truth calls: **3**; Pred calls: **7**; time_f1=0.916; seg_f1=0.600

Truth boundaries:
- 40.697 → 707.536
- 745.894 → 763.517
- 813.441 → 1007.483

Pred boundaries (mean_p):
- 44.500 → 503.000 (p=0.6055)
- 504.000 → 536.000 (p=0.3266)
- 559.500 → 636.000 (p=0.5737)
- 684.000 → 706.500 (p=0.5924)
- 749.000 → 762.000 (p=0.6735)
- 813.000 → 856.500 (p=0.5905)
- 899.000 → 1020.500 (p=0.5336)

### nNA0XQO_7jQ
Truth calls: **0**; Pred calls: **0**; time_f1=1.000; seg_f1=1.000

Truth boundaries:
- (none)

Pred boundaries (mean_p):
- (none)

### pCcAlwOYYx8
Truth calls: **1**; Pred calls: **7**; time_f1=0.867; seg_f1=0.000

Truth boundaries:
- 19.288 → 1095.080

Pred boundaries (mean_p):
- 19.000 → 89.500 (p=0.5732)
- 108.500 → 162.500 (p=0.6365)
- 165.000 → 538.500 (p=0.6191)
- 611.000 → 642.500 (p=0.4891)
- 768.000 → 784.500 (p=0.5741)
- 815.500 → 980.500 (p=0.5506)
- 982.500 → 1095.500 (p=0.5945)

### pgYGE9jmNPA
Truth calls: **1**; Pred calls: **6**; time_f1=0.956; seg_f1=0.000

Truth boundaries:
- 43.409 → 909.003

Pred boundaries (mean_p):
- 43.000 → 474.000 (p=0.7331)
- 508.500 → 607.000 (p=0.4708)
- 609.500 → 620.500 (p=0.4466)
- 621.000 → 655.000 (p=0.3995)
- 656.500 → 909.000 (p=0.6261)
- 949.500 → 986.000 (p=0.3272)

### pjf5uhOMcTc
Truth calls: **0**; Pred calls: **0**; time_f1=1.000; seg_f1=1.000

Truth boundaries:
- (none)

Pred boundaries (mean_p):
- (none)

