# exp_v8b_mfcc: Truth vs Predicted Call Boundaries (side-by-side)

Bucket (pred): `rezora-whisperx-us-east-1-864981718771`
Predictions: `s3://rezora-whisperx-us-east-1-864981718771/call_segmenter/predictions/exp_v8b_mfcc_20260205_234409_thr55_off45_gap10_gmax_p00_min05/`
Labels: `s3://rezora-whisperx-us-east-1-864981718771/labeling/corrected_boundaries/v1/`

Model git sha: `aad3a21adc31`
Decode: `threshold` (thr_on=0.55, thr_off=0.45000000000000007, gap_merge_s=10.0, gap_stat=max, gap_min_p=0.0, min_seg_s=5.0)

Format:
- Truth boundaries are your manual call labels.
- Pred boundaries are model outputs (post-processed) with mean probability `p` across the segment.

## Top 10 by Absolute Call-Count Delta

| video_id | truth_calls | pred_calls | delta | time_f1 | seg_f1 |
|---|---:|---:|---:|---:|---:|
| `ci-FdcWiJiA` | 7 | 23 | +16 | 0.725 | 0.400 |
| `D4uiHjHW4AU` | 19 | 9 | -10 | 0.857 | 0.214 |
| `DEt3IRqqUVs` | 12 | 2 | -10 | 0.975 | 0.000 |
| `Qa-ppZFUp0g` | 27 | 19 | -8 | 0.944 | 0.565 |
| `UWLJ81ezcpU` | 7 | 13 | +6 | 0.827 | 0.500 |
| `mxCrbMSfup4` | 3 | 8 | +5 | 0.944 | 0.364 |
| `pCcAlwOYYx8` | 1 | 6 | +5 | 0.946 | 0.000 |
| `bPv8JzD_bMI` | 4 | 8 | +4 | 0.735 | 0.167 |
| `pgYGE9jmNPA` | 1 | 5 | +4 | 0.975 | 0.333 |
| `-POaWp9_UaM` | 4 | 1 | -3 | 0.992 | 0.000 |

## Top 10 Worst by Segment IoU-F1

| video_id | truth_calls | pred_calls | time_f1 | seg_f1 | unmatched_pred | unmatched_truth |
|---|---:|---:|---:|---:|---:|---:|
| `-POaWp9_UaM` | 4 | 1 | 0.992 | 0.000 | 1 | 4 |
| `DEt3IRqqUVs` | 12 | 2 | 0.975 | 0.000 | 2 | 12 |
| `ZjhxWa0N4II` | 3 | 1 | 0.934 | 0.000 | 1 | 3 |
| `bCxvXMhbFls` | 1 | 1 | 0.081 | 0.000 | 1 | 1 |
| `pCcAlwOYYx8` | 1 | 6 | 0.946 | 0.000 | 6 | 1 |
| `bPv8JzD_bMI` | 4 | 8 | 0.735 | 0.167 | 7 | 3 |
| `D4uiHjHW4AU` | 19 | 9 | 0.857 | 0.214 | 6 | 16 |
| `RtOncKiOT48` | 2 | 5 | 0.733 | 0.286 | 4 | 1 |
| `VXOSVmvwV0c` | 3 | 3 | 0.952 | 0.333 | 2 | 2 |
| `pgYGE9jmNPA` | 1 | 5 | 0.975 | 0.333 | 4 | 0 |

## Counts Summary (all videos)

| video_id | truth_calls | pred_calls |
|---|---:|---:|
| `-POaWp9_UaM` | 4 | 1 |
| `-UzXRV8nHn4` | 1 | 3 |
| `15QzruVINDc` | 0 | 0 |
| `4OweikRF7bg` | 0 | 0 |
| `4ipwbOJRMck` | 1 | 1 |
| `6L-8f1eYOOY` | 1 | 1 |
| `6UWzVyNtbS0` | 1 | 2 |
| `9g2gwfWIbUk` | 2 | 1 |
| `CfMJ01KP_ns` | 6 | 3 |
| `D4uiHjHW4AU` | 19 | 9 |
| `DEt3IRqqUVs` | 12 | 2 |
| `FkgGv2iMjEo` | 8 | 7 |
| `FotSodGIVM8` | 2 | 1 |
| `GcjtU_V-mW4` | 2 | 3 |
| `JT1XSpK1pkM` | 2 | 2 |
| `Krnsw9WZtRA` | 8 | 7 |
| `MPHdNIiB_8E` | 0 | 0 |
| `Ox_LARRQmlw` | 1 | 1 |
| `PnTbJdhNbPk` | 1 | 1 |
| `Qa-ppZFUp0g` | 27 | 19 |
| `RL6Y5qig0Wg` | 7 | 7 |
| `RtOncKiOT48` | 2 | 5 |
| `S8yFUyD_JXU` | 1 | 4 |
| `UWLJ81ezcpU` | 7 | 13 |
| `Uytq1t3zAz8` | 1 | 2 |
| `VXOSVmvwV0c` | 3 | 3 |
| `VZwqygti-nc` | 4 | 4 |
| `ZjhxWa0N4II` | 3 | 1 |
| `aW8jAYnvqyI` | 0 | 0 |
| `bCxvXMhbFls` | 1 | 1 |
| `bPv8JzD_bMI` | 4 | 8 |
| `ci-FdcWiJiA` | 7 | 23 |
| `dZWKwNqDXIo` | 5 | 5 |
| `dyFaLq3kfHs` | 1 | 1 |
| `gDdoZC9Nhgk` | 7 | 5 |
| `hYjbrxLXJIU` | 0 | 0 |
| `hs44BGJtOwg` | 0 | 0 |
| `jvFLW5EClgk` | 0 | 0 |
| `lioFo9pz9x8` | 1 | 1 |
| `m2z1tgfv8Kc` | 0 | 0 |
| `mxCrbMSfup4` | 3 | 8 |
| `nNA0XQO_7jQ` | 0 | 0 |
| `pCcAlwOYYx8` | 1 | 6 |
| `pgYGE9jmNPA` | 1 | 5 |
| `pjf5uhOMcTc` | 0 | 0 |

## Detailed Boundaries

### -POaWp9_UaM
Truth calls: **4**; Pred calls: **1**; time_f1=0.992; seg_f1=0.000

Truth boundaries:
- 68.724 → 214.650
- 216.061 → 352.478
- 352.592 → 404.942
- 406.142 → 683.371

Pred boundaries (mean_p):
- 75.000 → 684.000 (p=0.7830)

### -UzXRV8nHn4
Truth calls: **1**; Pred calls: **3**; time_f1=0.970; seg_f1=0.500

Truth boundaries:
- 181.119 → 430.160

Pred boundaries (mean_p):
- 3.000 → 11.000 (p=0.2207)
- 22.000 → 28.000 (p=0.4915)
- 180.500 → 431.000 (p=0.9405)

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
Truth calls: **1**; Pred calls: **1**; time_f1=1.000; seg_f1=1.000

Truth boundaries:
- 75.044 → 2535.511

Pred boundaries (mean_p):
- 75.000 → 2536.500 (p=0.8856)

### 6L-8f1eYOOY
Truth calls: **1**; Pred calls: **1**; time_f1=0.999; seg_f1=1.000

Truth boundaries:
- 0.000 → 1044.431

Pred boundaries (mean_p):
- 0.500 → 1045.000 (p=0.8689)

### 6UWzVyNtbS0
Truth calls: **1**; Pred calls: **2**; time_f1=0.941; seg_f1=0.667

Truth boundaries:
- 68.833 → 199.138

Pred boundaries (mean_p):
- 21.500 → 34.500 (p=0.2596)
- 71.500 → 199.500 (p=0.5509)

### 9g2gwfWIbUk
Truth calls: **2**; Pred calls: **1**; time_f1=0.986; seg_f1=0.667

Truth boundaries:
- 0.000 → 934.003
- 937.520 → 1955.949

Pred boundaries (mean_p):
- 1.000 → 2008.500 (p=0.8594)

### CfMJ01KP_ns
Truth calls: **6**; Pred calls: **3**; time_f1=0.990; seg_f1=0.444

Truth boundaries:
- 44.818 → 198.874
- 198.931 → 635.358
- 729.783 → 967.727
- 970.201 → 1538.415
- 1539.348 → 2231.910
- 2232.687 → 2654.263

Pred boundaries (mean_p):
- 47.500 → 653.500 (p=0.8829)
- 734.000 → 1693.000 (p=0.8402)
- 1704.000 → 2665.500 (p=0.8708)

### D4uiHjHW4AU
Truth calls: **19**; Pred calls: **9**; time_f1=0.857; seg_f1=0.214

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
- 83.000 → 92.500 (p=0.3155)
- 238.000 → 379.000 (p=0.5322)
- 396.500 → 415.000 (p=0.4399)
- 433.500 → 641.500 (p=0.6912)
- 700.500 → 996.000 (p=0.6514)
- 1015.000 → 1784.000 (p=0.7929)
- 1795.000 → 2355.000 (p=0.6918)
- 2366.500 → 2476.500 (p=0.4967)
- 2493.500 → 2999.500 (p=0.6679)

### DEt3IRqqUVs
Truth calls: **12**; Pred calls: **2**; time_f1=0.975; seg_f1=0.000

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
- 5.000 → 175.500 (p=0.7814)
- 187.500 → 822.000 (p=0.7598)

### FkgGv2iMjEo
Truth calls: **8**; Pred calls: **7**; time_f1=0.986; seg_f1=0.667

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
- 140.500 → 253.500 (p=0.6649)
- 317.000 → 572.500 (p=0.7055)
- 654.000 → 811.000 (p=0.7048)
- 847.500 → 944.000 (p=0.6980)
- 1019.500 → 1098.000 (p=0.6885)
- 1111.500 → 3623.500 (p=0.7388)
- 3647.500 → 3653.500 (p=0.4505)

### FotSodGIVM8
Truth calls: **2**; Pred calls: **1**; time_f1=0.788; seg_f1=0.667

Truth boundaries:
- 33.524 → 366.326
- 461.912 → 633.082

Pred boundaries (mean_p):
- 38.500 → 366.500 (p=0.6688)

### GcjtU_V-mW4
Truth calls: **2**; Pred calls: **3**; time_f1=0.938; seg_f1=0.800

Truth boundaries:
- 142.589 → 238.604
- 513.423 → 604.887

Pred boundaries (mean_p):
- 149.500 → 238.500 (p=0.5774)
- 514.000 → 579.500 (p=0.4608)
- 593.500 → 604.500 (p=0.4069)

### JT1XSpK1pkM
Truth calls: **2**; Pred calls: **2**; time_f1=0.926; seg_f1=1.000

Truth boundaries:
- 40.397 → 64.726
- 335.183 → 357.848

Pred boundaries (mean_p):
- 40.000 → 68.500 (p=0.6064)
- 335.000 → 361.000 (p=0.7280)

### Krnsw9WZtRA
Truth calls: **8**; Pred calls: **7**; time_f1=0.913; seg_f1=0.800

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
- 28.500 → 61.500 (p=0.6883)
- 105.500 → 180.500 (p=0.5762)
- 248.500 → 261.500 (p=0.7560)
- 274.000 → 317.000 (p=0.5938)
- 339.500 → 357.500 (p=0.3275)
- 438.000 → 456.500 (p=0.3643)
- 487.500 → 559.000 (p=0.6265)

### MPHdNIiB_8E
Truth calls: **0**; Pred calls: **0**; time_f1=1.000; seg_f1=1.000

Truth boundaries:
- (none)

Pred boundaries (mean_p):
- (none)

### Ox_LARRQmlw
Truth calls: **1**; Pred calls: **1**; time_f1=0.998; seg_f1=1.000

Truth boundaries:
- 80.995 → 716.291

Pred boundaries (mean_p):
- 81.500 → 714.500 (p=0.8597)

### PnTbJdhNbPk
Truth calls: **1**; Pred calls: **1**; time_f1=0.998; seg_f1=1.000

Truth boundaries:
- 15.854 → 1384.158

Pred boundaries (mean_p):
- 22.000 → 1384.500 (p=0.8103)

### Qa-ppZFUp0g
Truth calls: **27**; Pred calls: **19**; time_f1=0.944; seg_f1=0.565

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
- 64.500 → 184.500 (p=0.7167)
- 195.000 → 201.500 (p=0.4545)
- 216.500 → 281.000 (p=0.5407)
- 298.500 → 320.000 (p=0.4161)
- 344.000 → 489.000 (p=0.6733)
- 527.000 → 559.500 (p=0.6873)
- 574.500 → 600.500 (p=0.4537)
- 678.500 → 711.500 (p=0.6014)
- 748.000 → 787.500 (p=0.4845)
- 801.000 → 996.000 (p=0.5294)
- 1036.500 → 1062.000 (p=0.4209)
- 1123.000 → 1149.500 (p=0.5301)
- 1189.500 → 1252.000 (p=0.5763)
- 1277.500 → 1309.000 (p=0.6389)
- 1331.500 → 1382.000 (p=0.6818)
- 1408.500 → 1482.000 (p=0.5569)
- 1577.500 → 1699.000 (p=0.6417)
- 1710.000 → 1770.500 (p=0.7388)
- 1785.000 → 1829.000 (p=0.4977)

### RL6Y5qig0Wg
Truth calls: **7**; Pred calls: **7**; time_f1=0.991; seg_f1=1.000

Truth boundaries:
- 88.710 → 222.038
- 455.360 → 559.472
- 638.357 → 675.494
- 766.126 → 779.070
- 893.717 → 1059.770
- 1157.827 → 1225.158
- 1434.781 → 1602.009

Pred boundaries (mean_p):
- 89.000 → 224.500 (p=0.6319)
- 458.000 → 561.500 (p=0.6908)
- 638.000 → 676.500 (p=0.7897)
- 765.500 → 779.000 (p=0.8105)
- 894.000 → 1060.000 (p=0.7546)
- 1157.500 → 1225.000 (p=0.6121)
- 1434.000 → 1602.500 (p=0.6056)

### RtOncKiOT48
Truth calls: **2**; Pred calls: **5**; time_f1=0.733; seg_f1=0.286

Truth boundaries:
- 98.156 → 484.037
- 563.603 → 1483.472

Pred boundaries (mean_p):
- 1.500 → 30.000 (p=0.5347)
- 100.000 → 820.500 (p=0.6419)
- 852.000 → 1069.000 (p=0.6545)
- 1712.500 → 1764.000 (p=0.2639)
- 1792.000 → 1810.000 (p=0.2332)

### S8yFUyD_JXU
Truth calls: **1**; Pred calls: **4**; time_f1=0.824; seg_f1=0.400

Truth boundaries:
- 77.611 → 443.224

Pred boundaries (mean_p):
- 7.500 → 34.000 (p=0.4272)
- 44.500 → 104.000 (p=0.5131)
- 144.000 → 330.500 (p=0.6079)
- 341.000 → 468.000 (p=0.5284)

### UWLJ81ezcpU
Truth calls: **7**; Pred calls: **13**; time_f1=0.827; seg_f1=0.500

Truth boundaries:
- 387.932 → 527.672
- 657.982 → 692.946
- 736.642 → 760.148
- 795.336 → 889.925
- 1033.707 → 1139.590
- 1199.640 → 1270.695
- 1313.627 → 1427.264

Pred boundaries (mean_p):
- 390.000 → 422.500 (p=0.5668)
- 433.500 → 438.500 (p=0.5570)
- 462.000 → 469.500 (p=0.2958)
- 504.000 → 560.500 (p=0.3641)
- 662.000 → 690.500 (p=0.4761)
- 745.500 → 759.500 (p=0.6033)
- 797.500 → 891.000 (p=0.5701)
- 1033.500 → 1053.000 (p=0.4341)
- 1065.500 → 1102.000 (p=0.6017)
- 1114.000 → 1144.500 (p=0.6074)
- 1200.500 → 1207.500 (p=0.3363)
- 1232.500 → 1270.000 (p=0.4009)
- 1314.000 → 1423.500 (p=0.4967)

### Uytq1t3zAz8
Truth calls: **1**; Pred calls: **2**; time_f1=0.996; seg_f1=0.667

Truth boundaries:
- 8.884 → 1098.453

Pred boundaries (mean_p):
- 8.500 → 1099.500 (p=0.8050)
- 1142.500 → 1150.000 (p=0.4103)

### VXOSVmvwV0c
Truth calls: **3**; Pred calls: **3**; time_f1=0.952; seg_f1=0.333

Truth boundaries:
- 37.446 → 173.066
- 181.256 → 267.866
- 274.943 → 393.007

Pred boundaries (mean_p):
- 43.000 → 70.000 (p=0.4071)
- 80.500 → 267.500 (p=0.4229)
- 282.500 → 393.000 (p=0.5515)

### VZwqygti-nc
Truth calls: **4**; Pred calls: **4**; time_f1=0.947; seg_f1=0.750

Truth boundaries:
- 40.401 → 83.021
- 93.259 → 140.640
- 167.200 → 197.157
- 201.827 → 332.638

Pred boundaries (mean_p):
- 40.500 → 82.000 (p=0.6530)
- 92.500 → 148.000 (p=0.6703)
- 166.000 → 295.500 (p=0.6015)
- 306.500 → 332.000 (p=0.7094)

### ZjhxWa0N4II
Truth calls: **3**; Pred calls: **1**; time_f1=0.934; seg_f1=0.000

Truth boundaries:
- 10.255 → 66.691
- 72.149 → 144.985
- 150.111 → 305.392

Pred boundaries (mean_p):
- 11.500 → 333.500 (p=0.8576)

### aW8jAYnvqyI
Truth calls: **0**; Pred calls: **0**; time_f1=1.000; seg_f1=1.000

Truth boundaries:
- (none)

Pred boundaries (mean_p):
- (none)

### bCxvXMhbFls
Truth calls: **1**; Pred calls: **1**; time_f1=0.081; seg_f1=0.000

Truth boundaries:
- 43.068 → 713.823

Pred boundaries (mean_p):
- 43.000 → 71.500 (p=0.4408)

### bPv8JzD_bMI
Truth calls: **4**; Pred calls: **8**; time_f1=0.735; seg_f1=0.167

Truth boundaries:
- 86.956 → 111.865
- 530.643 → 569.233
- 728.712 → 847.820
- 926.350 → 1034.779

Pred boundaries (mean_p):
- 94.000 → 129.000 (p=0.3963)
- 475.500 → 481.000 (p=0.3742)
- 505.000 → 514.000 (p=0.4411)
- 549.500 → 565.500 (p=0.3633)
- 614.500 → 653.000 (p=0.7153)
- 732.500 → 746.000 (p=0.4213)
- 781.500 → 838.000 (p=0.5875)
- 926.000 → 1039.000 (p=0.6636)

### ci-FdcWiJiA
Truth calls: **7**; Pred calls: **23**; time_f1=0.725; seg_f1=0.400

Truth boundaries:
- 0.000 → 82.192
- 257.803 → 276.974
- 389.934 → 435.465
- 595.341 → 616.391
- 1020.462 → 1124.976
- 1752.450 → 1772.900
- 2034.918 → 2171.015

Pred boundaries (mean_p):
- 6.000 → 86.000 (p=0.5375)
- 137.000 → 146.500 (p=0.3039)
- 192.500 → 198.500 (p=0.4150)
- 227.000 → 247.000 (p=0.3197)
- 257.500 → 282.000 (p=0.4813)
- 296.000 → 322.000 (p=0.5639)
- 378.500 → 424.500 (p=0.4232)
- 456.500 → 468.000 (p=0.1852)
- 595.500 → 619.500 (p=0.5893)
- 643.500 → 655.500 (p=0.2781)
- 681.500 → 706.000 (p=0.2140)
- 739.500 → 750.500 (p=0.2409)
- 873.500 → 902.500 (p=0.3249)
- 967.000 → 972.000 (p=0.5519)
- 1027.000 → 1051.500 (p=0.4834)
- 1062.000 → 1134.500 (p=0.5289)
- 1273.500 → 1280.000 (p=0.6026)
- 1419.500 → 1429.500 (p=0.4041)
- 1538.500 → 1553.000 (p=0.2601)
- 1609.000 → 1619.000 (p=0.3767)
- 1712.500 → 1717.500 (p=0.6157)
- 1736.000 → 1786.500 (p=0.3140)
- 2034.500 → 2172.500 (p=0.5624)

### dZWKwNqDXIo
Truth calls: **5**; Pred calls: **5**; time_f1=0.996; seg_f1=1.000

Truth boundaries:
- 60.993 → 158.577
- 231.489 → 498.542
- 553.184 → 807.714
- 840.897 → 1031.735
- 1080.048 → 1727.524

Pred boundaries (mean_p):
- 60.500 → 159.000 (p=0.7353)
- 232.000 → 501.000 (p=0.8641)
- 553.000 → 814.500 (p=0.7629)
- 840.500 → 1031.500 (p=0.6659)
- 1079.500 → 1727.500 (p=0.7839)

### dyFaLq3kfHs
Truth calls: **1**; Pred calls: **1**; time_f1=0.991; seg_f1=1.000

Truth boundaries:
- 73.850 → 499.583

Pred boundaries (mean_p):
- 73.500 → 507.000 (p=0.7988)

### gDdoZC9Nhgk
Truth calls: **7**; Pred calls: **5**; time_f1=0.981; seg_f1=0.500

Truth boundaries:
- 96.069 → 150.692
- 257.311 → 466.293
- 468.043 → 507.117
- 512.702 → 549.139
- 697.043 → 902.660
- 908.561 → 964.391
- 978.786 → 1248.417

Pred boundaries (mean_p):
- 95.500 → 156.500 (p=0.7720)
- 257.000 → 312.000 (p=0.4844)
- 323.500 → 549.500 (p=0.6335)
- 696.500 → 965.000 (p=0.6738)
- 979.000 → 1249.000 (p=0.6682)

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
Truth calls: **1**; Pred calls: **1**; time_f1=0.991; seg_f1=1.000

Truth boundaries:
- 55.469 → 1045.232

Pred boundaries (mean_p):
- 57.000 → 1062.000 (p=0.8293)

### m2z1tgfv8Kc
Truth calls: **0**; Pred calls: **0**; time_f1=1.000; seg_f1=1.000

Truth boundaries:
- (none)

Pred boundaries (mean_p):
- (none)

### mxCrbMSfup4
Truth calls: **3**; Pred calls: **8**; time_f1=0.944; seg_f1=0.364

Truth boundaries:
- 40.697 → 707.536
- 745.894 → 763.517
- 813.441 → 1007.483

Pred boundaries (mean_p):
- 40.500 → 248.500 (p=0.7005)
- 264.000 → 513.500 (p=0.6580)
- 534.500 → 540.500 (p=0.3364)
- 559.000 → 669.000 (p=0.5500)
- 682.500 → 707.000 (p=0.6646)
- 746.500 → 764.000 (p=0.6468)
- 811.000 → 947.000 (p=0.5786)
- 967.000 → 1007.500 (p=0.5674)

### nNA0XQO_7jQ
Truth calls: **0**; Pred calls: **0**; time_f1=1.000; seg_f1=1.000

Truth boundaries:
- (none)

Pred boundaries (mean_p):
- (none)

### pCcAlwOYYx8
Truth calls: **1**; Pred calls: **6**; time_f1=0.946; seg_f1=0.000

Truth boundaries:
- 19.288 → 1095.080

Pred boundaries (mean_p):
- 19.500 → 89.500 (p=0.6284)
- 105.000 → 585.000 (p=0.6543)
- 609.000 → 644.500 (p=0.5908)
- 684.000 → 751.500 (p=0.5232)
- 767.500 → 794.500 (p=0.5551)
- 809.000 → 1095.000 (p=0.6386)

### pgYGE9jmNPA
Truth calls: **1**; Pred calls: **5**; time_f1=0.975; seg_f1=0.333

Truth boundaries:
- 43.409 → 909.003

Pred boundaries (mean_p):
- 43.000 → 521.500 (p=0.6819)
- 534.500 → 593.000 (p=0.4640)
- 604.500 → 622.000 (p=0.4243)
- 633.000 → 909.000 (p=0.5818)
- 978.500 → 986.000 (p=0.2910)

### pjf5uhOMcTc
Truth calls: **0**; Pred calls: **0**; time_f1=1.000; seg_f1=1.000

Truth boundaries:
- (none)

Pred boundaries (mean_p):
- (none)

