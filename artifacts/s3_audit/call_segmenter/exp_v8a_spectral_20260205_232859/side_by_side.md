# exp_v8a_spectral: Truth vs Predicted Call Boundaries (side-by-side)

Bucket (pred): `rezora-whisperx-us-east-1-864981718771`
Predictions: `s3://rezora-whisperx-us-east-1-864981718771/call_segmenter/predictions/exp_v8a_spectral_20260205_232428_thr65_off55_gap30_gp90_p20_min10/`
Labels: `s3://rezora-whisperx-us-east-1-864981718771/labeling/corrected_boundaries/v1/`

Model git sha: `aad3a21adc31`
Decode: `threshold` (thr_on=0.65, thr_off=0.55, gap_merge_s=30.0, gap_stat=p90, gap_min_p=0.2, min_seg_s=10.0)

Format:
- Truth boundaries are your manual call labels.
- Pred boundaries are model outputs (post-processed) with mean probability `p` across the segment.

## Top 10 by Absolute Call-Count Delta

| video_id | truth_calls | pred_calls | delta | time_f1 | seg_f1 |
|---|---:|---:|---:|---:|---:|
| `Qa-ppZFUp0g` | 27 | 11 | -16 | 0.914 | 0.263 |
| `DEt3IRqqUVs` | 12 | 3 | -9 | 0.968 | 0.133 |
| `9g2gwfWIbUk` | 2 | 10 | +8 | 0.983 | 0.167 |
| `D4uiHjHW4AU` | 19 | 14 | -5 | 0.894 | 0.364 |
| `4ipwbOJRMck` | 1 | 5 | +4 | 0.964 | 0.000 |
| `Krnsw9WZtRA` | 8 | 4 | -4 | 0.836 | 0.500 |
| `RtOncKiOT48` | 2 | 6 | +4 | 0.717 | 0.000 |
| `pgYGE9jmNPA` | 1 | 5 | +4 | 0.925 | 0.333 |
| `-POaWp9_UaM` | 4 | 1 | -3 | 0.991 | 0.000 |
| `RL6Y5qig0Wg` | 7 | 10 | +3 | 0.987 | 0.824 |

## Top 10 Worst by Segment IoU-F1

| video_id | truth_calls | pred_calls | time_f1 | seg_f1 | unmatched_pred | unmatched_truth |
|---|---:|---:|---:|---:|---:|---:|
| `-POaWp9_UaM` | 4 | 1 | 0.991 | 0.000 | 1 | 4 |
| `4ipwbOJRMck` | 1 | 5 | 0.964 | 0.000 | 5 | 1 |
| `PnTbJdhNbPk` | 1 | 2 | 0.707 | 0.000 | 2 | 1 |
| `RtOncKiOT48` | 2 | 6 | 0.717 | 0.000 | 6 | 2 |
| `bCxvXMhbFls` | 1 | 1 | 0.053 | 0.000 | 1 | 1 |
| `pCcAlwOYYx8` | 1 | 3 | 0.940 | 0.000 | 3 | 1 |
| `DEt3IRqqUVs` | 12 | 3 | 0.968 | 0.133 | 2 | 11 |
| `9g2gwfWIbUk` | 2 | 10 | 0.983 | 0.167 | 9 | 1 |
| `Qa-ppZFUp0g` | 27 | 11 | 0.914 | 0.263 | 6 | 22 |
| `VZwqygti-nc` | 4 | 2 | 0.912 | 0.333 | 1 | 3 |

## Counts Summary (all videos)

| video_id | truth_calls | pred_calls |
|---|---:|---:|
| `-POaWp9_UaM` | 4 | 1 |
| `-UzXRV8nHn4` | 1 | 1 |
| `15QzruVINDc` | 0 | 0 |
| `4OweikRF7bg` | 0 | 0 |
| `4ipwbOJRMck` | 1 | 5 |
| `6L-8f1eYOOY` | 1 | 1 |
| `6UWzVyNtbS0` | 1 | 1 |
| `9g2gwfWIbUk` | 2 | 10 |
| `CfMJ01KP_ns` | 6 | 6 |
| `D4uiHjHW4AU` | 19 | 14 |
| `DEt3IRqqUVs` | 12 | 3 |
| `FkgGv2iMjEo` | 8 | 9 |
| `FotSodGIVM8` | 2 | 3 |
| `GcjtU_V-mW4` | 2 | 2 |
| `JT1XSpK1pkM` | 2 | 2 |
| `Krnsw9WZtRA` | 8 | 4 |
| `MPHdNIiB_8E` | 0 | 0 |
| `Ox_LARRQmlw` | 1 | 1 |
| `PnTbJdhNbPk` | 1 | 2 |
| `Qa-ppZFUp0g` | 27 | 11 |
| `RL6Y5qig0Wg` | 7 | 10 |
| `RtOncKiOT48` | 2 | 6 |
| `S8yFUyD_JXU` | 1 | 2 |
| `UWLJ81ezcpU` | 7 | 9 |
| `Uytq1t3zAz8` | 1 | 1 |
| `VXOSVmvwV0c` | 3 | 4 |
| `VZwqygti-nc` | 4 | 2 |
| `ZjhxWa0N4II` | 3 | 2 |
| `aW8jAYnvqyI` | 0 | 0 |
| `bCxvXMhbFls` | 1 | 1 |
| `bPv8JzD_bMI` | 4 | 6 |
| `ci-FdcWiJiA` | 7 | 10 |
| `dZWKwNqDXIo` | 5 | 6 |
| `dyFaLq3kfHs` | 1 | 2 |
| `gDdoZC9Nhgk` | 7 | 4 |
| `hYjbrxLXJIU` | 0 | 0 |
| `hs44BGJtOwg` | 0 | 0 |
| `jvFLW5EClgk` | 0 | 0 |
| `lioFo9pz9x8` | 1 | 1 |
| `m2z1tgfv8Kc` | 0 | 0 |
| `mxCrbMSfup4` | 3 | 5 |
| `nNA0XQO_7jQ` | 0 | 0 |
| `pCcAlwOYYx8` | 1 | 3 |
| `pgYGE9jmNPA` | 1 | 5 |
| `pjf5uhOMcTc` | 0 | 0 |

## Detailed Boundaries

### -POaWp9_UaM
Truth calls: **4**; Pred calls: **1**; time_f1=0.991; seg_f1=0.000

Truth boundaries:
- 68.724 → 214.650
- 216.061 → 352.478
- 352.592 → 404.942
- 406.142 → 683.371

Pred boundaries (mean_p):
- 76.500 → 683.000 (p=0.6863)

### -UzXRV8nHn4
Truth calls: **1**; Pred calls: **1**; time_f1=0.997; seg_f1=1.000

Truth boundaries:
- 181.119 → 430.160

Pred boundaries (mean_p):
- 182.500 → 430.500 (p=0.9131)

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
Truth calls: **1**; Pred calls: **5**; time_f1=0.964; seg_f1=0.000

Truth boundaries:
- 75.044 → 2535.511

Pred boundaries (mean_p):
- 80.000 → 801.000 (p=0.8439)
- 801.000 → 1938.500 (p=0.8054)
- 1938.500 → 2628.000 (p=0.7569)
- 2767.500 → 2814.500 (p=0.3199)
- 2850.500 → 2890.500 (p=0.3327)

### 6L-8f1eYOOY
Truth calls: **1**; Pred calls: **1**; time_f1=1.000; seg_f1=1.000

Truth boundaries:
- 0.000 → 1044.431

Pred boundaries (mean_p):
- 0.500 → 1044.500 (p=0.7473)

### 6UWzVyNtbS0
Truth calls: **1**; Pred calls: **1**; time_f1=0.969; seg_f1=1.000

Truth boundaries:
- 68.833 → 199.138

Pred boundaries (mean_p):
- 72.000 → 194.500 (p=0.4560)

### 9g2gwfWIbUk
Truth calls: **2**; Pred calls: **10**; time_f1=0.983; seg_f1=0.167

Truth boundaries:
- 0.000 → 934.003
- 937.520 → 1955.949

Pred boundaries (mean_p):
- 2.500 → 126.000 (p=0.8209)
- 126.000 → 761.000 (p=0.7372)
- 761.000 → 900.000 (p=0.6310)
- 902.500 → 1023.000 (p=0.7187)
- 1028.000 → 1445.500 (p=0.7177)
- 1445.500 → 1472.000 (p=0.7289)
- 1472.000 → 1592.500 (p=0.6857)
- 1592.500 → 1673.500 (p=0.7375)
- 1676.000 → 1704.000 (p=0.5317)
- 1704.500 → 2008.000 (p=0.5865)

### CfMJ01KP_ns
Truth calls: **6**; Pred calls: **6**; time_f1=0.987; seg_f1=0.500

Truth boundaries:
- 44.818 → 198.874
- 198.931 → 635.358
- 729.783 → 967.727
- 970.201 → 1538.415
- 1539.348 → 2231.910
- 2232.687 → 2654.263

Pred boundaries (mean_p):
- 51.500 → 635.500 (p=0.7717)
- 734.000 → 1217.000 (p=0.6818)
- 1265.000 → 1720.500 (p=0.6914)
- 1720.500 → 1762.500 (p=0.7812)
- 1763.500 → 2163.000 (p=0.7564)
- 2163.000 → 2654.500 (p=0.7280)

### D4uiHjHW4AU
Truth calls: **19**; Pred calls: **14**; time_f1=0.894; seg_f1=0.364

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
- 66.500 → 88.500 (p=0.2351)
- 268.000 → 338.500 (p=0.3777)
- 339.500 → 363.000 (p=0.3519)
- 397.500 → 470.000 (p=0.3696)
- 472.000 → 627.500 (p=0.5070)
- 718.000 → 888.500 (p=0.4929)
- 889.000 → 975.000 (p=0.7139)
- 1015.500 → 1647.500 (p=0.6230)
- 1647.500 → 1768.000 (p=0.6613)
- 1867.500 → 2325.500 (p=0.5598)
- 2372.000 → 2446.500 (p=0.3886)
- 2493.500 → 2591.000 (p=0.4902)
- 2591.000 → 2604.500 (p=0.4475)
- 2605.000 → 2980.500 (p=0.5297)

### DEt3IRqqUVs
Truth calls: **12**; Pred calls: **3**; time_f1=0.968; seg_f1=0.133

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
- 5.000 → 321.500 (p=0.7326)
- 321.500 → 684.000 (p=0.7065)
- 685.000 → 820.000 (p=0.5830)

### FkgGv2iMjEo
Truth calls: **8**; Pred calls: **9**; time_f1=0.988; seg_f1=0.588

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
- 145.000 → 253.500 (p=0.6020)
- 317.000 → 559.500 (p=0.6811)
- 654.000 → 811.000 (p=0.6467)
- 852.500 → 884.000 (p=0.5621)
- 884.000 → 959.000 (p=0.6322)
- 1037.500 → 2723.500 (p=0.6637)
- 2723.500 → 2749.000 (p=0.6760)
- 2749.000 → 2850.000 (p=0.5901)
- 2850.500 → 3623.500 (p=0.6600)

### FotSodGIVM8
Truth calls: **2**; Pred calls: **3**; time_f1=0.762; seg_f1=0.400

Truth boundaries:
- 33.524 → 366.326
- 461.912 → 633.082

Pred boundaries (mean_p):
- 38.500 → 336.500 (p=0.7180)
- 337.000 → 366.500 (p=0.4844)
- 1130.000 → 1158.000 (p=0.2421)

### GcjtU_V-mW4
Truth calls: **2**; Pred calls: **2**; time_f1=0.905; seg_f1=1.000

Truth boundaries:
- 142.589 → 238.604
- 513.423 → 604.887

Pred boundaries (mean_p):
- 149.500 → 269.000 (p=0.4874)
- 513.500 → 604.500 (p=0.4577)

### JT1XSpK1pkM
Truth calls: **2**; Pred calls: **2**; time_f1=0.835; seg_f1=1.000

Truth boundaries:
- 40.397 → 64.726
- 335.183 → 357.848

Pred boundaries (mean_p):
- 40.000 → 68.500 (p=0.5799)
- 335.000 → 372.000 (p=0.5701)

### Krnsw9WZtRA
Truth calls: **8**; Pred calls: **4**; time_f1=0.836; seg_f1=0.500

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
- 19.500 → 61.500 (p=0.5749)
- 108.000 → 179.500 (p=0.5858)
- 248.500 → 354.500 (p=0.4584)
- 455.000 → 559.000 (p=0.5263)

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
- 82.000 → 715.000 (p=0.7083)

### PnTbJdhNbPk
Truth calls: **1**; Pred calls: **2**; time_f1=0.707; seg_f1=0.000

Truth boundaries:
- 15.854 → 1384.158

Pred boundaries (mean_p):
- 30.000 → 599.500 (p=0.5910)
- 1204.500 → 1384.500 (p=0.5354)

### Qa-ppZFUp0g
Truth calls: **27**; Pred calls: **11**; time_f1=0.914; seg_f1=0.263

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
- 64.500 → 489.000 (p=0.5742)
- 527.500 → 600.500 (p=0.4925)
- 679.000 → 710.500 (p=0.6264)
- 751.500 → 877.000 (p=0.4678)
- 877.000 → 1010.500 (p=0.4906)
- 1042.000 → 1061.500 (p=0.3891)
- 1123.000 → 1149.500 (p=0.5575)
- 1189.500 → 1482.000 (p=0.4988)
- 1513.000 → 1525.000 (p=0.2512)
- 1577.500 → 1828.500 (p=0.6091)
- 1925.500 → 1943.500 (p=0.3039)

### RL6Y5qig0Wg
Truth calls: **7**; Pred calls: **10**; time_f1=0.987; seg_f1=0.824

Truth boundaries:
- 88.710 → 222.038
- 455.360 → 559.472
- 638.357 → 675.494
- 766.126 → 779.070
- 893.717 → 1059.770
- 1157.827 → 1225.158
- 1434.781 → 1602.009

Pred boundaries (mean_p):
- 89.000 → 182.000 (p=0.6432)
- 182.000 → 220.500 (p=0.5492)
- 458.000 → 561.500 (p=0.6378)
- 638.000 → 675.500 (p=0.7502)
- 765.500 → 777.500 (p=0.8180)
- 896.000 → 1060.000 (p=0.7197)
- 1160.000 → 1225.000 (p=0.5858)
- 1434.000 → 1529.500 (p=0.6673)
- 1530.000 → 1540.000 (p=0.4116)
- 1540.500 → 1599.500 (p=0.5131)

### RtOncKiOT48
Truth calls: **2**; Pred calls: **6**; time_f1=0.717; seg_f1=0.000

Truth boundaries:
- 98.156 → 484.037
- 563.603 → 1483.472

Pred boundaries (mean_p):
- 1.500 → 27.000 (p=0.5452)
- 100.500 → 194.000 (p=0.5026)
- 234.500 → 361.000 (p=0.4588)
- 362.000 → 604.500 (p=0.5975)
- 604.500 → 816.000 (p=0.6675)
- 874.500 → 1068.500 (p=0.5886)

### S8yFUyD_JXU
Truth calls: **1**; Pred calls: **2**; time_f1=0.836; seg_f1=0.667

Truth boundaries:
- 77.611 → 443.224

Pred boundaries (mean_p):
- 28.500 → 91.000 (p=0.4023)
- 144.000 → 463.500 (p=0.4888)

### UWLJ81ezcpU
Truth calls: **7**; Pred calls: **9**; time_f1=0.853; seg_f1=0.875

Truth boundaries:
- 387.932 → 527.672
- 657.982 → 692.946
- 736.642 → 760.148
- 795.336 → 889.925
- 1033.707 → 1139.590
- 1199.640 → 1270.695
- 1313.627 → 1427.264

Pred boundaries (mean_p):
- 390.500 → 471.000 (p=0.4417)
- 503.500 → 604.500 (p=0.4123)
- 662.000 → 690.500 (p=0.5636)
- 746.000 → 758.000 (p=0.6571)
- 797.500 → 922.000 (p=0.5312)
- 1031.500 → 1080.000 (p=0.4730)
- 1080.000 → 1146.500 (p=0.6432)
- 1200.000 → 1268.500 (p=0.3937)
- 1314.000 → 1423.500 (p=0.5392)

### Uytq1t3zAz8
Truth calls: **1**; Pred calls: **1**; time_f1=0.977; seg_f1=1.000

Truth boundaries:
- 8.884 → 1098.453

Pred boundaries (mean_p):
- 8.500 → 1148.500 (p=0.6126)

### VXOSVmvwV0c
Truth calls: **3**; Pred calls: **4**; time_f1=0.966; seg_f1=0.857

Truth boundaries:
- 37.446 → 173.066
- 181.256 → 267.866
- 274.943 → 393.007

Pred boundaries (mean_p):
- 43.500 → 168.000 (p=0.5042)
- 168.500 → 233.500 (p=0.4860)
- 234.500 → 324.000 (p=0.4307)
- 324.000 → 392.500 (p=0.6151)

### VZwqygti-nc
Truth calls: **4**; Pred calls: **2**; time_f1=0.912; seg_f1=0.333

Truth boundaries:
- 40.401 → 83.021
- 93.259 → 140.640
- 167.200 → 197.157
- 201.827 → 332.638

Pred boundaries (mean_p):
- 44.500 → 66.500 (p=0.5650)
- 67.000 → 331.500 (p=0.5082)

### ZjhxWa0N4II
Truth calls: **3**; Pred calls: **2**; time_f1=0.934; seg_f1=0.800

Truth boundaries:
- 10.255 → 66.691
- 72.149 → 144.985
- 150.111 → 305.392

Pred boundaries (mean_p):
- 11.500 → 94.000 (p=0.8329)
- 94.000 → 333.500 (p=0.7624)

### aW8jAYnvqyI
Truth calls: **0**; Pred calls: **0**; time_f1=1.000; seg_f1=1.000

Truth boundaries:
- (none)

Pred boundaries (mean_p):
- (none)

### bCxvXMhbFls
Truth calls: **1**; Pred calls: **1**; time_f1=0.053; seg_f1=0.000

Truth boundaries:
- 43.068 → 713.823

Pred boundaries (mean_p):
- 43.000 → 61.500 (p=0.4053)

### bPv8JzD_bMI
Truth calls: **4**; Pred calls: **6**; time_f1=0.762; seg_f1=0.600

Truth boundaries:
- 86.956 → 111.865
- 530.643 → 569.233
- 728.712 → 847.820
- 926.350 → 1034.779

Pred boundaries (mean_p):
- 86.500 → 110.000 (p=0.5059)
- 110.000 → 136.000 (p=0.5061)
- 475.500 → 566.500 (p=0.3880)
- 614.500 → 669.000 (p=0.6218)
- 739.000 → 845.000 (p=0.4609)
- 926.000 → 1056.500 (p=0.6168)

### ci-FdcWiJiA
Truth calls: **7**; Pred calls: **10**; time_f1=0.751; seg_f1=0.471

Truth boundaries:
- 0.000 → 82.192
- 257.803 → 276.974
- 389.934 → 435.465
- 595.341 → 616.391
- 1020.462 → 1124.976
- 1752.450 → 1772.900
- 2034.918 → 2171.015

Pred boundaries (mean_p):
- 6.000 → 63.500 (p=0.5423)
- 63.500 → 104.500 (p=0.2958)
- 227.500 → 319.000 (p=0.3182)
- 389.500 → 420.000 (p=0.3596)
- 595.500 → 645.500 (p=0.3441)
- 892.000 → 902.000 (p=0.4789)
- 1026.500 → 1151.000 (p=0.4086)
- 1274.500 → 1308.000 (p=0.1671)
- 1609.000 → 1619.000 (p=0.3458)
- 2034.500 → 2171.000 (p=0.4919)

### dZWKwNqDXIo
Truth calls: **5**; Pred calls: **6**; time_f1=0.981; seg_f1=0.909

Truth boundaries:
- 60.993 → 158.577
- 231.489 → 498.542
- 553.184 → 807.714
- 840.897 → 1031.735
- 1080.048 → 1727.524

Pred boundaries (mean_p):
- 61.500 → 178.500 (p=0.6609)
- 232.000 → 499.000 (p=0.8674)
- 553.000 → 854.500 (p=0.6692)
- 856.000 → 1030.500 (p=0.6778)
- 1080.000 → 1682.500 (p=0.7625)
- 1682.500 → 1727.500 (p=0.8588)

### dyFaLq3kfHs
Truth calls: **1**; Pred calls: **2**; time_f1=0.936; seg_f1=0.667

Truth boundaries:
- 73.850 → 499.583

Pred boundaries (mean_p):
- 60.500 → 94.000 (p=0.4649)
- 127.000 → 507.000 (p=0.7307)

### gDdoZC9Nhgk
Truth calls: **7**; Pred calls: **4**; time_f1=0.983; seg_f1=0.727

Truth boundaries:
- 96.069 → 150.692
- 257.311 → 466.293
- 468.043 → 507.117
- 512.702 → 549.139
- 697.043 → 902.660
- 908.561 → 964.391
- 978.786 → 1248.417

Pred boundaries (mean_p):
- 97.500 → 150.500 (p=0.7186)
- 257.000 → 549.500 (p=0.5135)
- 696.500 → 932.000 (p=0.5538)
- 932.000 → 1248.500 (p=0.5661)

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
Truth calls: **1**; Pred calls: **1**; time_f1=0.995; seg_f1=1.000

Truth boundaries:
- 55.469 → 1045.232

Pred boundaries (mean_p):
- 58.500 → 1052.500 (p=0.6820)

### m2z1tgfv8Kc
Truth calls: **0**; Pred calls: **0**; time_f1=1.000; seg_f1=1.000

Truth boundaries:
- (none)

Pred boundaries (mean_p):
- (none)

### mxCrbMSfup4
Truth calls: **3**; Pred calls: **5**; time_f1=0.983; seg_f1=0.750

Truth boundaries:
- 40.697 → 707.536
- 745.894 → 763.517
- 813.441 → 1007.483

Pred boundaries (mean_p):
- 44.500 → 503.000 (p=0.6258)
- 504.000 → 589.000 (p=0.3445)
- 589.000 → 706.500 (p=0.4974)
- 749.000 → 767.500 (p=0.5419)
- 808.500 → 1020.500 (p=0.5361)

### nNA0XQO_7jQ
Truth calls: **0**; Pred calls: **0**; time_f1=1.000; seg_f1=1.000

Truth boundaries:
- (none)

Pred boundaries (mean_p):
- (none)

### pCcAlwOYYx8
Truth calls: **1**; Pred calls: **3**; time_f1=0.940; seg_f1=0.000

Truth boundaries:
- 19.288 → 1095.080

Pred boundaries (mean_p):
- 22.500 → 552.500 (p=0.6028)
- 610.500 → 643.000 (p=0.5099)
- 685.000 → 1115.500 (p=0.5019)

### pgYGE9jmNPA
Truth calls: **1**; Pred calls: **5**; time_f1=0.925; seg_f1=0.333

Truth boundaries:
- 43.409 → 909.003

Pred boundaries (mean_p):
- 11.000 → 607.000 (p=0.6628)
- 609.500 → 620.500 (p=0.4568)
- 621.000 → 909.000 (p=0.6046)
- 949.500 → 1040.500 (p=0.2771)
- 1155.500 → 1169.000 (p=0.4343)

### pjf5uhOMcTc
Truth calls: **0**; Pred calls: **0**; time_f1=1.000; seg_f1=1.000

Truth boundaries:
- (none)

Pred boundaries (mean_p):
- (none)

