# exp_v5c_neg5_20260204_thr065_off055_gap30_p020_min10: Truth vs Predicted Call Boundaries (side-by-side)

Bucket (pred): `rezora-whisperx-us-east-1-864981718771`
Predictions: `s3://rezora-whisperx-us-east-1-864981718771/call_segmenter/predictions/exp_v5c_neg5_20260204_thr065_off055_gap30_p020_min10/`
Labels: `s3://rezora-whisperx-us-east-1-864981718771/labeling/corrected_boundaries/v1/`

Format:
- Truth boundaries are your manual call labels.
- Pred boundaries are model outputs (post-processed) with mean probability `p` across the segment.

## Top 10 by Absolute Call-Count Delta

| video_id | truth_calls | pred_calls | delta | time_f1 | seg_f1 |
|---|---:|---:|---:|---:|---:|
| `Qa-ppZFUp0g` | 27 | 11 | -16 | 0.907 | 0.211 |
| `4ipwbOJRMck` | 1 | 13 | +12 | 0.921 | 0.000 |
| `DEt3IRqqUVs` | 12 | 2 | -10 | 0.968 | 0.000 |
| `ci-FdcWiJiA` | 7 | 14 | +7 | 0.544 | 0.286 |
| `D4uiHjHW4AU` | 19 | 13 | -6 | 0.881 | 0.500 |
| `PnTbJdhNbPk` | 1 | 6 | +5 | 0.597 | 0.000 |
| `FkgGv2iMjEo` | 8 | 12 | +4 | 0.961 | 0.500 |
| `Krnsw9WZtRA` | 8 | 4 | -4 | 0.771 | 0.333 |
| `-POaWp9_UaM` | 4 | 1 | -3 | 0.991 | 0.000 |
| `RtOncKiOT48` | 2 | 5 | +3 | 0.698 | 0.000 |

## Top 10 Worst by Segment IoU-F1

| video_id | truth_calls | pred_calls | time_f1 | seg_f1 | unmatched_pred | unmatched_truth |
|---|---:|---:|---:|---:|---:|---:|
| `-POaWp9_UaM` | 4 | 1 | 0.991 | 0.000 | 1 | 4 |
| `4ipwbOJRMck` | 1 | 13 | 0.921 | 0.000 | 13 | 1 |
| `DEt3IRqqUVs` | 12 | 2 | 0.968 | 0.000 | 2 | 12 |
| `EVwBLXWlZiI` | 1 | 0 | 0.000 | 0.000 | 0 | 1 |
| `FBmODQn9grE` | 1 | 0 | 0.000 | 0.000 | 0 | 1 |
| `N7XqeLuVOzk` | 1 | 0 | 0.000 | 0.000 | 0 | 1 |
| `PnTbJdhNbPk` | 1 | 6 | 0.597 | 0.000 | 6 | 1 |
| `RtOncKiOT48` | 2 | 5 | 0.698 | 0.000 | 5 | 2 |
| `S8yFUyD_JXU` | 1 | 3 | 0.807 | 0.000 | 3 | 1 |
| `VXOSVmvwV0c` | 3 | 1 | 0.969 | 0.000 | 1 | 3 |

## Counts Summary (all videos)

| video_id | truth_calls | pred_calls |
|---|---:|---:|
| `-POaWp9_UaM` | 4 | 1 |
| `-UzXRV8nHn4` | 1 | 1 |
| `15QzruVINDc` | 0 | 0 |
| `4OweikRF7bg` | 0 | 0 |
| `4ipwbOJRMck` | 1 | 13 |
| `6L-8f1eYOOY` | 1 | 1 |
| `6UWzVyNtbS0` | 1 | 1 |
| `9g2gwfWIbUk` | 2 | 4 |
| `CfMJ01KP_ns` | 6 | 4 |
| `D4uiHjHW4AU` | 19 | 13 |
| `DEt3IRqqUVs` | 12 | 2 |
| `EVwBLXWlZiI` | 1 | 0 |
| `FBmODQn9grE` | 1 | 0 |
| `FkgGv2iMjEo` | 8 | 12 |
| `FotSodGIVM8` | 2 | 4 |
| `GcjtU_V-mW4` | 2 | 2 |
| `JT1XSpK1pkM` | 2 | 2 |
| `Krnsw9WZtRA` | 8 | 4 |
| `MPHdNIiB_8E` | 0 | 0 |
| `N7XqeLuVOzk` | 1 | 0 |
| `Ox_LARRQmlw` | 1 | 2 |
| `PnTbJdhNbPk` | 1 | 6 |
| `Qa-ppZFUp0g` | 27 | 11 |
| `RL6Y5qig0Wg` | 7 | 8 |
| `RtOncKiOT48` | 2 | 5 |
| `S8yFUyD_JXU` | 1 | 3 |
| `UWLJ81ezcpU` | 7 | 9 |
| `Uytq1t3zAz8` | 1 | 1 |
| `VXOSVmvwV0c` | 3 | 1 |
| `VZwqygti-nc` | 4 | 1 |
| `ZjhxWa0N4II` | 3 | 1 |
| `aW8jAYnvqyI` | 0 | 0 |
| `bCxvXMhbFls` | 1 | 1 |
| `bPv8JzD_bMI` | 4 | 7 |
| `ci-FdcWiJiA` | 7 | 14 |
| `dZWKwNqDXIo` | 5 | 6 |
| `dyFaLq3kfHs` | 1 | 2 |
| `gDdoZC9Nhgk` | 7 | 5 |
| `hYjbrxLXJIU` | 0 | 0 |
| `hs44BGJtOwg` | 0 | 0 |
| `jvFLW5EClgk` | 0 | 0 |
| `lioFo9pz9x8` | 1 | 3 |
| `m2z1tgfv8Kc` | 0 | 0 |
| `mxCrbMSfup4` | 3 | 3 |
| `nNA0XQO_7jQ` | 0 | 0 |
| `pCcAlwOYYx8` | 1 | 3 |
| `pgYGE9jmNPA` | 1 | 3 |
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
- 76.000 → 683.000 (p=0.5508)

### -UzXRV8nHn4
Truth calls: **1**; Pred calls: **1**; time_f1=0.982; seg_f1=1.000

Truth boundaries:
- 181.119 → 430.160

Pred boundaries (mean_p):
- 189.500 → 430.500 (p=0.5908)

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
Truth calls: **1**; Pred calls: **13**; time_f1=0.921; seg_f1=0.000

Truth boundaries:
- 75.044 → 2535.511

Pred boundaries (mean_p):
- 95.500 → 455.000 (p=0.7495)
- 455.000 → 488.500 (p=0.7722)
- 489.500 → 548.500 (p=0.8065)
- 548.500 → 796.000 (p=0.7434)
- 796.000 → 851.500 (p=0.7756)
- 851.500 → 1539.000 (p=0.7094)
- 1539.000 → 1636.000 (p=0.7404)
- 1699.000 → 1843.000 (p=0.6576)
- 1843.000 → 1858.500 (p=0.8514)
- 1858.500 → 1908.500 (p=0.7471)
- 1908.500 → 2025.500 (p=0.7825)
- 2025.500 → 2689.500 (p=0.6368)
- 2721.500 → 2890.000 (p=0.4356)

### 6L-8f1eYOOY
Truth calls: **1**; Pred calls: **1**; time_f1=0.999; seg_f1=1.000

Truth boundaries:
- 0.000 → 1044.431

Pred boundaries (mean_p):
- 0.500 → 1043.500 (p=0.6799)

### 6UWzVyNtbS0
Truth calls: **1**; Pred calls: **1**; time_f1=0.767; seg_f1=1.000

Truth boundaries:
- 68.833 → 199.138

Pred boundaries (mean_p):
- 78.500 → 159.500 (p=0.3772)

### 9g2gwfWIbUk
Truth calls: **2**; Pred calls: **4**; time_f1=0.984; seg_f1=0.667

Truth boundaries:
- 0.000 → 934.003
- 937.520 → 1955.949

Pred boundaries (mean_p):
- 3.500 → 1007.000 (p=0.5836)
- 1008.000 → 1751.000 (p=0.5354)
- 1751.000 → 1816.500 (p=0.5242)
- 1861.000 → 1947.500 (p=0.5345)

### CfMJ01KP_ns
Truth calls: **6**; Pred calls: **4**; time_f1=0.978; seg_f1=0.200

Truth boundaries:
- 44.818 → 198.874
- 198.931 → 635.358
- 729.783 → 967.727
- 970.201 → 1538.415
- 1539.348 → 2231.910
- 2232.687 → 2654.263

Pred boundaries (mean_p):
- 52.000 → 642.500 (p=0.7110)
- 734.500 → 1216.500 (p=0.5755)
- 1257.500 → 1682.500 (p=0.6479)
- 1721.000 → 2662.000 (p=0.6269)

### D4uiHjHW4AU
Truth calls: **19**; Pred calls: **13**; time_f1=0.881; seg_f1=0.500

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
- 74.500 → 91.500 (p=0.3952)
- 249.000 → 367.000 (p=0.3592)
- 397.500 → 609.000 (p=0.4242)
- 718.000 → 801.000 (p=0.4928)
- 878.000 → 974.000 (p=0.5347)
- 1024.500 → 1037.500 (p=0.5431)
- 1072.500 → 1773.000 (p=0.5786)
- 1868.500 → 1899.000 (p=0.4772)
- 1943.500 → 2332.500 (p=0.5348)
- 2373.000 → 2438.000 (p=0.4427)
- 2497.000 → 2547.500 (p=0.4969)
- 2580.000 → 2722.000 (p=0.4924)
- 2801.500 → 2981.000 (p=0.5811)

### DEt3IRqqUVs
Truth calls: **12**; Pred calls: **2**; time_f1=0.968; seg_f1=0.000

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
- 5.000 → 346.500 (p=0.6718)
- 348.000 → 820.000 (p=0.5639)

### EVwBLXWlZiI
Truth calls: **1**; Pred calls: **0**; time_f1=0.000; seg_f1=0.000

Truth boundaries:
- 146.029 → 652.645

Pred boundaries (mean_p):
- (none)

### FBmODQn9grE
Truth calls: **1**; Pred calls: **0**; time_f1=0.000; seg_f1=0.000

Truth boundaries:
- 58.159 → 1207.413

Pred boundaries (mean_p):
- (none)

### FkgGv2iMjEo
Truth calls: **8**; Pred calls: **12**; time_f1=0.961; seg_f1=0.500

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
- 136.000 → 160.000 (p=0.5255)
- 160.500 → 253.500 (p=0.4506)
- 317.500 → 558.000 (p=0.5116)
- 654.000 → 810.000 (p=0.4702)
- 850.500 → 944.000 (p=0.5350)
- 1039.000 → 2844.000 (p=0.5191)
- 2845.500 → 3028.000 (p=0.5041)
- 3028.500 → 3053.500 (p=0.4704)
- 3093.000 → 3240.500 (p=0.5580)
- 3298.500 → 3370.500 (p=0.6346)
- 3402.000 → 3532.000 (p=0.5520)
- 3563.500 → 3623.500 (p=0.6027)

### FotSodGIVM8
Truth calls: **2**; Pred calls: **4**; time_f1=0.700; seg_f1=0.333

Truth boundaries:
- 33.524 → 366.326
- 461.912 → 633.082

Pred boundaries (mean_p):
- 33.000 → 394.000 (p=0.6916)
- 441.500 → 467.500 (p=0.3584)
- 1130.000 → 1158.000 (p=0.3318)
- 1566.000 → 1613.500 (p=0.2960)

### GcjtU_V-mW4
Truth calls: **2**; Pred calls: **2**; time_f1=0.802; seg_f1=1.000

Truth boundaries:
- 142.589 → 238.604
- 513.423 → 604.887

Pred boundaries (mean_p):
- 145.500 → 300.500 (p=0.5198)
- 522.500 → 596.500 (p=0.5807)

### JT1XSpK1pkM
Truth calls: **2**; Pred calls: **2**; time_f1=0.775; seg_f1=0.500

Truth boundaries:
- 40.397 → 64.726
- 335.183 → 357.848

Pred boundaries (mean_p):
- 53.000 → 69.500 (p=0.6242)
- 336.500 → 357.500 (p=0.5542)

### Krnsw9WZtRA
Truth calls: **8**; Pred calls: **4**; time_f1=0.771; seg_f1=0.333

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
- 29.000 → 63.000 (p=0.4899)
- 109.000 → 178.500 (p=0.4139)
- 250.500 → 313.500 (p=0.4614)
- 454.500 → 557.000 (p=0.4156)

### MPHdNIiB_8E
Truth calls: **0**; Pred calls: **0**; time_f1=1.000; seg_f1=1.000

Truth boundaries:
- (none)

Pred boundaries (mean_p):
- (none)

### N7XqeLuVOzk
Truth calls: **1**; Pred calls: **0**; time_f1=0.000; seg_f1=0.000

Truth boundaries:
- 86.673 → 1666.241

Pred boundaries (mean_p):
- (none)

### Ox_LARRQmlw
Truth calls: **1**; Pred calls: **2**; time_f1=0.998; seg_f1=0.667

Truth boundaries:
- 80.995 → 716.291

Pred boundaries (mean_p):
- 81.000 → 446.500 (p=0.7418)
- 447.000 → 714.500 (p=0.6384)

### PnTbJdhNbPk
Truth calls: **1**; Pred calls: **6**; time_f1=0.597; seg_f1=0.000

Truth boundaries:
- 15.854 → 1384.158

Pred boundaries (mean_p):
- 30.000 → 252.500 (p=0.4707)
- 292.500 → 340.000 (p=0.4249)
- 396.000 → 480.000 (p=0.4192)
- 511.000 → 598.000 (p=0.5953)
- 1203.500 → 1327.500 (p=0.3608)
- 1363.000 → 1392.000 (p=0.5909)

### Qa-ppZFUp0g
Truth calls: **27**; Pred calls: **11**; time_f1=0.907; seg_f1=0.211

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
- 64.500 → 489.500 (p=0.5244)
- 527.500 → 556.000 (p=0.6763)
- 589.500 → 600.500 (p=0.5196)
- 679.000 → 710.500 (p=0.5874)
- 748.500 → 772.000 (p=0.4177)
- 772.000 → 1010.500 (p=0.4806)
- 1042.000 → 1062.000 (p=0.4209)
- 1123.000 → 1149.500 (p=0.5336)
- 1189.500 → 1481.500 (p=0.4880)
- 1513.000 → 1524.500 (p=0.3643)
- 1577.500 → 1828.500 (p=0.6138)

### RL6Y5qig0Wg
Truth calls: **7**; Pred calls: **8**; time_f1=0.956; seg_f1=0.933

Truth boundaries:
- 88.710 → 222.038
- 455.360 → 559.472
- 638.357 → 675.494
- 766.126 → 779.070
- 893.717 → 1059.770
- 1157.827 → 1225.158
- 1434.781 → 1602.009

Pred boundaries (mean_p):
- 90.000 → 221.000 (p=0.6297)
- 458.000 → 560.500 (p=0.6659)
- 638.000 → 684.500 (p=0.6224)
- 765.500 → 779.000 (p=0.7956)
- 896.000 → 1091.500 (p=0.6386)
- 1163.000 → 1225.000 (p=0.6543)
- 1435.500 → 1580.000 (p=0.6316)
- 1581.000 → 1597.000 (p=0.5504)

### RtOncKiOT48
Truth calls: **2**; Pred calls: **5**; time_f1=0.698; seg_f1=0.000

Truth boundaries:
- 98.156 → 484.037
- 563.603 → 1483.472

Pred boundaries (mean_p):
- 16.000 → 27.000 (p=0.2462)
- 114.500 → 187.500 (p=0.5738)
- 228.000 → 367.000 (p=0.4201)
- 400.000 → 812.500 (p=0.5312)
- 862.000 → 1065.500 (p=0.4255)

### S8yFUyD_JXU
Truth calls: **1**; Pred calls: **3**; time_f1=0.807; seg_f1=0.000

Truth boundaries:
- 77.611 → 443.224

Pred boundaries (mean_p):
- 54.500 → 97.000 (p=0.4468)
- 144.000 → 320.000 (p=0.4542)
- 359.500 → 467.500 (p=0.4942)

### UWLJ81ezcpU
Truth calls: **7**; Pred calls: **9**; time_f1=0.767; seg_f1=0.750

Truth boundaries:
- 387.932 → 527.672
- 657.982 → 692.946
- 736.642 → 760.148
- 795.336 → 889.925
- 1033.707 → 1139.590
- 1199.640 → 1270.695
- 1313.627 → 1427.264

Pred boundaries (mean_p):
- 128.500 → 142.500 (p=0.4684)
- 390.500 → 471.000 (p=0.4792)
- 503.000 → 604.500 (p=0.4374)
- 662.000 → 690.500 (p=0.5191)
- 743.500 → 757.000 (p=0.6633)
- 797.500 → 941.500 (p=0.4847)
- 1018.000 → 1154.500 (p=0.5588)
- 1249.500 → 1277.000 (p=0.5760)
- 1313.000 → 1421.500 (p=0.5400)

### Uytq1t3zAz8
Truth calls: **1**; Pred calls: **1**; time_f1=0.977; seg_f1=1.000

Truth boundaries:
- 8.884 → 1098.453

Pred boundaries (mean_p):
- 9.500 → 1149.500 (p=0.6689)

### VXOSVmvwV0c
Truth calls: **3**; Pred calls: **1**; time_f1=0.969; seg_f1=0.000

Truth boundaries:
- 37.446 → 173.066
- 181.256 → 267.866
- 274.943 → 393.007

Pred boundaries (mean_p):
- 43.500 → 393.000 (p=0.6186)

### VZwqygti-nc
Truth calls: **4**; Pred calls: **1**; time_f1=0.894; seg_f1=0.000

Truth boundaries:
- 40.401 → 83.021
- 93.259 → 140.640
- 167.200 → 197.157
- 201.827 → 332.638

Pred boundaries (mean_p):
- 23.500 → 331.500 (p=0.4120)

### ZjhxWa0N4II
Truth calls: **3**; Pred calls: **1**; time_f1=0.979; seg_f1=0.500

Truth boundaries:
- 10.255 → 66.691
- 72.149 → 144.985
- 150.111 → 305.392

Pred boundaries (mean_p):
- 11.500 → 306.000 (p=0.6414)

### aW8jAYnvqyI
Truth calls: **0**; Pred calls: **0**; time_f1=1.000; seg_f1=1.000

Truth boundaries:
- (none)

Pred boundaries (mean_p):
- (none)

### bCxvXMhbFls
Truth calls: **1**; Pred calls: **1**; time_f1=0.054; seg_f1=0.000

Truth boundaries:
- 43.068 → 713.823

Pred boundaries (mean_p):
- 47.000 → 65.500 (p=0.5062)

### bPv8JzD_bMI
Truth calls: **4**; Pred calls: **7**; time_f1=0.737; seg_f1=0.545

Truth boundaries:
- 86.956 → 111.865
- 530.643 → 569.233
- 728.712 → 847.820
- 926.350 → 1034.779

Pred boundaries (mean_p):
- 57.500 → 137.000 (p=0.4036)
- 388.000 → 405.500 (p=0.3343)
- 515.000 → 564.500 (p=0.4803)
- 620.500 → 669.000 (p=0.4960)
- 739.000 → 767.000 (p=0.4883)
- 767.000 → 863.000 (p=0.5353)
- 918.000 → 1056.500 (p=0.5805)

### ci-FdcWiJiA
Truth calls: **7**; Pred calls: **14**; time_f1=0.544; seg_f1=0.286

Truth boundaries:
- 0.000 → 82.192
- 257.803 → 276.974
- 389.934 → 435.465
- 595.341 → 616.391
- 1020.462 → 1124.976
- 1752.450 → 1772.900
- 2034.918 → 2171.015

Pred boundaries (mean_p):
- 6.000 → 144.500 (p=0.4575)
- 192.500 → 273.000 (p=0.3957)
- 305.000 → 328.000 (p=0.4143)
- 360.000 → 430.000 (p=0.4483)
- 590.500 → 656.500 (p=0.4297)
- 744.500 → 759.000 (p=0.3829)
- 954.500 → 1181.000 (p=0.4717)
- 1224.500 → 1266.500 (p=0.2546)
- 1539.000 → 1568.000 (p=0.4062)
- 1618.000 → 1666.500 (p=0.3355)
- 1713.500 → 1780.500 (p=0.3647)
- 1818.500 → 1839.500 (p=0.4630)
- 2040.000 → 2225.000 (p=0.5369)
- 2322.500 → 2384.000 (p=0.3487)

### dZWKwNqDXIo
Truth calls: **5**; Pred calls: **6**; time_f1=0.980; seg_f1=0.909

Truth boundaries:
- 60.993 → 158.577
- 231.489 → 498.542
- 553.184 → 807.714
- 840.897 → 1031.735
- 1080.048 → 1727.524

Pred boundaries (mean_p):
- 61.500 → 159.500 (p=0.6411)
- 232.500 → 504.000 (p=0.7074)
- 553.500 → 816.500 (p=0.6465)
- 856.000 → 1042.500 (p=0.5815)
- 1080.000 → 1184.000 (p=0.5981)
- 1184.500 → 1743.000 (p=0.6820)

### dyFaLq3kfHs
Truth calls: **1**; Pred calls: **2**; time_f1=0.934; seg_f1=0.667

Truth boundaries:
- 73.850 → 499.583

Pred boundaries (mean_p):
- 59.000 → 93.500 (p=0.4602)
- 127.000 → 507.000 (p=0.6116)

### gDdoZC9Nhgk
Truth calls: **7**; Pred calls: **5**; time_f1=0.938; seg_f1=0.500

Truth boundaries:
- 96.069 → 150.692
- 257.311 → 466.293
- 468.043 → 507.117
- 512.702 → 549.139
- 697.043 → 902.660
- 908.561 → 964.391
- 978.786 → 1248.417

Pred boundaries (mean_p):
- 98.500 → 150.500 (p=0.5653)
- 260.500 → 545.000 (p=0.4404)
- 697.000 → 1042.000 (p=0.5576)
- 1074.500 → 1116.500 (p=0.5237)
- 1151.000 → 1249.000 (p=0.6055)

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
Truth calls: **1**; Pred calls: **3**; time_f1=0.959; seg_f1=0.000

Truth boundaries:
- 55.469 → 1045.232

Pred boundaries (mean_p):
- 58.500 → 318.000 (p=0.5870)
- 351.000 → 718.000 (p=0.6264)
- 756.000 → 1049.000 (p=0.6559)

### m2z1tgfv8Kc
Truth calls: **0**; Pred calls: **0**; time_f1=1.000; seg_f1=1.000

Truth boundaries:
- (none)

Pred boundaries (mean_p):
- (none)

### mxCrbMSfup4
Truth calls: **3**; Pred calls: **3**; time_f1=0.943; seg_f1=0.667

Truth boundaries:
- 40.697 → 707.536
- 745.894 → 763.517
- 813.441 → 1007.483

Pred boundaries (mean_p):
- 44.500 → 716.500 (p=0.5408)
- 747.500 → 943.500 (p=0.4976)
- 975.500 → 1012.500 (p=0.5468)

### nNA0XQO_7jQ
Truth calls: **0**; Pred calls: **0**; time_f1=1.000; seg_f1=1.000

Truth boundaries:
- (none)

Pred boundaries (mean_p):
- (none)

### pCcAlwOYYx8
Truth calls: **1**; Pred calls: **3**; time_f1=0.958; seg_f1=0.500

Truth boundaries:
- 19.288 → 1095.080

Pred boundaries (mean_p):
- 22.500 → 572.500 (p=0.5567)
- 615.000 → 643.500 (p=0.5310)
- 684.000 → 1096.000 (p=0.5266)

### pgYGE9jmNPA
Truth calls: **1**; Pred calls: **3**; time_f1=0.896; seg_f1=0.000

Truth boundaries:
- 43.409 → 909.003

Pred boundaries (mean_p):
- 43.500 → 474.000 (p=0.5859)
- 505.000 → 570.500 (p=0.3455)
- 682.000 → 933.500 (p=0.5449)

### pjf5uhOMcTc
Truth calls: **0**; Pred calls: **0**; time_f1=1.000; seg_f1=1.000

Truth boundaries:
- (none)

Pred boundaries (mean_p):
- (none)

