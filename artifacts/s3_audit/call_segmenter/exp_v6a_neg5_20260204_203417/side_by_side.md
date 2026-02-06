# exp_v6a_neg5: Truth vs Predicted Call Boundaries (side-by-side)

Bucket (pred): `rezora-whisperx-us-east-1-864981718771`
Predictions: `s3://rezora-whisperx-us-east-1-864981718771/call_segmenter/predictions/exp_v6a_neg5_20260204_202830_thr70_off50_gap30_p20_min10/`
Labels: `s3://rezora-whisperx-us-east-1-864981718771/labeling/corrected_boundaries/v1/`

Format:
- Truth boundaries are your manual call labels.
- Pred boundaries are model outputs (post-processed) with mean probability `p` across the segment.

## Top 10 by Absolute Call-Count Delta

| video_id | truth_calls | pred_calls | delta | time_f1 | seg_f1 |
|---|---:|---:|---:|---:|---:|
| `Qa-ppZFUp0g` | 27 | 10 | -17 | 0.924 | 0.270 |
| `DEt3IRqqUVs` | 12 | 1 | -11 | 0.969 | 0.000 |
| `D4uiHjHW4AU` | 19 | 10 | -9 | 0.928 | 0.345 |
| `RtOncKiOT48` | 2 | 9 | +7 | 0.718 | 0.000 |
| `9g2gwfWIbUk` | 2 | 8 | +6 | 0.980 | 0.200 |
| `ci-FdcWiJiA` | 7 | 13 | +6 | 0.633 | 0.500 |
| `4ipwbOJRMck` | 1 | 6 | +5 | 0.948 | 0.286 |
| `UWLJ81ezcpU` | 7 | 11 | +4 | 0.837 | 0.556 |
| `-POaWp9_UaM` | 4 | 1 | -3 | 0.992 | 0.000 |
| `Krnsw9WZtRA` | 8 | 5 | -3 | 0.877 | 0.615 |

## Top 10 Worst by Segment IoU-F1

| video_id | truth_calls | pred_calls | time_f1 | seg_f1 | unmatched_pred | unmatched_truth |
|---|---:|---:|---:|---:|---:|---:|
| `-POaWp9_UaM` | 4 | 1 | 0.992 | 0.000 | 1 | 4 |
| `DEt3IRqqUVs` | 12 | 1 | 0.969 | 0.000 | 1 | 12 |
| `PnTbJdhNbPk` | 1 | 2 | 0.702 | 0.000 | 2 | 1 |
| `RtOncKiOT48` | 2 | 9 | 0.718 | 0.000 | 9 | 2 |
| `VZwqygti-nc` | 4 | 1 | 0.905 | 0.000 | 1 | 4 |
| `bCxvXMhbFls` | 1 | 1 | 0.054 | 0.000 | 1 | 1 |
| `pCcAlwOYYx8` | 1 | 3 | 0.946 | 0.000 | 3 | 1 |
| `9g2gwfWIbUk` | 2 | 8 | 0.980 | 0.200 | 7 | 1 |
| `Qa-ppZFUp0g` | 27 | 10 | 0.924 | 0.270 | 5 | 22 |
| `4ipwbOJRMck` | 1 | 6 | 0.948 | 0.286 | 5 | 0 |

## Counts Summary (all videos)

| video_id | truth_calls | pred_calls |
|---|---:|---:|
| `-POaWp9_UaM` | 4 | 1 |
| `-UzXRV8nHn4` | 1 | 1 |
| `15QzruVINDc` | 0 | 0 |
| `4OweikRF7bg` | 0 | 0 |
| `4ipwbOJRMck` | 1 | 6 |
| `6L-8f1eYOOY` | 1 | 1 |
| `6UWzVyNtbS0` | 1 | 1 |
| `9g2gwfWIbUk` | 2 | 8 |
| `CfMJ01KP_ns` | 6 | 8 |
| `D4uiHjHW4AU` | 19 | 10 |
| `DEt3IRqqUVs` | 12 | 1 |
| `FkgGv2iMjEo` | 8 | 9 |
| `FotSodGIVM8` | 2 | 2 |
| `GcjtU_V-mW4` | 2 | 3 |
| `JT1XSpK1pkM` | 2 | 2 |
| `Krnsw9WZtRA` | 8 | 5 |
| `MPHdNIiB_8E` | 0 | 0 |
| `Ox_LARRQmlw` | 1 | 1 |
| `PnTbJdhNbPk` | 1 | 2 |
| `Qa-ppZFUp0g` | 27 | 10 |
| `RL6Y5qig0Wg` | 7 | 9 |
| `RtOncKiOT48` | 2 | 9 |
| `S8yFUyD_JXU` | 1 | 2 |
| `UWLJ81ezcpU` | 7 | 11 |
| `Uytq1t3zAz8` | 1 | 1 |
| `VXOSVmvwV0c` | 3 | 2 |
| `VZwqygti-nc` | 4 | 1 |
| `ZjhxWa0N4II` | 3 | 2 |
| `aW8jAYnvqyI` | 0 | 0 |
| `bCxvXMhbFls` | 1 | 1 |
| `bPv8JzD_bMI` | 4 | 7 |
| `ci-FdcWiJiA` | 7 | 13 |
| `dZWKwNqDXIo` | 5 | 7 |
| `dyFaLq3kfHs` | 1 | 2 |
| `gDdoZC9Nhgk` | 7 | 6 |
| `hYjbrxLXJIU` | 0 | 0 |
| `hs44BGJtOwg` | 0 | 0 |
| `jvFLW5EClgk` | 0 | 0 |
| `lioFo9pz9x8` | 1 | 2 |
| `m2z1tgfv8Kc` | 0 | 0 |
| `mxCrbMSfup4` | 3 | 6 |
| `nNA0XQO_7jQ` | 0 | 0 |
| `pCcAlwOYYx8` | 1 | 3 |
| `pgYGE9jmNPA` | 1 | 4 |
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
- 76.000 → 683.500 (p=0.6709)

### -UzXRV8nHn4
Truth calls: **1**; Pred calls: **1**; time_f1=0.998; seg_f1=1.000

Truth boundaries:
- 181.119 → 430.160

Pred boundaries (mean_p):
- 182.000 → 430.000 (p=0.9102)

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
Truth calls: **1**; Pred calls: **6**; time_f1=0.948; seg_f1=0.286

Truth boundaries:
- 75.044 → 2535.511

Pred boundaries (mean_p):
- 80.000 → 1647.500 (p=0.8267)
- 1697.000 → 2025.500 (p=0.8532)
- 2025.500 → 2628.500 (p=0.7210)
- 2661.000 → 2689.500 (p=0.3493)
- 2767.500 → 2814.500 (p=0.3047)
- 2850.500 → 2889.500 (p=0.3400)

### 6L-8f1eYOOY
Truth calls: **1**; Pred calls: **1**; time_f1=0.999; seg_f1=1.000

Truth boundaries:
- 0.000 → 1044.431

Pred boundaries (mean_p):
- 0.500 → 1045.000 (p=0.7494)

### 6UWzVyNtbS0
Truth calls: **1**; Pred calls: **1**; time_f1=0.959; seg_f1=1.000

Truth boundaries:
- 68.833 → 199.138

Pred boundaries (mean_p):
- 75.000 → 195.000 (p=0.4254)

### 9g2gwfWIbUk
Truth calls: **2**; Pred calls: **8**; time_f1=0.980; seg_f1=0.200

Truth boundaries:
- 0.000 → 934.003
- 937.520 → 1955.949

Pred boundaries (mean_p):
- 2.500 → 126.000 (p=0.8030)
- 126.000 → 900.000 (p=0.6922)
- 902.500 → 981.000 (p=0.7057)
- 982.000 → 1023.000 (p=0.6936)
- 1028.000 → 1053.000 (p=0.6926)
- 1055.000 → 1530.500 (p=0.6843)
- 1538.500 → 1704.000 (p=0.6593)
- 1704.500 → 2008.000 (p=0.5543)

### CfMJ01KP_ns
Truth calls: **6**; Pred calls: **8**; time_f1=0.978; seg_f1=0.714

Truth boundaries:
- 44.818 → 198.874
- 198.931 → 635.358
- 729.783 → 967.727
- 970.201 → 1538.415
- 1539.348 → 2231.910
- 2232.687 → 2654.263

Pred boundaries (mean_p):
- 51.500 → 140.500 (p=0.6930)
- 140.500 → 635.500 (p=0.7843)
- 734.500 → 1104.500 (p=0.7041)
- 1104.500 → 1213.000 (p=0.5321)
- 1272.000 → 1353.500 (p=0.7839)
- 1355.000 → 1682.500 (p=0.7231)
- 1716.000 → 2087.000 (p=0.7229)
- 2087.000 → 2654.500 (p=0.7332)

### D4uiHjHW4AU
Truth calls: **19**; Pred calls: **10**; time_f1=0.928; seg_f1=0.345

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
- 276.000 → 363.000 (p=0.3343)
- 436.500 → 470.000 (p=0.4165)
- 472.000 → 607.500 (p=0.5114)
- 718.000 → 978.000 (p=0.5040)
- 1016.500 → 1035.000 (p=0.4594)
- 1067.000 → 1768.000 (p=0.6050)
- 1863.000 → 2325.500 (p=0.5212)
- 2372.000 → 2418.000 (p=0.4268)
- 2493.500 → 2722.000 (p=0.4487)
- 2780.500 → 2980.000 (p=0.5583)

### DEt3IRqqUVs
Truth calls: **12**; Pred calls: **1**; time_f1=0.969; seg_f1=0.000

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
- 5.000 → 820.000 (p=0.6752)

### FkgGv2iMjEo
Truth calls: **8**; Pred calls: **9**; time_f1=0.986; seg_f1=0.471

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
- 145.000 → 253.500 (p=0.5562)
- 317.000 → 560.000 (p=0.6426)
- 654.000 → 806.000 (p=0.6161)
- 861.500 → 963.000 (p=0.5873)
- 1037.500 → 1600.000 (p=0.5694)
- 1600.500 → 1650.000 (p=0.8168)
- 1650.000 → 2616.000 (p=0.6747)
- 2616.000 → 2789.500 (p=0.6194)
- 2799.000 → 3623.500 (p=0.6214)

### FotSodGIVM8
Truth calls: **2**; Pred calls: **2**; time_f1=0.762; seg_f1=0.500

Truth boundaries:
- 33.524 → 366.326
- 461.912 → 633.082

Pred boundaries (mean_p):
- 38.500 → 366.500 (p=0.7037)
- 1130.000 → 1158.000 (p=0.2444)

### GcjtU_V-mW4
Truth calls: **2**; Pred calls: **3**; time_f1=0.905; seg_f1=0.800

Truth boundaries:
- 142.589 → 238.604
- 513.423 → 604.887

Pred boundaries (mean_p):
- 149.500 → 163.500 (p=0.5501)
- 163.500 → 269.000 (p=0.5436)
- 513.500 → 604.500 (p=0.5229)

### JT1XSpK1pkM
Truth calls: **2**; Pred calls: **2**; time_f1=0.949; seg_f1=1.000

Truth boundaries:
- 40.397 → 64.726
- 335.183 → 357.848

Pred boundaries (mean_p):
- 40.000 → 68.500 (p=0.5449)
- 335.000 → 358.500 (p=0.7098)

### Krnsw9WZtRA
Truth calls: **8**; Pred calls: **5**; time_f1=0.877; seg_f1=0.615

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
- 19.500 → 40.500 (p=0.4144)
- 40.500 → 61.500 (p=0.6988)
- 108.000 → 179.000 (p=0.5483)
- 249.000 → 354.500 (p=0.4189)
- 487.500 → 558.500 (p=0.6038)

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
- 81.500 → 715.500 (p=0.7380)

### PnTbJdhNbPk
Truth calls: **1**; Pred calls: **2**; time_f1=0.702; seg_f1=0.000

Truth boundaries:
- 15.854 → 1384.158

Pred boundaries (mean_p):
- 24.500 → 599.500 (p=0.5458)
- 1218.500 → 1384.500 (p=0.5085)

### Qa-ppZFUp0g
Truth calls: **27**; Pred calls: **10**; time_f1=0.924; seg_f1=0.270

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
- 64.500 → 489.000 (p=0.5699)
- 527.500 → 600.500 (p=0.4703)
- 679.000 → 710.500 (p=0.6083)
- 751.500 → 877.000 (p=0.4587)
- 877.000 → 893.500 (p=0.7158)
- 893.500 → 1010.500 (p=0.4017)
- 1042.000 → 1061.500 (p=0.3720)
- 1123.000 → 1149.500 (p=0.5354)
- 1189.500 → 1481.500 (p=0.4688)
- 1577.500 → 1829.000 (p=0.5669)

### RL6Y5qig0Wg
Truth calls: **7**; Pred calls: **9**; time_f1=0.987; seg_f1=0.875

Truth boundaries:
- 88.710 → 222.038
- 455.360 → 559.472
- 638.357 → 675.494
- 766.126 → 779.070
- 893.717 → 1059.770
- 1157.827 → 1225.158
- 1434.781 → 1602.009

Pred boundaries (mean_p):
- 89.000 → 221.500 (p=0.6508)
- 458.000 → 494.000 (p=0.6863)
- 498.500 → 561.500 (p=0.6788)
- 638.000 → 676.500 (p=0.7864)
- 765.500 → 779.000 (p=0.8074)
- 896.000 → 1060.000 (p=0.7493)
- 1160.000 → 1225.000 (p=0.6298)
- 1434.500 → 1540.000 (p=0.6681)
- 1540.500 → 1602.500 (p=0.5288)

### RtOncKiOT48
Truth calls: **2**; Pred calls: **9**; time_f1=0.718; seg_f1=0.000

Truth boundaries:
- 98.156 → 484.037
- 563.603 → 1483.472

Pred boundaries (mean_p):
- 1.500 → 27.000 (p=0.5357)
- 100.500 → 187.500 (p=0.4762)
- 234.500 → 250.500 (p=0.5615)
- 250.500 → 359.000 (p=0.4054)
- 359.500 → 581.500 (p=0.5470)
- 581.500 → 604.500 (p=0.6964)
- 604.500 → 618.000 (p=0.8207)
- 620.500 → 816.500 (p=0.6828)
- 862.000 → 1066.500 (p=0.5917)

### S8yFUyD_JXU
Truth calls: **1**; Pred calls: **2**; time_f1=0.820; seg_f1=0.667

Truth boundaries:
- 77.611 → 443.224

Pred boundaries (mean_p):
- 28.500 → 92.000 (p=0.3513)
- 152.000 → 468.000 (p=0.4632)

### UWLJ81ezcpU
Truth calls: **7**; Pred calls: **11**; time_f1=0.837; seg_f1=0.556

Truth boundaries:
- 387.932 → 527.672
- 657.982 → 692.946
- 736.642 → 760.148
- 795.336 → 889.925
- 1033.707 → 1139.590
- 1199.640 → 1270.695
- 1313.627 → 1427.264

Pred boundaries (mean_p):
- 390.500 → 471.000 (p=0.4064)
- 503.500 → 549.500 (p=0.5138)
- 586.000 → 604.500 (p=0.3873)
- 662.000 → 690.500 (p=0.5485)
- 747.500 → 758.000 (p=0.6929)
- 798.000 → 922.000 (p=0.5171)
- 1033.000 → 1080.000 (p=0.4555)
- 1080.000 → 1146.500 (p=0.6452)
- 1193.000 → 1211.000 (p=0.3301)
- 1248.000 → 1269.500 (p=0.6531)
- 1314.000 → 1421.500 (p=0.5059)

### Uytq1t3zAz8
Truth calls: **1**; Pred calls: **1**; time_f1=0.999; seg_f1=1.000

Truth boundaries:
- 8.884 → 1098.453

Pred boundaries (mean_p):
- 8.500 → 1099.500 (p=0.6091)

### VXOSVmvwV0c
Truth calls: **3**; Pred calls: **2**; time_f1=0.962; seg_f1=0.400

Truth boundaries:
- 37.446 → 173.066
- 181.256 → 267.866
- 274.943 → 393.007

Pred boundaries (mean_p):
- 48.000 → 324.000 (p=0.5219)
- 324.000 → 393.000 (p=0.6545)

### VZwqygti-nc
Truth calls: **4**; Pred calls: **1**; time_f1=0.905; seg_f1=0.000

Truth boundaries:
- 40.401 → 83.021
- 93.259 → 140.640
- 167.200 → 197.157
- 201.827 → 332.638

Pred boundaries (mean_p):
- 48.500 → 331.500 (p=0.5129)

### ZjhxWa0N4II
Truth calls: **3**; Pred calls: **2**; time_f1=0.952; seg_f1=0.800

Truth boundaries:
- 10.255 → 66.691
- 72.149 → 144.985
- 150.111 → 305.392

Pred boundaries (mean_p):
- 11.500 → 145.000 (p=0.7683)
- 149.500 → 326.500 (p=0.7520)

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
- 33.500 → 62.000 (p=0.3932)

### bPv8JzD_bMI
Truth calls: **4**; Pred calls: **7**; time_f1=0.738; seg_f1=0.545

Truth boundaries:
- 86.956 → 111.865
- 530.643 → 569.233
- 728.712 → 847.820
- 926.350 → 1034.779

Pred boundaries (mean_p):
- 86.000 → 110.000 (p=0.5299)
- 110.000 → 155.500 (p=0.4316)
- 475.500 → 567.000 (p=0.4330)
- 614.500 → 669.000 (p=0.6332)
- 742.000 → 803.500 (p=0.3632)
- 803.500 → 845.500 (p=0.6772)
- 926.000 → 1056.500 (p=0.6343)

### ci-FdcWiJiA
Truth calls: **7**; Pred calls: **13**; time_f1=0.633; seg_f1=0.500

Truth boundaries:
- 0.000 → 82.192
- 257.803 → 276.974
- 389.934 → 435.465
- 595.341 → 616.391
- 1020.462 → 1124.976
- 1752.450 → 1772.900
- 2034.918 → 2171.015

Pred boundaries (mean_p):
- 6.000 → 28.500 (p=0.6410)
- 28.500 → 104.500 (p=0.5946)
- 186.500 → 322.000 (p=0.3983)
- 384.500 → 432.000 (p=0.4870)
- 595.500 → 682.500 (p=0.3377)
- 891.500 → 926.000 (p=0.3266)
- 967.500 → 1151.000 (p=0.4973)
- 1258.500 → 1308.000 (p=0.2857)
- 1544.000 → 1569.000 (p=0.3257)
- 1609.000 → 1656.000 (p=0.2670)
- 1753.000 → 1781.000 (p=0.4559)
- 1935.000 → 1957.000 (p=0.2070)
- 2034.500 → 2171.000 (p=0.6637)

### dZWKwNqDXIo
Truth calls: **5**; Pred calls: **7**; time_f1=0.987; seg_f1=0.833

Truth boundaries:
- 60.993 → 158.577
- 231.489 → 498.542
- 553.184 → 807.714
- 840.897 → 1031.735
- 1080.048 → 1727.524

Pred boundaries (mean_p):
- 61.500 → 178.500 (p=0.6418)
- 232.500 → 501.500 (p=0.8525)
- 553.000 → 574.000 (p=0.8001)
- 574.500 → 808.000 (p=0.7012)
- 849.000 → 1028.500 (p=0.6219)
- 1080.000 → 1682.500 (p=0.7368)
- 1682.500 → 1727.500 (p=0.8399)

### dyFaLq3kfHs
Truth calls: **1**; Pred calls: **2**; time_f1=0.946; seg_f1=0.667

Truth boundaries:
- 73.850 → 499.583

Pred boundaries (mean_p):
- 74.000 → 85.500 (p=0.6738)
- 127.000 → 502.000 (p=0.7494)

### gDdoZC9Nhgk
Truth calls: **7**; Pred calls: **6**; time_f1=0.982; seg_f1=0.615

Truth boundaries:
- 96.069 → 150.692
- 257.311 → 466.293
- 468.043 → 507.117
- 512.702 → 549.139
- 697.043 → 902.660
- 908.561 → 964.391
- 978.786 → 1248.417

Pred boundaries (mean_p):
- 97.500 → 150.500 (p=0.7270)
- 257.000 → 525.000 (p=0.5098)
- 526.000 → 549.500 (p=0.5628)
- 697.000 → 932.000 (p=0.5468)
- 932.000 → 1114.000 (p=0.5808)
- 1114.000 → 1248.500 (p=0.5144)

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
Truth calls: **1**; Pred calls: **2**; time_f1=0.982; seg_f1=0.667

Truth boundaries:
- 55.469 → 1045.232

Pred boundaries (mean_p):
- 58.500 → 319.000 (p=0.6475)
- 351.000 → 1046.000 (p=0.6640)

### m2z1tgfv8Kc
Truth calls: **0**; Pred calls: **0**; time_f1=1.000; seg_f1=1.000

Truth boundaries:
- (none)

Pred boundaries (mean_p):
- (none)

### mxCrbMSfup4
Truth calls: **3**; Pred calls: **6**; time_f1=0.936; seg_f1=0.667

Truth boundaries:
- 40.697 → 707.536
- 745.894 → 763.517
- 813.441 → 1007.483

Pred boundaries (mean_p):
- 45.000 → 536.000 (p=0.5756)
- 584.000 → 639.500 (p=0.5996)
- 684.000 → 707.000 (p=0.5934)
- 749.000 → 762.000 (p=0.6444)
- 813.000 → 856.500 (p=0.5429)
- 859.000 → 1008.000 (p=0.4960)

### nNA0XQO_7jQ
Truth calls: **0**; Pred calls: **0**; time_f1=1.000; seg_f1=1.000

Truth boundaries:
- (none)

Pred boundaries (mean_p):
- (none)

### pCcAlwOYYx8
Truth calls: **1**; Pred calls: **3**; time_f1=0.946; seg_f1=0.000

Truth boundaries:
- 19.288 → 1095.080

Pred boundaries (mean_p):
- 22.500 → 550.000 (p=0.5853)
- 613.500 → 643.000 (p=0.4681)
- 685.500 → 1095.500 (p=0.4828)

### pgYGE9jmNPA
Truth calls: **1**; Pred calls: **4**; time_f1=0.968; seg_f1=0.400

Truth boundaries:
- 43.409 → 909.003

Pred boundaries (mean_p):
- 30.500 → 607.000 (p=0.6738)
- 609.500 → 909.000 (p=0.6011)
- 968.500 → 986.000 (p=0.3436)
- 1156.000 → 1180.500 (p=0.3546)

### pjf5uhOMcTc
Truth calls: **0**; Pred calls: **0**; time_f1=1.000; seg_f1=1.000

Truth boundaries:
- (none)

Pred boundaries (mean_p):
- (none)

