# ab_pilot9_az_notext_spectral: Truth vs Predicted Call Boundaries (side-by-side)

Bucket (pred): `rezora-whisperx-us-east-1-864981718771`
Predictions: `s3://rezora-whisperx-us-east-1-864981718771/call_segmenter/predictions/ab_pilot9_az_notext_spectral_20260206_042222_thr55_off45_gap02_gmax_p00_min03/`
Labels: `s3://rezora-whisperx-us-east-1-864981718771/labeling/corrected_boundaries/v1/`

Model git sha: `a32c097a88c2`
Decode: `threshold` (thr_on=0.55, thr_off=0.45000000000000007, gap_merge_s=2.0, gap_stat=max, gap_min_p=0.0, min_seg_s=3.0)

Format:
- Truth boundaries are your manual call labels.
- Pred boundaries are model outputs (post-processed) with mean probability `p` across the segment.

## Top 10 by Absolute Call-Count Delta

| video_id | truth_calls | pred_calls | delta | time_f1 | seg_f1 |
|---|---:|---:|---:|---:|---:|
| `4OweikRF7bg` | 0 | 19 | +19 | 0.000 | 0.000 |
| `bPv8JzD_bMI` | 4 | 21 | +17 | 0.633 | 0.240 |
| `-POaWp9_UaM` | 4 | 15 | +11 | 0.922 | 0.211 |
| `9g2gwfWIbUk` | 2 | 3 | +1 | 0.999 | 0.800 |
| `VZwqygti-nc` | 4 | 5 | +1 | 0.980 | 0.889 |
| `ZjhxWa0N4II` | 3 | 2 | -1 | 0.985 | 0.800 |
| `15QzruVINDc` | 0 | 0 | +0 | 1.000 | 1.000 |
| `MPHdNIiB_8E` | 0 | 0 | +0 | 1.000 | 1.000 |
| `VXOSVmvwV0c` | 3 | 3 | +0 | 0.994 | 1.000 |

## Top 10 Worst by Segment IoU-F1

| video_id | truth_calls | pred_calls | time_f1 | seg_f1 | unmatched_pred | unmatched_truth |
|---|---:|---:|---:|---:|---:|---:|
| `4OweikRF7bg` | 0 | 19 | 0.000 | 0.000 | 19 | 0 |
| `-POaWp9_UaM` | 4 | 15 | 0.922 | 0.211 | 13 | 2 |
| `bPv8JzD_bMI` | 4 | 21 | 0.633 | 0.240 | 18 | 1 |
| `9g2gwfWIbUk` | 2 | 3 | 0.999 | 0.800 | 1 | 0 |
| `ZjhxWa0N4II` | 3 | 2 | 0.985 | 0.800 | 0 | 1 |
| `VZwqygti-nc` | 4 | 5 | 0.980 | 0.889 | 1 | 0 |
| `15QzruVINDc` | 0 | 0 | 1.000 | 1.000 | 0 | 0 |
| `MPHdNIiB_8E` | 0 | 0 | 1.000 | 1.000 | 0 | 0 |
| `VXOSVmvwV0c` | 3 | 3 | 0.994 | 1.000 | 0 | 0 |

## Counts Summary (all videos)

| video_id | truth_calls | pred_calls |
|---|---:|---:|
| `-POaWp9_UaM` | 4 | 15 |
| `15QzruVINDc` | 0 | 0 |
| `4OweikRF7bg` | 0 | 19 |
| `9g2gwfWIbUk` | 2 | 3 |
| `MPHdNIiB_8E` | 0 | 0 |
| `VXOSVmvwV0c` | 3 | 3 |
| `VZwqygti-nc` | 4 | 5 |
| `ZjhxWa0N4II` | 3 | 2 |
| `bPv8JzD_bMI` | 4 | 21 |

## Detailed Boundaries

### -POaWp9_UaM
Truth calls: **4**; Pred calls: **15**; time_f1=0.922; seg_f1=0.211

Truth boundaries:
- 68.724 → 214.650
- 216.061 → 352.478
- 352.592 → 404.942
- 406.142 → 683.371

Pred boundaries (mean_p):
- 5.500 → 14.000 (p=0.4996)
- 26.500 → 30.000 (p=0.2382)
- 56.500 → 61.500 (p=0.4808)
- 70.000 → 79.500 (p=0.9174)
- 83.500 → 87.500 (p=0.8454)
- 90.500 → 244.000 (p=0.8373)
- 248.000 → 329.500 (p=0.8799)
- 335.000 → 368.500 (p=0.7326)
- 375.500 → 513.500 (p=0.7649)
- 521.000 → 525.500 (p=0.8194)
- 531.000 → 577.000 (p=0.7437)
- 585.000 → 630.500 (p=0.7159)
- 637.500 → 696.000 (p=0.6251)
- 698.500 → 703.500 (p=0.5358)
- 706.500 → 711.000 (p=0.2828)

### 15QzruVINDc
Truth calls: **0**; Pred calls: **0**; time_f1=1.000; seg_f1=1.000

Truth boundaries:
- (none)

Pred boundaries (mean_p):
- (none)

### 4OweikRF7bg
Truth calls: **0**; Pred calls: **19**; time_f1=0.000; seg_f1=0.000

Truth boundaries:
- (none)

Pred boundaries (mean_p):
- 0.500 → 41.000 (p=0.5301)
- 54.000 → 58.000 (p=0.4186)
- 61.500 → 67.500 (p=0.4313)
- 71.500 → 113.000 (p=0.6594)
- 169.000 → 271.000 (p=0.7846)
- 286.500 → 296.000 (p=0.3541)
- 298.500 → 307.000 (p=0.4386)
- 309.500 → 323.000 (p=0.6057)
- 325.500 → 328.500 (p=0.7711)
- 332.500 → 340.500 (p=0.4432)
- 356.500 → 426.000 (p=0.5335)
- 475.000 → 484.000 (p=0.3911)
- 500.500 → 510.000 (p=0.1940)
- 541.500 → 564.000 (p=0.4451)
- 566.500 → 569.500 (p=0.5657)
- 588.000 → 591.000 (p=0.7183)
- 594.000 → 648.500 (p=0.7272)
- 715.500 → 729.500 (p=0.5030)
- 733.500 → 746.000 (p=0.6187)

### 9g2gwfWIbUk
Truth calls: **2**; Pred calls: **3**; time_f1=0.999; seg_f1=0.800

Truth boundaries:
- 0.000 → 934.003
- 937.520 → 1955.949

Pred boundaries (mean_p):
- 0.000 → 934.500 (p=0.9862)
- 938.000 → 1855.000 (p=0.9744)
- 1857.500 → 1955.500 (p=0.9892)

### MPHdNIiB_8E
Truth calls: **0**; Pred calls: **0**; time_f1=1.000; seg_f1=1.000

Truth boundaries:
- (none)

Pred boundaries (mean_p):
- (none)

### VXOSVmvwV0c
Truth calls: **3**; Pred calls: **3**; time_f1=0.994; seg_f1=1.000

Truth boundaries:
- 37.446 → 173.066
- 181.256 → 267.866
- 274.943 → 393.007

Pred boundaries (mean_p):
- 37.500 → 174.000 (p=0.9657)
- 180.500 → 269.000 (p=0.9367)
- 274.000 → 393.000 (p=0.9588)

### VZwqygti-nc
Truth calls: **4**; Pred calls: **5**; time_f1=0.980; seg_f1=0.889

Truth boundaries:
- 40.401 → 83.021
- 93.259 → 140.640
- 167.200 → 197.157
- 201.827 → 332.638

Pred boundaries (mean_p):
- 42.000 → 83.500 (p=0.9065)
- 92.500 → 141.500 (p=0.9392)
- 166.000 → 198.000 (p=0.9730)
- 201.000 → 220.500 (p=0.8296)
- 223.000 → 333.500 (p=0.9044)

### ZjhxWa0N4II
Truth calls: **3**; Pred calls: **2**; time_f1=0.985; seg_f1=0.800

Truth boundaries:
- 10.255 → 66.691
- 72.149 → 144.985
- 150.111 → 305.392

Pred boundaries (mean_p):
- 9.500 → 67.000 (p=0.9633)
- 71.000 → 306.500 (p=0.9393)

### bPv8JzD_bMI
Truth calls: **4**; Pred calls: **21**; time_f1=0.633; seg_f1=0.240

Truth boundaries:
- 86.956 → 111.865
- 530.643 → 569.233
- 728.712 → 847.820
- 926.350 → 1034.779

Pred boundaries (mean_p):
- 21.500 → 26.500 (p=0.4261)
- 94.000 → 97.500 (p=0.4649)
- 100.000 → 130.000 (p=0.7494)
- 219.000 → 222.000 (p=0.3494)
- 237.500 → 242.500 (p=0.5363)
- 355.000 → 369.000 (p=0.6413)
- 371.500 → 376.500 (p=0.6234)
- 388.000 → 394.500 (p=0.5172)
- 403.000 → 434.500 (p=0.7429)
- 474.500 → 522.000 (p=0.7853)
- 524.500 → 572.000 (p=0.7553)
- 579.000 → 584.500 (p=0.5691)
- 589.000 → 593.000 (p=0.3467)
- 614.000 → 679.000 (p=0.8886)
- 716.000 → 725.000 (p=0.6790)
- 728.500 → 880.500 (p=0.7812)
- 917.000 → 1035.500 (p=0.8908)
- 1043.000 → 1065.500 (p=0.6183)
- 1075.500 → 1089.500 (p=0.6326)
- 1092.500 → 1095.500 (p=0.5117)
- 1098.000 → 1104.000 (p=0.4948)

