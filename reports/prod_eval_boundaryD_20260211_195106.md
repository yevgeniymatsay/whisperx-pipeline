# Call extractor production report

- git_sha: `5b231f6`
- split_config_path: `configs/call_extractor/split_v2.config.json`
- label_prefix: `labeling/corrected_boundaries/v2/`
- segments_s3_prefix: `call_extractor/wavlm_large_v1/models/run_20260209_034309_8105f4c/decoded_prod_boundaryD_20260211_195106/segments`
- probs_s3_prefix: `call_extractor/wavlm_large_v1/models/run_20260209_034309_8105f4c`
- match_tol_s: 0.25
- overlap_eps_s: 0.1
- min_coverage: 0.3

## Gate metrics (eval only; unchanged formulas)
- merges: 9
- oversplits: 62
- FP_total: 59
- keep_rate_iou_0_5: 0.139
- keep_rate_coverage: 0.152

## Production metrics (selection)
- fp_no_call_seconds_per_hour: 5.852
- fp_no_call_seconds: 17.620
- fp_no_call_segments: 4
- purity_segment_p10: 0.2890115934347507
- purity_segment_median: 1.0
- non_call_seconds_total: 845.820
- segments_per_hour: 79.219
- predicted_seconds_per_hour: 1574.339
- calls_coverage_ge_0_8_rate: 0.620
- call_coverage_p10: 0.4915245417300311
- call_coverage_median: 0.8729385524108314
- union_recall: 0.803
- union_purity: 0.934

## No-call per-video breakdown
- pjf5uhOMcTc: pred_segments=2 pred_seconds=9.580 audio_seconds=1550.594
- 15QzruVINDc: pred_segments=1 pred_seconds=4.240 audio_seconds=767.478
- aW8jAYnvqyI: pred_segments=1 pred_seconds=3.800 audio_seconds=2096.303
- 4OweikRF7bg: pred_segments=0 pred_seconds=0.000 audio_seconds=746.901
- MPHdNIiB_8E: pred_segments=0 pred_seconds=0.000 audio_seconds=501.376
- hYjbrxLXJIU: pred_segments=0 pred_seconds=0.000 audio_seconds=1888.641
- hs44BGJtOwg: pred_segments=0 pred_seconds=0.000 audio_seconds=898.148
- jvFLW5EClgk: pred_segments=0 pred_seconds=0.000 audio_seconds=644.416
- m2z1tgfv8Kc: pred_segments=0 pred_seconds=0.000 audio_seconds=1041.877
- nNA0XQO_7jQ: pred_segments=0 pred_seconds=0.000 audio_seconds=703.100

## Merge autopsy (eval only)
- merged_pred_segments: 9
- CfMJ01KP_ns pred=[196.76,199.10] gt_calls=2 min_in_call_smoothed=0.5023457407951355
  - boundary_t=198.874223 gap_dur=0.05660699999998542 min_gap_in_call=0.6603255271911621 max_start_p_win=0.7966435551643372 max_end_p_win=0.8512588739395142
- D4uiHjHW4AU pred=[2491.26,2529.72] gt_calls=2 min_in_call_smoothed=0.65077143907547
  - boundary_t=2527.034843 gap_dur=0.07043599999997241 min_gap_in_call=0.9556751251220703 max_start_p_win=0.18821369111537933 max_end_p_win=0.32686224579811096
- DEt3IRqqUVs pred=[49.82,126.12] gt_calls=2 min_in_call_smoothed=0.504867434501648
  - boundary_t=125.779573 gap_dur=0.0 min_gap_in_call=None max_start_p_win=0.226010262966156 max_end_p_win=0.8402538299560547
- Qa-ppZFUp0g pred=[816.12,828.62] gt_calls=2 min_in_call_smoothed=0.5023609399795532
  - boundary_t=820.141798 gap_dur=0.4111639999999852 min_gap_in_call=0.9446288347244263 max_start_p_win=0.11881387233734131 max_end_p_win=0.12298384308815002
- Qa-ppZFUp0g pred=[882.12,891.82] gt_calls=2 min_in_call_smoothed=0.5004051327705383
  - boundary_t=884.018508 gap_dur=0.2909700000000157 min_gap_in_call=0.950313150882721 max_start_p_win=0.8067461848258972 max_end_p_win=0.3157997727394104
- Qa-ppZFUp0g pred=[902.48,924.26] gt_calls=2 min_in_call_smoothed=0.5080021619796753
  - boundary_t=912.600822 gap_dur=3.80585700000006 min_gap_in_call=0.9146504402160645 max_start_p_win=0.13941225409507751 max_end_p_win=0.14249619841575623
- Qa-ppZFUp0g pred=[1279.02,1291.70] gt_calls=2 min_in_call_smoothed=0.5162327289581299
  - boundary_t=1287.211399 gap_dur=0.0 min_gap_in_call=None max_start_p_win=0.1725357621908188 max_end_p_win=0.16295750439167023
- Qa-ppZFUp0g pred=[1320.98,1374.76] gt_calls=2 min_in_call_smoothed=0.47594165802001953
  - boundary_t=1370.04968 gap_dur=0.4083689999999933 min_gap_in_call=0.9523142576217651 max_start_p_win=0.13915112614631653 max_end_p_win=0.16157008707523346
- Qa-ppZFUp0g pred=[1436.94,1468.62] gt_calls=2 min_in_call_smoothed=0.5039772987365723
  - boundary_t=1461.092285 gap_dur=0.18829900000014277 min_gap_in_call=0.9426816701889038 max_start_p_win=0.12623342871665955 max_end_p_win=0.11970400810241699
