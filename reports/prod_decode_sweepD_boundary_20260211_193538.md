# Production decode sweep report

- run_s3_prefix: `call_extractor/wavlm_large_v1/models/run_20260209_034309_8105f4c`
- split_config_path: `configs/call_extractor/split_v2.config.json`
- metrics_version: `gate_metrics_v1`
- gate_policy_version: `strict_gates_v1_fp_total`
- match_tol_s: 0.25
- overlap_eps_s: 0.1
- min_coverage: 0.3
- grid_size: 64

## Best config (ranked by objective)
- merges: 9
- calls_coverage_ge_0_8_rate: 0.620
- union_recall: 0.803
- union_purity: 0.934
- non_call_seconds_total: 845.820
- fp_no_call_seconds_per_hour (reporting): 5.852
- segments_per_hour: 79.220
- keep_rate_iou_0_5 (reporting-only): 0.139
- params: `{"boundary_end_thr": 0.2, "boundary_nms_sep_s": 0.5, "boundary_pair_max_gap_s": 5.0, "boundary_score_thr": 0.7, "boundary_smooth_win_s": 0.1, "boundary_split_gap_s": 0.2, "boundary_split_margin_s": 0.0, "boundary_start_thr": 0.2, "max_in_call_min": 0.7, "max_segment_s": 900.0, "mean_in_call_min": 0.45, "split_lo": 0.3, "split_min_off_s": 0.25, "trim_s": 0.0, "viterbi_cost_mult": 0.5, "viterbi_min_off_s": 0.0, "viterbi_min_on_s": 0.5, "viterbi_smooth_win_s": 0.5}`

## Top 10 table

| idx | merges | cov>=0.8 | union_recall | union_purity | fp_no_call_sec/hr | seg/hr | keep@0.5 | params |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 4 | 9 | 0.620 | 0.803 | 0.934 | 5.852 | 79.2 | 0.139 | `{"boundary_end_thr": 0.2, "boundary_nms_sep_s": 0.5, "boundary_pair_max_gap_s": 5.0, "boundary_score_thr": 0.7, "boundary_smooth_win_s": 0.1, "boundary_split_gap_s": 0.2, "boundary_split_margin_s": 0.0, "boundary_start_thr": 0.2, "max_in_call_min": 0.7, "max_segment_s": 900.0, "mean_in_call_min": 0.45, "split_lo": 0.3, "split_min_off_s": 0.25, "trim_s": 0.0, "viterbi_cost_mult": 0.5, "viterbi_min_off_s": 0.0, "viterbi_min_on_s": 0.5, "viterbi_smooth_win_s": 0.5}` |
| 3 | 9 | 0.620 | 0.803 | 0.934 | 5.852 | 79.2 | 0.139 | `{"boundary_end_thr": 0.2, "boundary_nms_sep_s": 0.35, "boundary_pair_max_gap_s": 5.0, "boundary_score_thr": 0.7, "boundary_smooth_win_s": 0.1, "boundary_split_gap_s": 0.2, "boundary_split_margin_s": 0.0, "boundary_start_thr": 0.2, "max_in_call_min": 0.7, "max_segment_s": 900.0, "mean_in_call_min": 0.45, "split_lo": 0.3, "split_min_off_s": 0.25, "trim_s": 0.0, "viterbi_cost_mult": 0.5, "viterbi_min_off_s": 0.0, "viterbi_min_on_s": 0.5, "viterbi_smooth_win_s": 0.5}` |
| 2 | 9 | 0.620 | 0.796 | 0.935 | 5.500 | 80.8 | 0.114 | `{"boundary_end_thr": 0.2, "boundary_nms_sep_s": 0.5, "boundary_pair_max_gap_s": 5.0, "boundary_score_thr": 0.7, "boundary_smooth_win_s": 0.0, "boundary_split_gap_s": 0.2, "boundary_split_margin_s": 0.0, "boundary_start_thr": 0.2, "max_in_call_min": 0.7, "max_segment_s": 900.0, "mean_in_call_min": 0.45, "split_lo": 0.3, "split_min_off_s": 0.25, "trim_s": 0.0, "viterbi_cost_mult": 0.5, "viterbi_min_off_s": 0.0, "viterbi_min_on_s": 0.5, "viterbi_smooth_win_s": 0.5}` |
| 1 | 9 | 0.608 | 0.794 | 0.935 | 5.255 | 80.6 | 0.127 | `{"boundary_end_thr": 0.2, "boundary_nms_sep_s": 0.35, "boundary_pair_max_gap_s": 5.0, "boundary_score_thr": 0.7, "boundary_smooth_win_s": 0.0, "boundary_split_gap_s": 0.2, "boundary_split_margin_s": 0.0, "boundary_start_thr": 0.2, "max_in_call_min": 0.7, "max_segment_s": 900.0, "mean_in_call_min": 0.45, "split_lo": 0.3, "split_min_off_s": 0.25, "trim_s": 0.0, "viterbi_cost_mult": 0.5, "viterbi_min_off_s": 0.0, "viterbi_min_on_s": 0.5, "viterbi_smooth_win_s": 0.5}` |
| 16 | 10 | 0.671 | 0.820 | 0.933 | 7.453 | 64.9 | 0.139 | `{"boundary_end_thr": 0.5, "boundary_nms_sep_s": 0.5, "boundary_pair_max_gap_s": 5.0, "boundary_score_thr": 0.7, "boundary_smooth_win_s": 0.1, "boundary_split_gap_s": 0.2, "boundary_split_margin_s": 0.0, "boundary_start_thr": 0.2, "max_in_call_min": 0.7, "max_segment_s": 900.0, "mean_in_call_min": 0.45, "split_lo": 0.3, "split_min_off_s": 0.25, "trim_s": 0.0, "viterbi_cost_mult": 0.5, "viterbi_min_off_s": 0.0, "viterbi_min_on_s": 0.5, "viterbi_smooth_win_s": 0.5}` |
| 15 | 10 | 0.671 | 0.820 | 0.933 | 7.453 | 64.8 | 0.139 | `{"boundary_end_thr": 0.5, "boundary_nms_sep_s": 0.35, "boundary_pair_max_gap_s": 5.0, "boundary_score_thr": 0.7, "boundary_smooth_win_s": 0.1, "boundary_split_gap_s": 0.2, "boundary_split_margin_s": 0.0, "boundary_start_thr": 0.2, "max_in_call_min": 0.7, "max_segment_s": 900.0, "mean_in_call_min": 0.45, "split_lo": 0.3, "split_min_off_s": 0.25, "trim_s": 0.0, "viterbi_cost_mult": 0.5, "viterbi_min_off_s": 0.0, "viterbi_min_on_s": 0.5, "viterbi_smooth_win_s": 0.5}` |
| 12 | 10 | 0.671 | 0.817 | 0.933 | 6.776 | 69.3 | 0.139 | `{"boundary_end_thr": 0.4, "boundary_nms_sep_s": 0.5, "boundary_pair_max_gap_s": 5.0, "boundary_score_thr": 0.7, "boundary_smooth_win_s": 0.1, "boundary_split_gap_s": 0.2, "boundary_split_margin_s": 0.0, "boundary_start_thr": 0.2, "max_in_call_min": 0.7, "max_segment_s": 900.0, "mean_in_call_min": 0.45, "split_lo": 0.3, "split_min_off_s": 0.25, "trim_s": 0.0, "viterbi_cost_mult": 0.5, "viterbi_min_off_s": 0.0, "viterbi_min_on_s": 0.5, "viterbi_smooth_win_s": 0.5}` |
| 11 | 10 | 0.671 | 0.817 | 0.933 | 6.776 | 69.3 | 0.139 | `{"boundary_end_thr": 0.4, "boundary_nms_sep_s": 0.35, "boundary_pair_max_gap_s": 5.0, "boundary_score_thr": 0.7, "boundary_smooth_win_s": 0.1, "boundary_split_gap_s": 0.2, "boundary_split_margin_s": 0.0, "boundary_start_thr": 0.2, "max_in_call_min": 0.7, "max_segment_s": 900.0, "mean_in_call_min": 0.45, "split_lo": 0.3, "split_min_off_s": 0.25, "trim_s": 0.0, "viterbi_cost_mult": 0.5, "viterbi_min_off_s": 0.0, "viterbi_min_on_s": 0.5, "viterbi_smooth_win_s": 0.5}` |
| 13 | 10 | 0.671 | 0.816 | 0.934 | 7.513 | 66.1 | 0.127 | `{"boundary_end_thr": 0.5, "boundary_nms_sep_s": 0.35, "boundary_pair_max_gap_s": 5.0, "boundary_score_thr": 0.7, "boundary_smooth_win_s": 0.0, "boundary_split_gap_s": 0.2, "boundary_split_margin_s": 0.0, "boundary_start_thr": 0.2, "max_in_call_min": 0.7, "max_segment_s": 900.0, "mean_in_call_min": 0.45, "split_lo": 0.3, "split_min_off_s": 0.25, "trim_s": 0.0, "viterbi_cost_mult": 0.5, "viterbi_min_off_s": 0.0, "viterbi_min_on_s": 0.5, "viterbi_smooth_win_s": 0.5}` |
| 8 | 10 | 0.671 | 0.812 | 0.934 | 6.776 | 72.9 | 0.127 | `{"boundary_end_thr": 0.3, "boundary_nms_sep_s": 0.5, "boundary_pair_max_gap_s": 5.0, "boundary_score_thr": 0.7, "boundary_smooth_win_s": 0.1, "boundary_split_gap_s": 0.2, "boundary_split_margin_s": 0.0, "boundary_start_thr": 0.2, "max_in_call_min": 0.7, "max_segment_s": 900.0, "mean_in_call_min": 0.45, "split_lo": 0.3, "split_min_off_s": 0.25, "trim_s": 0.0, "viterbi_cost_mult": 0.5, "viterbi_min_off_s": 0.0, "viterbi_min_on_s": 0.5, "viterbi_smooth_win_s": 0.5}` |
