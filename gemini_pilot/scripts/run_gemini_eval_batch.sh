#!/usr/bin/env bash
# Run gemini_pilot call-timestamp extraction on all 10 labeled videos.
#
# Usage:
#   bash gemini_pilot/scripts/run_gemini_eval_batch.sh
#
# Prerequisites:
#   - GEMINI_API_KEY set (or in gemini_pilot/.env)
#   - AWS CLI configured for s3 access
#
# The 10 videos with the most manually-labeled call boundaries (v1):
#   Qa-ppZFUp0g (27), D4uiHjHW4AU (19), DEt3IRqqUVs (12), Krnsw9WZtRA (8),
#   FkgGv2iMjEo (8), UWLJ81ezcpU (7), RL6Y5qig0Wg (7), gDdoZC9Nhgk (7),
#   ci-FdcWiJiA (7), CfMJ01KP_ns (6)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

BUCKET="s3://rezora-whisperx-us-east-1-864981718771/audio/pretraining"
OUT_DIR="artifacts/gemini_pilot/eval_batch_$(date +%Y%m%d_%H%M%S)"

python "${REPO_ROOT}/gemini_pilot/scripts/gemini_pilot_extract_call_timestamps.py" \
  --audio-s3-uri "${BUCKET}/Not_Sleeping_Until_I_Book_in_30_SMMA_Appointments_(LIVE_Cold_Calling) - Qa-ppZFUp0g.mp3" \
  --audio-s3-uri "${BUCKET}/How_To_Book_6_Meetings_A_Day_(Live_SMMA_Cold_Calling) - D4uiHjHW4AU.mp3" \
  --audio-s3-uri "${BUCKET}/10_Live_Cold_Calls_from_6_Different_Sales_Reps - DEt3IRqqUVs.mp3" \
  --audio-s3-uri "${BUCKET}/Selling_AI_Receptionists_To_Local_Businesses_With_Cold_Calls_(\$5k_Per) - Krnsw9WZtRA.mp3" \
  --audio-s3-uri "${BUCKET}/LIVE_SMMA_beginner's_first_time_cold_calling_(BRUTAL) - FkgGv2iMjEo.mp3" \
  --audio-s3-uri "${BUCKET}/Let’s_Call_Some_Leads__Real_Life_Sales_Calls - UWLJ81ezcpU.mp3" \
  --audio-s3-uri "${BUCKET}/Live_Dialing_Insurance_Leads!_[Final_Expense_Insurance] - RL6Y5qig0Wg.mp3" \
  --audio-s3-uri "${BUCKET}/Live_Cold_Calls_For_My_Ai_Marketing_Agency! - gDdoZC9Nhgk.mp3" \
  --audio-s3-uri "${BUCKET}/50_LIVE_COLD_CALLS__REAL_Objections__Best_Responses_To_Use - ci-FdcWiJiA.mp3" \
  --audio-s3-uri "${BUCKET}/Watch_me_book_5-7_meetings_a_day_(ai_agency_cold_calling) - CfMJ01KP_ns.mp3" \
  --out-dir "${OUT_DIR}"

echo ""
echo "Run complete → ${OUT_DIR}"
echo ""
echo "To evaluate:"
echo "  python ${REPO_ROOT}/gemini_pilot/scripts/eval_gemini_call_detector.py --run-dir ${OUT_DIR}"
