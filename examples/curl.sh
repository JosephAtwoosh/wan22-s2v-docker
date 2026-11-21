# Renders one performance and waits for it. Requires curl and jq.
#
#   ./examples/curl.sh localhost:8000 start.jpg audio.wav prompt.txt
#
set -euo pipefail

BASE="${BASE:-http://localhost:8000}"
KEY="${WAN_API_KEY:-}"
START="${1:?start image}"
AUDIO="${2:?audio}"
PROMPT_FILE="${3:?prompt.txt}"

AUTH=()
[ -n "$KEY" ] && AUTH=(-H "Authorization: Bearer $KEY")

JOB=$(curl -sS -X POST "$BASE/v1/performances" ${AUTH[@]+"${AUTH[@]}"} \
  -F "start=@${START}" \
  -F "audio=@${AUDIO}" \
  -F "prompt=$(cat "${PROMPT_FILE}")" \
  -F "max_clips=${MAX_CLIPS:-1}" | jq -r .id)

echo "job ${JOB}"

for _ in $(seq 1 720); do
  STATE=$(curl -sS "$BASE/v1/jobs/${JOB}" ${AUTH[@]+"${AUTH[@]}"})
  STATUS=$(jq -r .status <<<"$STATE")
  case "$STATUS" in
    done)
      jq -r '.plan | "\(.out_width)x\(.out_height) \(.clips) clip(s) \(.video_seconds)s"' <<<"$STATE"
      curl -sS "$BASE/v1/jobs/${JOB}/video" ${AUTH[@]+"${AUTH[@]}"} -o "performance-${JOB}.mp4"
      curl -sS "$BASE/v1/jobs/${JOB}/manifest.json" ${AUTH[@]+"${AUTH[@]}"} -o "performance-${JOB}.json"
      echo "wrote performance-${JOB}.mp4"
      exit 0
      ;;
    error)
      jq -r .error <<<"$STATE" >&2
      exit 1
      ;;
  esac
  sleep 5
done

echo "timed out waiting for ${JOB}" >&2
exit 1
