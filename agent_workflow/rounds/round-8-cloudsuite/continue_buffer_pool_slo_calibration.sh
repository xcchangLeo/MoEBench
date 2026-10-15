#!/usr/bin/env bash
set -Eeuo pipefail
ROUND_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULT_DIR="${ROUND_DIR}/results/bsr8bpcal-20261014T003808Z-453545"
RUN_ID="bsr8bpcal-20261014T003808Z-453545"
NETWORK="${RUN_ID}-net"
ORDER=("2g/fpm8" "128m/fpm4" "128m/fpm8" "2g/fpm4" "128m/fpm4" "128m/fpm8" "2g/fpm8" "2g/fpm4" "128m/fpm4" "2g/fpm8" "2g/fpm4" "128m/fpm8")
SEEDED_IMAGE="cloudsuite/round8-db-seeded:ready"
WEB_IMAGE="cloudsuite/web-serving:web_server"
MEM_IMAGE="cloudsuite/web-serving:memcached_server"
CLIENT_IMAGE="cloudsuite/web-serving:faban_client"
CREATED=()
NETWORK_CREATED=0

archive_container() {
  local name="$1"
  if ! docker inspect "${name}" >/dev/null 2>&1; then return 0; fi
  docker logs "${name}" > "${RESULT_DIR}/${name}.container.log" 2>&1 || true
  docker inspect "${name}" > "${RESULT_DIR}/${name}.inspect.json" 2>/dev/null || true
  docker diff "${name}" > "${RESULT_DIR}/${name}.diff.txt" 2>/dev/null || true
  docker rm -f "${name}" >/dev/null 2>&1 || true
}
cleanup() {
  local status=$? name
  for name in "${CREATED[@]}"; do archive_container "${name}"; done
  if [[ "${NETWORK_CREATED}" == 1 ]]; then docker network rm "${NETWORK}" >/dev/null 2>&1 || true; fi
  exit "${status}"
}
trap cleanup EXIT INT TERM

[[ -d "${RESULT_DIR}" && -f "${RESULT_DIR}/cells.csv" ]] || { printf 'existing calibration result directory not found\n' >&2; exit 2; }
docker network inspect "${NETWORK}" >/dev/null 2>&1 || { printf 'calibration network missing; refusing another environment\n' >&2; exit 2; }
NETWORK_CREATED=1
sha256sum "${BASH_SOURCE[0]}" > "${RESULT_DIR}/continuation-source-sha256.txt"
printf 'continuation_started_utc=%s\nstart_order_index=7\n' "$(date -u +%FT%TZ)" > "${RESULT_DIR}/continuation-metadata.txt"

run_one() {
  local index="$1" condition="$2" repeat=$(( (index - 1) / 4 + 1 ))
  local pool config children expected_pool db mem web client ready=0 cell
  case "${condition}" in
    128m/fpm4) pool=134217728; config=fpm4; children=4 ;;
    128m/fpm8) pool=134217728; config=fpm8; children=8 ;;
    2g/fpm4) pool=2147483648; config=fpm4; children=4 ;;
    2g/fpm8) pool=2147483648; config=fpm8; children=8 ;;
    *) return 2 ;;
  esac
  cell="repeat${repeat}-$(printf '%02d' "${index}")-${condition//\//-}-scale20"
  db="${RUN_ID}-db-${cell}"; mem="${RUN_ID}-mem-${cell}"
  web="${RUN_ID}-web-${cell}"; client="${RUN_ID}-client-${cell}"
  local start_ns end_ns elapsed exit_code copy_code validation_code protocol_error=0
  local output_dir="${RESULT_DIR}/${cell}-faban-output" summary_file="" summary_count=0
  local validation_file="${RESULT_DIR}/${cell}-validation.json" started_utc
  if docker inspect "${client}" >/dev/null 2>&1; then printf 'cell exists; refuse duplicate %s\n' "${cell}" >&2; return 2; fi
  started_utc="$(date -u +%FT%TZ)"
  printf 'starting cell=%s condition=%s at=%s\n' "${index}" "${condition}" "${started_utc}" | tee -a "${RESULT_DIR}/status.txt"
  docker run -d --name "${db}" --network "${NETWORK}" --network-alias mysql_server --entrypoint /usr/sbin/mysqld "${SEEDED_IMAGE}" --innodb-buffer-pool-size="${pool}" > "${RESULT_DIR}/${db}.container-id.txt"
  CREATED+=("${db}")
  for _ in $(seq 1 120); do
    if docker exec "${db}" mysql -uroot -proot -Nse 'SELECT 1' >/dev/null 2>&1; then ready=1; break; fi
    if [[ "$(docker inspect -f '{{.State.Running}}' "${db}" 2>/dev/null || true)" != true ]]; then break; fi
    sleep 2
  done
  if [[ "${ready}" != 1 ]]; then printf 'database_not_ready\n' > "${RESULT_DIR}/${cell}.status.txt"; return 1; fi
  expected_pool="$(docker exec "${db}" mysql -uroot -proot -Nse 'SELECT @@innodb_buffer_pool_size;')"
  docker exec "${db}" mysql -uroot -proot -e 'SELECT @@version AS mariadb_version, @@innodb_buffer_pool_size AS innodb_buffer_pool_size; SHOW CREATE TABLE ELGG_DB.elgg_entities; SHOW CREATE TABLE ELGG_DB.elgg_metadata;' > "${RESULT_DIR}/${cell}.db-config-and-schema.txt"
  docker exec "${db}" mysql -uroot -proot -Nse 'SELECT COUNT(*) FROM information_schema.tables WHERE table_schema="ELGG_DB";' > "${RESULT_DIR}/${cell}.table-count.txt"
  if [[ "${expected_pool}" != "${pool}" || "$(<"${RESULT_DIR}/${cell}.table-count.txt")" != 18 ]]; then printf 'database_condition_or_seed_invalid\n' > "${RESULT_DIR}/${cell}.status.txt"; return 1; fi
  docker run -d --name "${mem}" --network "${NETWORK}" --network-alias memcache_server "${MEM_IMAGE}" -m 512 -t 2 > "${RESULT_DIR}/${mem}.container-id.txt"
  CREATED+=("${mem}")
  docker run -d --name "${web}" --network "${NETWORK}" --network-alias root-server --network-alias web_server "${WEB_IMAGE}" bash -lc "set -Eeuo pipefail; /etc/bootstrap.sh http root-server mysql_server memcache_server ${children} auto; exec tail -f /dev/null" > "${RESULT_DIR}/${web}.container-id.txt"
  CREATED+=("${web}")
  ready=0
  for _ in $(seq 1 90); do
    if docker exec "${web}" wget -q -O /dev/null http://127.0.0.1:8080/; then ready=1; break; fi
    if [[ "$(docker inspect -f '{{.State.Running}}' "${web}" 2>/dev/null || true)" != true ]]; then break; fi
    sleep 2
  done
  docker logs "${web}" > "${RESULT_DIR}/${web}.bootstrap.log" 2>&1 || true
  if [[ "${ready}" != 1 ]]; then printf 'web_not_ready\n' > "${RESULT_DIR}/${cell}.status.txt"; return 1; fi
  start_ns="$(date +%s%N)"
  set +e
  timeout --signal=TERM --kill-after=10s 300s docker run --name "${client}" --network "${NETWORK}" "${CLIENT_IMAGE}" --oper=usergen\&run --ramp-up=5 --ramp-down=5 --steady=120 --min=1000 --max=1500 --type=THINKTIME --dist=fixed root-server 20 > "${RESULT_DIR}/${cell}.stdout.log" 2>&1
  exit_code=$?
  set -e
  end_ns="$(date +%s%N)"
  elapsed="$(awk -v a="${start_ns}" -v b="${end_ns}" 'BEGIN {printf "%.6f", (b-a)/1000000000}')"
  set +e; docker cp "${client}:/usr/src/faban/output" "${output_dir}" >/dev/null 2>&1; copy_code=$?; set -e
  if [[ "${copy_code}" == 0 ]]; then while IFS= read -r -d '' candidate; do summary_file="${candidate}"; summary_count=$((summary_count + 1)); done < <(find "${output_dir}" -type f -name summary.xml -print0); fi
  validation_code=1
  if [[ "${copy_code}" == 0 && "${summary_count}" == 1 ]]; then
    set +e; python3 "${ROUND_DIR}/validate_faban_summary.py" "${summary_file}" "${validation_file}" > "${RESULT_DIR}/${cell}.validation.stdout.txt" 2> "${RESULT_DIR}/${cell}.validation.stderr.txt"; validation_code=$?; set -e
  else
    printf '{"valid":false,"reason":"expected_exactly_one_summary_xml","summary_xml_count":%s}\n' "${summary_count}" > "${validation_file}"
  fi
  if rg -q 'does not contain a hostname|FileNotFoundException: http://[^[:space:]]+/http://|Exception in thread' "${RESULT_DIR}/${cell}.stdout.log"; then protocol_error=1; fi
  local summary_valid=false mix_valid=false failures=unknown slo=false valid=false
  summary_valid="$(jq -r '.summary_valid // false' "${validation_file}")"
  mix_valid="$(jq -r '.mix_valid // false' "${validation_file}")"
  failures="$(jq -r '.operation_failures // "unknown"' "${validation_file}")"
  slo="$(jq -r '([.response_times[].slo_passed] | length > 0) and ([.response_times[].slo_passed] | all)' "${validation_file}")"
  if [[ "${exit_code}" == 0 && "${copy_code}" == 0 && "${validation_code}" == 0 && "${protocol_error}" == 0 && "${summary_valid}" == true && "${mix_valid}" == true && "${failures}" == 0 && "${expected_pool}" == "${pool}" ]]; then valid=true; fi
  printf '%s,%s,%s,%s,20,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' "${cell}" "${repeat}" "${condition}" "${config}" "${started_utc}" "$(date -u +%FT%TZ)" "${elapsed}" "${exit_code}" "${expected_pool}" "${summary_count}" "${summary_valid}" "${mix_valid}" "${failures}" "${slo}" "${valid}" >> "${RESULT_DIR}/cells.csv"
  printf '{"cell":"%s","repeat":%s,"condition":"%s","config":"%s","buffer_pool_bytes":%s,"scale":20,"started_utc":"%s","ended_utc":"%s","elapsed_seconds":%s,"client_exit_code":%s,"client_protocol_error":%s,"summary_xml_count":%s,"summary_validation_json":"%s","valid":%s,"latency_slo_passed":%s}\n' "${cell}" "${repeat}" "${condition}" "${config}" "${expected_pool}" "${started_utc}" "$(date -u +%FT%TZ)" "${elapsed}" "${exit_code}" "${protocol_error}" "${summary_count}" "${validation_file}" "${valid}" "${slo}" >> "${RESULT_DIR}/runs.jsonl"
  archive_container "${client}"; archive_container "${web}"; archive_container "${mem}"; archive_container "${db}"
}

for index in {7..12}; do condition="${ORDER[index-1]}"; run_one "${index}" "${condition}" || printf 'runner_error_cell=%s\n' "${index}" >> "${RESULT_DIR}/status.txt"; done
python3 - "${RESULT_DIR}" <<'PY'
import csv, json, pathlib, sys
root=pathlib.Path(sys.argv[1]); rows=list(csv.DictReader((root/'cells.csv').open()))
eligible={(r['condition'],r['repeat']):r['valid']=='true' and r['latency_slo_passed']=='true' for r in rows}
conditions=('128m/fpm4','128m/fpm8','2g/fpm4','2g/fpm8')
passing=[c for c in conditions if all(eligible.get((c,str(i)),False) for i in range(1,4))]
(root/'continuation-decision.json').write_text(json.dumps({'cells_recorded':len(rows),'conditions_passing_all_three_repeats':passing,'continue':bool(passing),'continuation_rule':'one fixed condition passes all native SLOs in all three valid repeats'},indent=2)+'\n')
with (root/'continuation-metadata.txt').open('a') as f:f.write('continuation_completed_utc='+__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()+'\n')
with (root/'status.txt').open('a') as f:f.write('complete\n')
PY
printf 'Continuation completed. Evidence: %s\n' "${RESULT_DIR}"
