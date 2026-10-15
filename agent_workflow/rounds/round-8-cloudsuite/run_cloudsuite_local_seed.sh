#!/usr/bin/env bash
set -Eeuo pipefail

ROUND_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:---full}"
if [[ "${MODE}" != "--full" && "${MODE}" != "--pilot" && "${MODE}" != "--smoke" && "${MODE}" != "--calibrate" ]]; then
  printf 'Usage: %s [--smoke|--calibrate|--pilot|--full]\n' "$0" >&2
  exit 2
fi

SEEDED_IMAGE="cloudsuite/round8-db-seeded:ready"
PREFIX="bsr8-local-$(date -u +%Y%m%dT%H%M%SZ)-$$"
NETWORK="${PREFIX}-net"
RESULT_DIR="${ROUND_DIR}/results/${PREFIX}"
EXPECTED_TABLES_FILE="${ROUND_DIR}/expected-elgg-tables.txt"
CREATED=()
NETWORK_CREATED=0
mkdir -p "${RESULT_DIR}"
command -v timeout >/dev/null || { printf 'coreutils timeout is required\n' >&2; exit 2; }
command -v python3 >/dev/null || { printf 'python3 is required for Faban XML validation\n' >&2; exit 2; }

archive_container() {
  local name="$1"
  if ! docker inspect "${name}" >/dev/null 2>&1; then return 0; fi
  docker logs "${name}" > "${RESULT_DIR}/${name}.container.log" 2>&1 || true
  docker inspect "${name}" > "${RESULT_DIR}/${name}.inspect.json" 2>/dev/null || true
  docker diff "${name}" > "${RESULT_DIR}/${name}.diff.txt" 2>/dev/null || true
  docker rm -f "${name}" >/dev/null 2>&1 || true
}

cleanup() {
  local status=$?
  local name
  for name in "${CREATED[@]}"; do archive_container "${name}"; done
  if [[ "${NETWORK_CREATED}" == 1 ]]; then docker network rm "${NETWORK}" >/dev/null 2>&1 || true; fi
  exit "${status}"
}
trap cleanup EXIT INT TERM

if ! docker image inspect "${SEEDED_IMAGE}" >/dev/null 2>&1; then
  printf 'Missing local seeded image %s; run bootstrap_local_seed.sh first\n' "${SEEDED_IMAGE}" >&2
  exit 2
fi

docker network create "${NETWORK}" > "${RESULT_DIR}/network-id.txt"
NETWORK_CREATED=1
printf 'mode=%s\nstarted_utc=%s\n' "${MODE}" "$(date -u +%FT%TZ)" > "${RESULT_DIR}/run-metadata.txt"
docker version --format '{{.Server.Version}}' >> "${RESULT_DIR}/run-metadata.txt"
sha256sum "${ROUND_DIR}/run_cloudsuite_local_seed.sh" \
  "${ROUND_DIR}/validate_faban_summary.py" \
  "${ROUND_DIR}/analysis_contract.md" >> "${RESULT_DIR}/run-metadata.txt"
docker image inspect "${SEEDED_IMAGE}" cloudsuite/web-serving:web_server \
  cloudsuite/web-serving:memcached_server cloudsuite/web-serving:faban_client \
  --format '{{index .RepoTags 0}} {{.Id}} {{json .RepoDigests}}' > "${RESULT_DIR}/image-digests.txt"
: > "${RESULT_DIR}/runs.jsonl"
: > "${RESULT_DIR}/phase-times.jsonl"
: > "${RESULT_DIR}/cell-times.jsonl"
: > "${RESULT_DIR}/randomized-order.txt"
: > "${RESULT_DIR}/status.txt"

start_backends() {
  local config="$1" session="$2" phase="$3" ready=0 table_count
  DB="${PREFIX}-mysql-${config}-session${session}-${phase}"
  MEM="${PREFIX}-mem-${config}-session${session}-${phase}"
  docker run -d --name "${DB}" --network "${NETWORK}" --network-alias mysql_server \
    --entrypoint /usr/sbin/mysqld "${SEEDED_IMAGE}" \
    > "${RESULT_DIR}/${DB}.container-id.txt"
  CREATED+=("${DB}")
  for _ in $(seq 1 120); do
    if docker exec "${DB}" mysql -uroot -proot -Nse 'SELECT 1' >/dev/null 2>&1; then ready=1; break; fi
    if [[ "$(docker inspect -f '{{.State.Running}}' "${DB}" 2>/dev/null || true)" != true ]]; then break; fi
    sleep 2
  done
  if [[ "${ready}" != 1 ]]; then
    printf 'database_not_ready config=%s session=%s phase=%s\n' \
      "${config}" "${session}" "${phase}" >> "${RESULT_DIR}/status.txt"
    return 1
  fi
  table_count="$(docker exec "${DB}" mysql -uroot -proot -Nse \
    'SELECT COUNT(*) FROM information_schema.tables WHERE table_schema="ELGG_DB";')"
  local actual_tables="${RESULT_DIR}/${DB}.tables.txt"
  docker exec "${DB}" mysql -uroot -proot -Nse \
    'SELECT table_name FROM information_schema.tables WHERE table_schema="ELGG_DB" ORDER BY table_name;' \
    > "${actual_tables}"
  local expected_tables_sorted="${RESULT_DIR}/${DB}.expected-tables.sorted.txt"
  local actual_tables_sorted="${RESULT_DIR}/${DB}.actual-tables.sorted.txt"
  sort "${EXPECTED_TABLES_FILE}" > "${expected_tables_sorted}"
  sort "${actual_tables}" > "${actual_tables_sorted}"
  local entity_count metadata_count
  entity_count="$(docker exec "${DB}" mysql -uroot -proot -Nse \
    'SELECT COUNT(*) FROM ELGG_DB.elgg_entities;')"
  metadata_count="$(docker exec "${DB}" mysql -uroot -proot -Nse \
    'SELECT COUNT(*) FROM ELGG_DB.elgg_metadata;')"
  printf 'database=ELGG_DB\ntable_count=%s\nelgg_entities_rows=%s\nelgg_metadata_rows=%s\n' \
    "${table_count}" "${entity_count}" "${metadata_count}" \
    > "${RESULT_DIR}/${DB}.validation.txt"
  if [[ ! "${table_count}" =~ ^[0-9]+$ || "${table_count}" -ne 18 ]] || \
     ! diff -u "${expected_tables_sorted}" "${actual_tables_sorted}" \
       > "${RESULT_DIR}/${DB}.schema.diff" || \
     [[ ! "${entity_count}" =~ ^[0-9]+$ || "${entity_count}" -lt 1 ||
        ! "${metadata_count}" =~ ^[0-9]+$ || "${metadata_count}" -lt 1 ]]; then
    printf 'database_seed_invalid config=%s session=%s phase=%s table_count=%s entities=%s metadata=%s\n' \
      "${config}" "${session}" "${phase}" "${table_count}" "${entity_count}" "${metadata_count}" \
      >> "${RESULT_DIR}/status.txt"
    return 1
  fi

  docker run -d --name "${MEM}" --network "${NETWORK}" --network-alias memcache_server \
    cloudsuite/web-serving:memcached_server -m 512 -t 2 \
    > "${RESULT_DIR}/${MEM}.container-id.txt"
  CREATED+=("${MEM}")
}

stop_backends() {
  archive_container "${DB}"
  archive_container "${MEM}"
}

start_web() {
  local config="$1" children="$2" session="$3" phase="$4" ready=0
  WEB="${PREFIX}-web-${config}-session${session}-${phase}"
  docker run -d --name "${WEB}" --network "${NETWORK}" \
    --network-alias root-server --network-alias web_server \
    cloudsuite/web-serving:web_server \
    bash -lc "set -Eeuo pipefail; /etc/bootstrap.sh http root-server mysql_server memcache_server ${children} auto; exec tail -f /dev/null" \
    > "${RESULT_DIR}/${WEB}.container-id.txt"
  CREATED+=("${WEB}")
  for _ in $(seq 1 90); do
    if docker exec "${WEB}" wget -q -O /dev/null http://127.0.0.1:8080/; then ready=1; break; fi
    if [[ "$(docker inspect -f '{{.State.Running}}' "${WEB}" 2>/dev/null || true)" != true ]]; then break; fi
    sleep 2
  done
  docker logs "${WEB}" > "${RESULT_DIR}/${WEB}.bootstrap.log" 2>&1 || true
  if [[ "${ready}" != 1 ]]; then
    printf 'web_not_ready config=%s session=%s phase=%s\n' \
      "${config}" "${session}" "${phase}" >> "${RESULT_DIR}/status.txt"
    return 1
  fi
}

run_cell() {
  local config="$1" children="$2" scale="$3" session="$4" phase="$5" steady="$6"
  local cell="${config}-session${session}-${phase}-scale${scale}"
  local client="${PREFIX}-client-${cell}"
  local start_ns end_ns elapsed exit_code output_dir copy_code summary_count summary_file validation_code log_protocol_error
  start_ns="$(date +%s%N)"
  CREATED+=("${client}")
  set +e
  timeout --signal=TERM --kill-after=10s 300s docker run --name "${client}" --network "${NETWORK}" cloudsuite/web-serving:faban_client \
    --oper=usergen\&run --ramp-up=5 --ramp-down=5 --steady="${steady}" \
    --min=1000 --max=1500 --type=THINKTIME --dist=fixed root-server "${scale}" \
    > "${RESULT_DIR}/${cell}.stdout.log" 2>&1
  exit_code=$?
  set -e
  end_ns="$(date +%s%N)"
  elapsed="$(awk -v a="${start_ns}" -v b="${end_ns}" 'BEGIN { printf "%.6f", (b-a)/1000000000 }')"
  output_dir="${RESULT_DIR}/${cell}-faban-output"
  set +e
  docker cp "${client}:/usr/src/faban/output" "${output_dir}" >/dev/null 2>&1
  copy_code=$?
  set -e
  summary_file=""
  summary_count=0
  if [[ "${copy_code}" == 0 ]]; then
    find "${output_dir}" -type f -printf '%p %s bytes\n' \
      > "${RESULT_DIR}/${cell}-faban-files.txt" 2>/dev/null || true
    while IFS= read -r -d '' candidate; do
      summary_file="${candidate}"
      summary_count=$((summary_count + 1))
    done < <(find "${output_dir}" -type f -name summary.xml -print0)
  else
    printf 'faban_output_unavailable\n' > "${RESULT_DIR}/${cell}-validation.txt"
  fi
  validation_code=1
  if [[ "${copy_code}" == 0 && "${summary_count}" -eq 1 ]]; then
    set +e
    python3 "${ROUND_DIR}/validate_faban_summary.py" "${summary_file}" \
      "${RESULT_DIR}/${cell}-validation.json" \
      > "${RESULT_DIR}/${cell}-validation.stdout.txt" \
      2> "${RESULT_DIR}/${cell}-validation.stderr.txt"
    validation_code=$?
    set -e
  else
    printf 'expected_exactly_one_summary_xml found=%s\n' "${summary_count}" \
      > "${RESULT_DIR}/${cell}-validation.stderr.txt"
    printf '{"valid":false,"summary_xml_count":%s}\n' "${summary_count}" \
      > "${RESULT_DIR}/${cell}-validation.json"
  fi
  log_protocol_error=0
  if rg -q 'does not contain a hostname|FileNotFoundException: http://[^[:space:]]+/http://|Exception in thread' \
      "${RESULT_DIR}/${cell}.stdout.log"; then
    log_protocol_error=1
    printf 'client_protocol_error_in_stdout\n' >> "${RESULT_DIR}/${cell}-validation.stderr.txt"
  fi
  docker logs "${client}" > "${RESULT_DIR}/${cell}.container.log" 2>&1 || true
  docker inspect "${client}" > "${RESULT_DIR}/${cell}.inspect.json" 2>/dev/null || true
  printf '{"cell":"%s","config":"%s","php_fpm_max_children":%s,"scale":%s,"session":%s,"phase":"%s","planned_steady_seconds":%s,"start_epoch_ns":%s,"end_epoch_ns":%s,"elapsed_seconds":%s,"exit_code":%s,"faban_output_copied":%s,"summary_xml":"%s","summary_xml_count":%s,"summary_validation_exit_code":%s,"client_log_protocol_error":%s,"summary_validation_json":"%s"}\n' \
    "${cell}" "${config}" "${children}" "${scale}" "${session}" "${phase}" "${steady}" \
    "${start_ns}" "${end_ns}" "${elapsed}" "${exit_code}" \
    "$([[ "${copy_code}" == 0 ]] && printf true || printf false)" \
    "${summary_file}" "${summary_count}" "${validation_code}" "${log_protocol_error}" \
    "${RESULT_DIR}/${cell}-validation.json" \
    >> "${RESULT_DIR}/runs.jsonl"
  if [[ "${exit_code}" != 0 || "${copy_code}" != 0 || "${validation_code}" != 0 || "${log_protocol_error}" != 0 ]]; then
    printf 'workload_failed_cell=%s_exit=%s_output_copy=%s_validation=%s_log_protocol=%s_summary_count=%s\n' \
      "${cell}" "${exit_code}" "${copy_code}" "${validation_code}" "${log_protocol_error}" "${summary_count}" \
      >> "${RESULT_DIR}/status.txt"
    return 1
  fi
}

run_phase() {
  local config="$1" children="$2" session="$3" phase="$4" steady="$5"
  shift 5
  local load_order=("$@") phase_failed=0
  local setup_start_ns setup_end_ns phase_end_ns cell_setup_start_ns cell_setup_end_ns cell_end_ns
  local total_setup_seconds=0 cell_setup_seconds
  setup_start_ns="$(date +%s%N)"
  printf 'session=%s config=%s php_fpm_max_children=%s phase=%s steady_seconds=%s load_order=%s\n' \
    "${session}" "${config}" "${children}" "${phase}" "${steady}" \
    "$(IFS=,; printf '%s' "${load_order[*]}")" >> "${RESULT_DIR}/randomized-order.txt"
  for scale in "${load_order[@]}"; do
    cell_setup_start_ns="$(date +%s%N)"
    if ! start_backends "${config}" "${session}" "${phase}-scale${scale}"; then
      stop_backends
      phase_failed=1
      break
    fi
    if ! start_web "${config}" "${children}" "${session}" "${phase}-scale${scale}"; then
      archive_container "${WEB}"
      stop_backends
      phase_failed=1
      break
    fi
    cell_setup_end_ns="$(date +%s%N)"
    if [[ -z "${setup_end_ns:-}" ]]; then setup_end_ns="${cell_setup_end_ns}"; fi
    cell_setup_seconds="$(awk -v a="${cell_setup_start_ns}" -v b="${cell_setup_end_ns}" \
      'BEGIN {printf "%.6f", (b-a)/1000000000}')"
    total_setup_seconds="$(awk -v a="${total_setup_seconds}" -v b="${cell_setup_seconds}" \
      'BEGIN {printf "%.6f", a+b}')"
    if ! run_cell "${config}" "${children}" "${scale}" "${session}" "${phase}" "${steady}"; then
      phase_failed=1
    fi
    archive_container "${WEB}"
    stop_backends
    cell_end_ns="$(date +%s%N)"
    printf '{"config":"%s","session":%s,"phase":"%s","scale":%s,"setup_start_epoch_ns":%s,"setup_end_epoch_ns":%s,"cell_end_epoch_ns":%s,"setup_seconds":%s,"cell_total_seconds":%.6f,"failed":%s}\n' \
      "${config}" "${session}" "${phase}" "${scale}" \
      "${cell_setup_start_ns}" "${cell_setup_end_ns}" "${cell_end_ns}" \
      "${cell_setup_seconds}" \
      "$(awk -v a="${cell_setup_start_ns}" -v b="${cell_end_ns}" 'BEGIN {print (b-a)/1000000000}')" \
      "$([[ "${phase_failed}" == 0 ]] && printf false || printf true)" \
      >> "${RESULT_DIR}/cell-times.jsonl"
    if [[ "${phase_failed}" == 1 ]]; then break; fi
  done
  phase_end_ns="$(date +%s%N)"
  if [[ -z "${setup_end_ns:-}" ]]; then setup_end_ns="${phase_end_ns}"; fi
  printf '{"config":"%s","session":%s,"phase":"%s","setup_start_epoch_ns":%s,"setup_end_epoch_ns":%s,"phase_end_epoch_ns":%s,"setup_seconds":%.6f,"phase_total_seconds":%.6f,"failed":%s}\n' \
    "${config}" "${session}" "${phase}" "${setup_start_ns}" "${setup_end_ns}" "${phase_end_ns}" \
    "${total_setup_seconds}" \
    "$(awk -v a="${setup_start_ns}" -v b="${phase_end_ns}" 'BEGIN {print (b-a)/1000000000}')" \
    "$([[ "${phase_failed}" == 0 ]] && printf false || printf true)" \
    >> "${RESULT_DIR}/phase-times.jsonl"
  return "${phase_failed}"
}

if [[ "${MODE}" == "--smoke" ]]; then
  sessions=(1)
  config_specs=(fpm4:4)
  loads=(20 40)
elif [[ "${MODE}" == "--calibrate" ]]; then
  sessions=(1)
  config_specs=(fpm4:4)
  loads=(20)
elif [[ "${MODE}" == "--pilot" ]]; then
  sessions=(1)
  config_specs=(fpm4:4 fpm8:8)
  loads=(20 40 60 80 100)
else
  sessions=(1 2 3 4)
  config_specs=(fpm4:4 fpm8:8)
  loads=(20 40 60 80 100)
fi

for session in "${sessions[@]}"; do
  if [[ "${MODE}" == "--smoke" || "${MODE}" == "--calibrate" ]]; then
    block_order=("${config_specs[@]}")
  else
    mapfile -t block_order < <(printf '%s\n' "${config_specs[@]}" | shuf)
  fi
  for spec in "${block_order[@]}"; do
    config="${spec%%:*}"
    children="${spec#*:}"
    if [[ "${MODE}" == "--smoke" ]]; then
      probe_order=("${loads[@]}")
      full_order=("${loads[@]}")
    else
      mapfile -t probe_order < <(shuf -e "${loads[@]}")
      mapfile -t full_order < <(shuf -e "${loads[@]}")
    fi
    if [[ "${MODE}" == "--calibrate" ]]; then
      calibration_failed=0
      if ! run_phase "${config}" "${children}" "${session}" calibrate60 60 "${loads[0]}"; then
        calibration_failed=1
      fi
      if ! run_phase "${config}" "${children}" "${session}" calibrate120 120 "${loads[0]}"; then
        calibration_failed=1
      fi
      if [[ "${calibration_failed}" == 1 ]]; then
        printf 'calibration_invalid config=%s session=%s\n' "${config}" "${session}" \
          >> "${RESULT_DIR}/status.txt"
        exit 1
      fi
    elif [[ "${MODE}" == "--smoke" ]]; then
      run_phase "${config}" "${children}" "${session}" smoke 30 "${probe_order[@]}"
    else
      run_phase "${config}" "${children}" "${session}" probe 60 "${probe_order[@]}"
      run_phase "${config}" "${children}" "${session}" full 120 "${full_order[@]}"
    fi
  done
done

printf 'completed_utc=%s\n' "$(date -u +%FT%TZ)" >> "${RESULT_DIR}/run-metadata.txt"
printf 'complete\n' >> "${RESULT_DIR}/status.txt"
printf 'Completed %s mode. Evidence: %s\n' "${MODE}" "${RESULT_DIR}"
