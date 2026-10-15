#!/usr/bin/env bash
set -Eeuo pipefail

ROUND_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPECTED_TABLES_FILE="${ROUND_DIR}/expected-elgg-tables.txt"
SEEDED_IMAGE="cloudsuite/round8-db-seeded:ready"
WEB_IMAGE="cloudsuite/web-serving:web_server"
MEM_IMAGE="cloudsuite/web-serving:memcached_server"
CLIENT_IMAGE="cloudsuite/web-serving:faban_client"
PREDICTOR="${ROUND_DIR}/change_predictor.py"
VALIDATOR="${ROUND_DIR}/validate_faban_summary.py"
SCORER="${ROUND_DIR}/score_fpm8_target_sessions.py"
LOADS=(20 40 60 80 100)
PROBE_SECONDS=60
REFERENCE_SECONDS=120

usage() {
  printf 'Usage: %s --manifest PATH --expected-sha256 HEX [--dry-run]\n' "$0" >&2
}

MANIFEST=""
EXPECTED_SHA256=""
DRY_RUN=0
while (($#)); do
  case "$1" in
    --manifest) (($# >= 2)) || { usage; exit 2; }; MANIFEST="$2"; shift 2 ;;
    --expected-sha256) (($# >= 2)) || { usage; exit 2; }; EXPECTED_SHA256="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage; exit 2 ;;
  esac
done
[[ -n "${MANIFEST}" && -n "${EXPECTED_SHA256}" ]] || { usage; exit 2; }
[[ "${EXPECTED_SHA256}" =~ ^[[:xdigit:]]{64}$ ]] || { printf 'Expected SHA-256 must contain 64 hex characters\n' >&2; exit 2; }
[[ -f "${MANIFEST}" && ! -L "${MANIFEST}" ]] || { printf 'Manifest must be an existing regular, non-symlink file\n' >&2; exit 2; }
MANIFEST="$(realpath -e -- "${MANIFEST}")"
python3 - "${MANIFEST}" <<'PY'
import os,stat,sys
mode=stat.S_IMODE(os.stat(sys.argv[1]).st_mode)
if mode & 0o222:
    raise SystemExit("frozen manifest must have no write permission bits")
PY
[[ "$(sha256sum "${MANIFEST}" | awk '{print $1}')" == "${EXPECTED_SHA256,,}" ]] || {
  printf 'Frozen manifest SHA-256 mismatch\n' >&2; exit 2;
}
command -v python3 >/dev/null || { printf 'python3 is required\n' >&2; exit 2; }
command -v sha256sum >/dev/null || { printf 'sha256sum is required\n' >&2; exit 2; }

# Validate the frozen protocol and all source evidence before any Docker action.
python3 "${PREDICTOR}" check --manifest "${MANIFEST}" --expected-sha256 "${EXPECTED_SHA256,,}" > /dev/null
python3 - "${MANIFEST}" "${ROUND_DIR}" <<'PY'
import hashlib
import json
import math
import sys
from pathlib import Path

manifest_path, round_dir = Path(sys.argv[1]), Path(sys.argv[2])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
required = {
    "protocol", "source_sessions", "target_sessions", "source_full_cells",
    "source_probe_cells", "target_config", "probe_seconds", "reference_seconds",
    "load_levels", "selected_loads", "predictor_method", "predictor_source_sha256",
    "analyzer_source_sha256", "randomization_seed", "numpy_version", "model_id",
    "model_version", "preprocessing_version", "hyperparameters",
}
missing = required - manifest.keys()
if missing:
    raise SystemExit(f"manifest missing required fields: {sorted(missing)}")
if manifest["target_config"] != "fpm8" or manifest["probe_seconds"] != 60 or manifest["reference_seconds"] != 120:
    raise SystemExit("manifest does not describe the frozen fpm8 protocol")
if manifest["load_levels"] != [20, 40, 60, 80, 100]:
    raise SystemExit("manifest load levels differ from the allowed five-load grid")
selected = manifest["selected_loads"]
if len(selected) != 3 or len(set(selected)) != 3 or any(x not in manifest["load_levels"] for x in selected):
    raise SystemExit("manifest must select exactly three distinct allowed probe scales")
if manifest["source_sessions"] != [1, 2, 3, 4] or manifest["target_sessions"] != [101, 102, 103, 104]:
    raise SystemExit("manifest source/target session IDs differ from the required frozen split")
if manifest["predictor_method"] not in {"correlation", "history_scale", "linear_interpolation"}:
    raise SystemExit("manifest names an unsupported predictor")
if not all(isinstance(manifest[key], str) and manifest[key] for key in ("model_id", "model_version", "preprocessing_version")):
    raise SystemExit("manifest model and preprocessing identifiers must be non-empty strings")
if not isinstance(manifest["hyperparameters"], dict):
    raise SystemExit("manifest hyperparameters must be a JSON object")
if not isinstance(manifest["randomization_seed"], int):
    raise SystemExit("manifest randomization_seed must be an integer")

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

for field in ("source_full_cells", "source_probe_cells"):
    rows = manifest[field]
    expected = {(s, l) for s in range(1, 5) for l in manifest["load_levels"]}
    keys = {(int(row["session"]), int(row["scale"])) for row in rows}
    if len(rows) != 20 or keys != expected:
        raise SystemExit(f"{field} must contain exactly 20 cells across four source sessions")
    for row in rows:
        if any("fpm8" in row[k].lower() for k in ("validation_path", "summary_path")):
            raise SystemExit("manifest source evidence references an fpm8 record")
        for path_key, hash_key in (("validation_path", "validation_sha256"), ("summary_path", "summary_sha256")):
            path = Path(row[path_key])
            if not path.is_absolute() or not path.is_file() or digest(path) != row[hash_key]:
                raise SystemExit(f"source evidence missing or hash mismatch: {path}")
        validation = json.loads(Path(row["validation_path"]).read_text(encoding="utf-8"))
        if not validation.get("summary_valid") or not validation.get("mix_valid") or \
           int(validation.get("operation_failures", -1)) != 0 or \
           not validation.get("throughput_usable"):
            raise SystemExit(f"source cell not valid and throughput-usable: {row['session']}/{row['scale']}")
        value = validation.get("throughput_ops_per_sec")
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise SystemExit(f"source cell has invalid throughput: {row['session']}/{row['scale']}")

# Require frozen provenance settings rather than runner-side defaults.
if not manifest.get("selection_rule") or not manifest.get("selection_scores"):
    raise SystemExit("manifest lacks frozen model-selection provenance")
if not manifest.get("analyzer_source_sha256") or not manifest.get("predictor_source_sha256"):
    raise SystemExit("manifest lacks predictor/preprocessing source hashes")
print("manifest and all 40 source cells validated")
PY

if ((DRY_RUN)); then
  PREFIX="bsr8-change-dryrun-$(date -u +%Y%m%dT%H%M%SZ)-$$"
  if command -v docker >/dev/null && docker info >/dev/null 2>&1; then
    for session in 101 102 103 104; do
      for phase in probe full; do
        for scale in 20 40 60 80 100; do
          if [[ "${phase}" == probe ]]; then
            selected="$(python3 -c 'import json,sys; print(" ".join(map(str,json.load(open(sys.argv[1]))["selected_loads"])))' "${MANIFEST}")"
            [[ " ${selected} " == *" ${scale} "* ]] || continue
          fi
          cell="fpm8-session${session}-${phase}-scale${scale}"
          for kind in mysql mem web client; do
            if docker container inspect "${PREFIX}-${kind}-${cell}" >/dev/null 2>&1; then
              printf 'exact container name collision in dry-run: %s\n' "${PREFIX}-${kind}-${cell}" >&2
              exit 1
            fi
          done
        done
      done
    done
    if docker network inspect "${PREFIX}-net" >/dev/null 2>&1; then
      printf 'exact network name collision in dry-run: %s\n' "${PREFIX}-net" >&2; exit 1
    fi
    printf 'resource_name_preflight=passed\n'
  else
    printf 'resource_name_preflight=skipped_docker_unavailable\n'
  fi
  printf 'dry_run=valid\nmanifest=%s\nmanifest_sha256=%s\nprefix=%s\nprobe_loads=' \
    "${MANIFEST}" "${EXPECTED_SHA256,,}" "${PREFIX}"
  python3 - "${MANIFEST}" <<'PY'
import json, sys
print(",".join(map(str, json.load(open(sys.argv[1], encoding="utf-8"))["selected_loads"])))
PY
  printf 'target_sessions=101,102,103,104\nphases=probe,prediction_lock,full_reference\n'
  exit 0
fi

command -v docker >/dev/null || { printf 'docker is required\n' >&2; exit 2; }
command -v timeout >/dev/null || { printf 'coreutils timeout is required\n' >&2; exit 2; }
command -v awk >/dev/null || { printf 'awk is required\n' >&2; exit 2; }

PREFIX="bsr8-change-$(date -u +%Y%m%dT%H%M%SZ)-$$"
NETWORK="${PREFIX}-net"
RESULT_DIR="${ROUND_DIR}/results/change-${PREFIX}"
mkdir -p "${RESULT_DIR}"
CREATED=()
declare -A CREATED_IDS=()
NETWORK_ID=""
RUN_STATUS="incomplete"
RUNNING_CELL=""
SCORE_EXIT_CODE=0
trap 'on_exit $?' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

on_exit() {
  local status="$1" name actual_id
  trap - EXIT INT TERM
  for name in "${CREATED[@]}"; do archive_container "${name}" || true; done
  if [[ -n "${NETWORK_ID}" ]]; then
    actual_id="$(docker network inspect -f '{{.Id}}' "${NETWORK}" 2>/dev/null || true)"
    if [[ -n "${actual_id}" && "${actual_id}" == "${NETWORK_ID}" ]]; then
      docker network rm "${NETWORK}" >/dev/null 2>&1 || true
    fi
  fi
  if [[ "${RUN_STATUS}" == "complete" && "${status}" == 0 ]]; then
    printf 'complete\n' > "${RESULT_DIR}/status.txt"
  elif [[ "${RUN_STATUS}" == "collection_complete_analysis_incomplete" && "${status}" == 0 ]]; then
    printf 'collection_complete\nanalysis=incomplete\nscoring_exit_code=%s\nexit_code=0\n' \
      "${SCORE_EXIT_CODE}" > "${RESULT_DIR}/status.txt"
  else
    printf 'incomplete\nexit_code=%s\nactive_cell=%s\n' "${status}" "${RUNNING_CELL}" > "${RESULT_DIR}/status.txt"
  fi
  exit "${status}"
}

archive_container() {
  local name="$1" expected_id actual_id
  local cid=""
  for registered in "${CREATED[@]}"; do [[ "${registered}" == "${name}" ]] && cid="yes"; done
  [[ -n "${cid}" ]] || return 0
  expected_id="${CREATED_IDS[${name}]:-}"
  [[ -n "${expected_id}" ]] || { printf 'missing_saved_container_id name=%s\n' "${name}" >> "${RESULT_DIR}/status-events.txt"; return 1; }
  actual_id="$(docker container inspect -f '{{.Id}}' "${name}" 2>/dev/null || true)"
  [[ -n "${actual_id}" ]] || return 0
  if [[ "${actual_id}" != "${expected_id}" ]]; then
    printf 'container_id_mismatch name=%s expected=%s actual=%s\n' "${name}" "${expected_id}" "${actual_id}" >> "${RESULT_DIR}/status-events.txt"
    return 1
  fi
  docker logs "${expected_id}" > "${RESULT_DIR}/${name}.container.log" 2>&1 || true
  docker inspect "${expected_id}" > "${RESULT_DIR}/${name}.inspect.json" 2>/dev/null || true
  docker diff "${expected_id}" > "${RESULT_DIR}/${name}.diff.txt" 2>/dev/null || true
  actual_id="$(docker container inspect -f '{{.Id}}' "${name}" 2>/dev/null || true)"
  [[ "${actual_id}" == "${expected_id}" ]] || {
    printf 'container_id_changed_before_remove name=%s expected=%s actual=%s\n' "${name}" "${expected_id}" "${actual_id}" >> "${RESULT_DIR}/status-events.txt"
    return 1
  }
  docker rm -f "${expected_id}" >/dev/null 2>&1 || true
}

container_id() {
  local name="$1" id owner
  id="$(docker container inspect -f '{{.Id}}' "${name}")"
  owner="$(docker container inspect -f '{{ index .Config.Labels "org.moebench.runner" }}' "${name}")"
  [[ "${owner}" == "${PREFIX}" ]] || { printf 'container ownership label mismatch: %s\n' "${name}" >&2; return 1; }
  CREATED+=("${name}")
  CREATED_IDS["${name}"]="${id}"
  printf '%s\n' "${id}" > "${RESULT_DIR}/${name}.container-id.txt"
  printf '%s' "${id}"
}

assert_available_name() {
  local name="$1"
  if docker container inspect "${name}" >/dev/null 2>&1; then
    printf 'exact container name collision: %s\n' "${name}" >&2
    return 1
  fi
}

# The only container inspections before creation are these exact planned names.
for session in 101 102 103 104; do
  for phase in probe full; do
    for scale in "${LOADS[@]}"; do
      [[ "${phase}" != probe ]] || {
        selected="$(python3 -c 'import json,sys; print(" ".join(map(str,json.load(open(sys.argv[1]))["selected_loads"])))' "${MANIFEST}")"
        [[ " ${selected} " == *" ${scale} "* ]] || continue
      }
      cell="fpm8-session${session}-${phase}-scale${scale}"
      for kind in mysql mem web client; do assert_available_name "${PREFIX}-${kind}-${cell}"; done
    done
  done
done
if docker network inspect "${NETWORK}" >/dev/null 2>&1; then
  printf 'exact network name collision: %s\n' "${NETWORK}" >&2
  exit 1
fi

for image in "${SEEDED_IMAGE}" "${WEB_IMAGE}" "${MEM_IMAGE}" "${CLIENT_IMAGE}"; do
  docker image inspect "${image}" >/dev/null 2>&1 || { printf 'missing image: %s\n' "${image}" >&2; exit 2; }
done

: > "${RESULT_DIR}/runs.jsonl"
: > "${RESULT_DIR}/phase-times.jsonl"
: > "${RESULT_DIR}/cell-times.jsonl"
: > "${RESULT_DIR}/randomized-order.txt"
printf 'started_utc=%s\nmanifest=%s\nmanifest_sha256=%s\npredictor_method=%s\npredictor_source_sha256=%s\n' \
  "$(date -u +%FT%TZ)" "${MANIFEST}" "${EXPECTED_SHA256,,}" \
  "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["predictor_method"])' "${MANIFEST}")" \
  "$(sha256sum "${PREDICTOR}" | awk '{print $1}')" > "${RESULT_DIR}/run-metadata.txt"
docker version --format '{{.Server.Version}}' >> "${RESULT_DIR}/run-metadata.txt"
docker image inspect "${SEEDED_IMAGE}" "${WEB_IMAGE}" "${MEM_IMAGE}" "${CLIENT_IMAGE}" \
  --format '{{index .RepoTags 0}} {{.Id}} {{json .RepoDigests}}' > "${RESULT_DIR}/image-digests.txt"
docker network create "${NETWORK}" > "${RESULT_DIR}/network-id.txt"
NETWORK_ID="$(<"${RESULT_DIR}/network-id.txt")"

host_snapshot() {
  local session="$1"
  {
    printf 'session=%s\nrecorded_utc=%s\n' "${session}" "$(date -u +%FT%TZ)"
    printf 'manifest=%s\nmanifest_sha256=%s\n' "${MANIFEST}" "${EXPECTED_SHA256,,}"
    printf 'randomization_seed=%s\n' "${seed}"
    uptime
    cat /proc/loadavg
    free -b
    printf '\nimage_digests:\n'
    cat "${RESULT_DIR}/image-digests.txt"
  } > "${RESULT_DIR}/session${session}-host-load.txt"
}

start_backends() {
  local cell="$1" db="${PREFIX}-mysql-${cell}" mem="${PREFIX}-mem-${cell}" ready=0
  docker run -d --name "${db}" --label "org.moebench.runner=${PREFIX}" --network "${NETWORK}" --network-alias mysql_server \
    --entrypoint /usr/sbin/mysqld "${SEEDED_IMAGE}" > "${RESULT_DIR}/${db}.container-id.txt"
  container_id "${db}" >/dev/null
  for _ in $(seq 1 120); do
    if docker exec "${db}" mysql -uroot -proot -Nse 'SELECT 1' >/dev/null 2>&1; then ready=1; break; fi
    if [[ "$(docker inspect -f '{{.State.Running}}' "${db}" 2>/dev/null || true)" != true ]]; then break; fi
    sleep 2
  done
  [[ "${ready}" == 1 ]] || { printf 'database_not_ready cell=%s\n' "${cell}" >> "${RESULT_DIR}/status-events.txt"; return 1; }
  local tables="${RESULT_DIR}/${db}.tables.txt" expected="${RESULT_DIR}/${db}.expected-tables.sorted.txt"
  local actual="${RESULT_DIR}/${db}.actual-tables.sorted.txt" count entities metadata
  count="$(docker exec "${db}" mysql -uroot -proot -Nse 'SELECT COUNT(*) FROM information_schema.tables WHERE table_schema="ELGG_DB";')"
  docker exec "${db}" mysql -uroot -proot -Nse 'SELECT table_name FROM information_schema.tables WHERE table_schema="ELGG_DB" ORDER BY table_name;' > "${tables}"
  sort "${EXPECTED_TABLES_FILE}" > "${expected}"; sort "${tables}" > "${actual}"
  entities="$(docker exec "${db}" mysql -uroot -proot -Nse 'SELECT COUNT(*) FROM ELGG_DB.elgg_entities;')"
  metadata="$(docker exec "${db}" mysql -uroot -proot -Nse 'SELECT COUNT(*) FROM ELGG_DB.elgg_metadata;')"
  printf 'database=ELGG_DB\ntable_count=%s\nelgg_entities_rows=%s\nelgg_metadata_rows=%s\n' \
    "${count}" "${entities}" "${metadata}" > "${RESULT_DIR}/${db}.validation.txt"
  if [[ "${count}" != 18 || "${entities}" -lt 1 || "${metadata}" -lt 1 ]] || ! diff -u "${expected}" "${actual}" > "${RESULT_DIR}/${db}.schema.diff"; then
    printf 'database_seed_invalid cell=%s\n' "${cell}" >> "${RESULT_DIR}/status-events.txt"; return 1
  fi
  docker run -d --name "${mem}" --label "org.moebench.runner=${PREFIX}" --network "${NETWORK}" --network-alias memcache_server \
    "${MEM_IMAGE}" -m 512 -t 2 > "${RESULT_DIR}/${mem}.container-id.txt"
  container_id "${mem}" >/dev/null
}

start_web() {
  local cell="$1" ready=0 web="${PREFIX}-web-${cell}"
  docker run -d --name "${web}" --label "org.moebench.runner=${PREFIX}" --network "${NETWORK}" \
    --network-alias root-server --network-alias web_server "${WEB_IMAGE}" \
    bash -lc 'set -Eeuo pipefail; /etc/bootstrap.sh http root-server mysql_server memcache_server 8 auto; exec tail -f /dev/null' \
    > "${RESULT_DIR}/${web}.container-id.txt"
  container_id "${web}" >/dev/null
  for _ in $(seq 1 90); do
    if docker exec "${web}" wget -q -O /dev/null http://127.0.0.1:8080/; then ready=1; break; fi
    if [[ "$(docker inspect -f '{{.State.Running}}' "${web}" 2>/dev/null || true)" != true ]]; then break; fi
    sleep 2
  done
  docker logs "${web}" > "${RESULT_DIR}/${web}.bootstrap.log" 2>&1 || true
  [[ "${ready}" == 1 ]] || { printf 'web_not_ready cell=%s\n' "${cell}" >> "${RESULT_DIR}/status-events.txt"; return 1; }
}

cleanup_cell() {
  local cell="$1" kind
  for kind in client web mem mysql; do archive_container "${PREFIX}-${kind}-${cell}"; done
}

run_cell() {
  local session="$1" phase="$2" scale="$3" steady="$4"
  local cell="fpm8-session${session}-${phase}-scale${scale}" client="${PREFIX}-client-fpm8-session${session}-${phase}-scale${scale}"
  local cell_start_ns client_start_ns client_end_ns cell_end_ns client_elapsed cell_total setup_start setup_end setup_seconds client_code copy_code validation_code protocol_error=0 client_owned=0 client_id=""
  local outdir="${RESULT_DIR}/${cell}-faban-output" summary="" summary_count=0 xml_hash="" valid_json="${RESULT_DIR}/${cell}-validation.json"
  RUNNING_CELL="${cell}"
  cell_start_ns="$(date +%s%N)"; setup_start="${cell_start_ns}"
  if ! start_backends "fpm8-session${session}-${phase}-scale${scale}"; then cleanup_cell "fpm8-session${session}-${phase}-scale${scale}"; return 1; fi
  if ! start_web "fpm8-session${session}-${phase}-scale${scale}"; then cleanup_cell "fpm8-session${session}-${phase}-scale${scale}"; return 1; fi
  setup_end="$(date +%s%N)"
  client_start_ns="$(date +%s%N)"
  set +e
  timeout --signal=TERM --kill-after=10s 300s docker run --name "${client}" --label "org.moebench.runner=${PREFIX}" --network "${NETWORK}" "${CLIENT_IMAGE}" \
    --oper=usergen\&run --ramp-up=5 --ramp-down=5 --steady="${steady}" \
    --min=1000 --max=1500 --type=THINKTIME --dist=fixed root-server "${scale}" \
    > "${RESULT_DIR}/${cell}.stdout.log" 2>&1
  client_code=$?
  set -e
  client_end_ns="$(date +%s%N)"
  if docker container inspect "${client}" >/dev/null 2>&1; then
    if container_id "${client}" >/dev/null; then
      client_owned=1
      client_id="${CREATED_IDS[${client}]}"
    else
      printf 'client_container_ownership_unverified name=%s\n' "${client}" >> "${RESULT_DIR}/status-events.txt"
    fi
  fi
  client_elapsed="$(awk -v a="${client_start_ns}" -v b="${client_end_ns}" 'BEGIN {printf "%.6f", (b-a)/1000000000}')"
  setup_seconds="$(awk -v a="${setup_start}" -v b="${setup_end}" 'BEGIN {printf "%.6f", (b-a)/1000000000}')"
  set +e
  if [[ "${client_owned}" == 1 ]]; then
    docker cp "${client_id}:/usr/src/faban/output" "${outdir}" >/dev/null 2>&1
    copy_code=$?
  else
    copy_code=1
  fi
  set -e
  if [[ "${copy_code}" == 0 ]]; then
    find "${outdir}" -type f -printf '%p %s bytes\n' > "${RESULT_DIR}/${cell}-faban-files.txt" 2>/dev/null || true
    while IFS= read -r -d '' item; do summary="${item}"; summary_count=$((summary_count + 1)); done < <(find "${outdir}" -type f -name summary.xml -print0)
  fi
  validation_code=0
  if [[ "${copy_code}" == 0 && "${summary_count}" == 1 ]]; then
    python3 "${VALIDATOR}" "${summary}" "${valid_json}" > "${RESULT_DIR}/${cell}-validation.stdout.txt" 2> "${RESULT_DIR}/${cell}-validation.stderr.txt" || validation_code=$?
    [[ -f "${summary}" ]] && xml_hash="$(sha256sum "${summary}" | awk '{print $1}')"
    printf '%s  %s\n' "${xml_hash}" "${summary}" > "${RESULT_DIR}/${cell}-summary.sha256"
  else
    printf 'expected_exactly_one_summary_xml found=%s\n' "${summary_count}" > "${RESULT_DIR}/${cell}-validation.stderr.txt"
    printf '{"valid":false,"summary_valid":false,"summary_xml_count":%s,"reason":"missing_or_ambiguous_summary_xml"}\n' "${summary_count}" > "${valid_json}"
  fi
  if rg -q 'does not contain a hostname|FileNotFoundException: http://[^[:space:]]+/http://|Exception in thread' "${RESULT_DIR}/${cell}.stdout.log"; then
    protocol_error=1; printf 'client_protocol_error_in_stdout\n' >> "${RESULT_DIR}/${cell}-validation.stderr.txt"
  fi
  if [[ "${client_owned}" == 1 ]]; then
    docker logs "${client_id}" > "${RESULT_DIR}/${cell}.container.log" 2>&1 || true
    docker inspect "${client_id}" > "${RESULT_DIR}/${cell}.inspect.json" 2>/dev/null || true
  fi
  local bench_passed=null mix_valid=false throughput_usable=false latency_passed=null failed_ops='[]' slo_ops='[]'
  if [[ -f "${valid_json}" ]]; then
    read -r bench_passed mix_valid throughput_usable latency_passed failed_ops slo_ops < <(python3 - "${valid_json}" <<'PY'
import json,sys
d=json.load(open(sys.argv[1],encoding="utf-8"))
failed=sorted(row["name"] for row in d.get("operations",[]) if row.get("failures",0)>0)
print(json.dumps(d.get("bench_passed"),separators=(",",":")),json.dumps(d.get("mix_valid",False),separators=(",",":")),json.dumps(d.get("throughput_usable",False),separators=(",",":")),json.dumps(d.get("latency_slo_passed"),separators=(",",":")),json.dumps(failed,separators=(",",":")),json.dumps(d.get("latency_slo_violations",[]),separators=(",",":")))
PY
)
  fi
  cleanup_cell "fpm8-session${session}-${phase}-scale${scale}"
  cell_end_ns="$(date +%s%N)"
  cell_total="$(awk -v a="${cell_start_ns}" -v b="${cell_end_ns}" 'BEGIN {printf "%.6f", (b-a)/1000000000}')"
  printf '{"cell":"%s","session":%s,"phase":"%s","scale":%s,"planned_steady_seconds":%s,"cell_start_epoch_ns":%s,"client_start_epoch_ns":%s,"client_end_epoch_ns":%s,"cell_end_epoch_ns":%s,"elapsed_seconds":%s,"client_execution_seconds":%s,"cell_total_seconds":%s,"setup_seconds":%s,"client_exit_code":%s,"output_copied":%s,"summary_xml":"%s","summary_sha256":"%s","validation_json":"%s","bench_passed":%s,"operation_mix_valid":%s,"throughput_usable":%s,"latency_slo_passed":%s,"failed_operations":%s,"latency_slo_violations":%s,"client_protocol_error":%s}\n' \
    "${cell}" "${session}" "${phase}" "${scale}" "${steady}" "${cell_start_ns}" "${client_start_ns}" "${client_end_ns}" "${cell_end_ns}" "${client_elapsed}" "${client_elapsed}" "${cell_total}" "${setup_seconds}" \
    "${client_code}" "$([[ "${copy_code}" == 0 ]] && printf true || printf false)" "${summary}" "${xml_hash}" "${valid_json}" \
    "${bench_passed}" "${mix_valid}" "${throughput_usable}" "${latency_passed}" "${failed_ops}" "${slo_ops}" "$([[ "${protocol_error}" == 0 ]] && printf false || printf true)" >> "${RESULT_DIR}/runs.jsonl"
  printf '{"session":%s,"phase":"%s","load":%s,"setup_seconds":%s,"runner_seconds":%s,"cell_total_seconds":%s,"planned_steady_seconds":%s,"cell_start_epoch_ns":%s,"client_start_epoch_ns":%s,"client_end_epoch_ns":%s,"cell_end_epoch_ns":%s,"status":"%s"}\n' \
    "${session}" "${phase}" "${scale}" "${setup_seconds}" "${client_elapsed}" "${cell_total}" "${steady}" "${cell_start_ns}" "${client_start_ns}" "${client_end_ns}" "${cell_end_ns}" \
    "$([[ "${client_code}" == 0 && "${copy_code}" == 0 && "${validation_code}" == 0 && "${protocol_error}" == 0 ]] && printf complete || printf failed)" >> "${RESULT_DIR}/cell-times.jsonl"
  RUNNING_CELL=""
  [[ "${client_code}" == 0 && "${copy_code}" == 0 && "${validation_code}" == 0 && "${protocol_error}" == 0 ]] || return 1
  if [[ "${phase}" == probe && ( "${mix_valid}" != true || "${throughput_usable}" != true || "${failed_ops}" != '[]' ) ]]; then
    printf 'probe_not_usable cell=%s mix_valid=%s throughput_usable=%s failed_operations=%s\n' \
      "${cell}" "${mix_valid}" "${throughput_usable}" "${failed_ops}" >> "${RESULT_DIR}/status-events.txt"
    return 1
  fi
  return 0
}

random_order() {
  local seed="$1"; shift
  python3 - "$seed" "$@" <<'PY'
import random,sys
r=random.Random(int(sys.argv[1])); values=list(map(int,sys.argv[2:])); r.shuffle(values); print(" ".join(map(str,values)))
PY
}

manifest_values="$(python3 - "${MANIFEST}" <<'PY'
import json,sys
m=json.load(open(sys.argv[1],encoding="utf-8"))
print(m["randomization_seed"])
print(" ".join(map(str,m["selected_loads"])))
PY
)"
seed="$(sed -n '1p' <<<"${manifest_values}")"
read -r -a selected_loads <<<"$(sed -n '2p' <<<"${manifest_values}")"

for session in 101 102 103 104; do
  host_snapshot "${session}"
  probe_order=( $(random_order "$((seed + session * 1009))" "${selected_loads[@]}") )
  full_order=( $(random_order "$((seed + session * 7919 + 17))" "${LOADS[@]}") )
  printf 'session=%s phase=probe seed=%s order=%s\n' "${session}" "$((seed + session * 1009))" "$(IFS=,; printf '%s' "${probe_order[*]}")" >> "${RESULT_DIR}/randomized-order.txt"
  phase_start="$(date +%s%N)"
  for scale in "${probe_order[@]}"; do
    run_cell "${session}" probe "${scale}" "${PROBE_SECONDS}" || {
      printf 'probe_phase_failed session=%s\n' "${session}" >> "${RESULT_DIR}/status-events.txt"; exit 1;
    }
  done
  phase_end="$(date +%s%N)"
  printf '{"session":%s,"phase":"probe","start_epoch_ns":%s,"end_epoch_ns":%s,"elapsed_seconds":%s,"status":"complete"}\n' \
    "${session}" "${phase_start}" "${phase_end}" "$(awk -v a="${phase_start}" -v b="${phase_end}" 'BEGIN{printf "%.6f",(b-a)/1000000000}')" >> "${RESULT_DIR}/phase-times.jsonl"

  prediction="${RESULT_DIR}/fpm8-session${session}-prediction.json"
  provenance="${RESULT_DIR}/fpm8-session${session}-prediction-provenance.json"
  args=(predict --manifest "${MANIFEST}" --session "${session}" --output-prediction "${prediction}")
  for scale in "${probe_order[@]}"; do args+=(--probe-validation "${RESULT_DIR}/fpm8-session${session}-probe-scale${scale}-validation.json"); done
  [[ "$(sha256sum "${MANIFEST}" | awk '{print $1}')" == "${EXPECTED_SHA256,,}" ]] || {
    printf 'manifest_changed_before_prediction session=%s\n' "${session}" >> "${RESULT_DIR}/status-events.txt"; exit 1;
  }
  predictor_start_ns="$(date +%s%N)"
  predictor_started="$(date -u +%FT%TZ)"
  python3 "${PREDICTOR}" "${args[@]}" > "${RESULT_DIR}/fpm8-session${session}-predictor.stdout.txt" 2> "${RESULT_DIR}/fpm8-session${session}-predictor.stderr.txt" || {
    printf 'prediction_failed session=%s\n' "${session}" >> "${RESULT_DIR}/status-events.txt"; exit 1;
  }
  prediction_hash="$(sha256sum "${prediction}" | awk '{print $1}')"
  python3 - "${prediction}" "${EXPECTED_SHA256,,}" "${session}" "${MANIFEST}" <<'PY'
import hashlib,json,sys
from pathlib import Path
p=Path(sys.argv[1]); expected=sys.argv[2]; session=int(sys.argv[3]); manifest=Path(sys.argv[4])
d=json.loads(p.read_text(encoding="utf-8"))
actual=hashlib.sha256(manifest.read_bytes()).hexdigest()
m=json.loads(manifest.read_text(encoding="utf-8"))
comparators=d.get("same_budget_comparators_ops_per_sec",{})
if (actual != expected or d.get("manifest_sha256") != expected or
    d.get("target_session") != session or d.get("model_id") != m.get("model_id") or
    set(comparators) != {"correlation","history_scale","linear_interpolation"} or
    comparators.get(m["predictor_method"]) != d.get("predicted_throughput_ops_per_sec")):
    raise SystemExit("prediction output does not match pinned manifest/session/model")
PY
  python3 - "${provenance}" "${session}" "${MANIFEST}" "${EXPECTED_SHA256,,}" "${prediction}" "${prediction_hash}" "${PREDICTOR}" "${probe_order[*]}" "${predictor_started}" <<'PY'
import hashlib,json,os,sys,tempfile
from pathlib import Path
dest=Path(sys.argv[1]); session=int(sys.argv[2]); manifest=Path(sys.argv[3]); prediction=Path(sys.argv[5]); predictor=Path(sys.argv[7])
probe_scales=list(map(int,sys.argv[8].split()))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
rows=[]
for scale in probe_scales:
    v=Path(str(prediction.parent/f"fpm8-session{session}-probe-scale{scale}-validation.json"))
    d=json.loads(v.read_text(encoding="utf-8")); x=Path(d["summary_xml"])
    rows.append({"scale":scale,"validation_path":str(v.resolve()),"validation_sha256":sha(v),"summary_path":str(x.resolve()),"summary_sha256":sha(x)})
command=["python3",str(predictor),"predict","--manifest",str(manifest.resolve()),"--session",str(session)]
for scale in probe_scales:
    command.extend(["--probe-validation",str(prediction.parent/f"fpm8-session{session}-probe-scale{scale}-validation.json")])
command.extend(["--output-prediction",str(prediction.resolve())])
data={"target_session":session,"manifest_path":str(manifest.resolve()),"manifest_sha256":sys.argv[4],"prediction_path":str(prediction.resolve()),"prediction_sha256":sys.argv[6],"model_id":json.loads(prediction.read_text(encoding="utf-8"))["model_id"],"predictor_path":str(predictor.resolve()),"predictor_sha256":sha(predictor),"probe_cells":rows,"command_argv":command,"started_utc":sys.argv[9],"completed_utc":__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()}
fd,tmp=tempfile.mkstemp(prefix=f".{dest.name}.",dir=dest.parent)
try:
    with os.fdopen(fd,"w",encoding="utf-8") as f: json.dump(data,f,indent=2,allow_nan=False); f.write("\n"); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,dest)
except Exception:
    try: os.unlink(tmp)
    except OSError: pass
    raise
PY
  printf '%s  %s\n' "${prediction_hash}" "${prediction}" > "${prediction}.sha256"
  provenance_hash="$(sha256sum "${provenance}" | awk '{print $1}')"
  printf '%s  %s\n' "${provenance_hash}" "${provenance}" > "${provenance}.sha256"
  [[ "$(sha256sum "${prediction}" | awk '{print $1}')" == "${prediction_hash}" && \
     "$(sha256sum "${provenance}" | awk '{print $1}')" == "${provenance_hash}" ]] || {
    printf 'prediction_lock_hash_verification_failed session=%s\n' "${session}" >> "${RESULT_DIR}/status-events.txt"; exit 1;
  }
  chmod a-w "${prediction}" "${prediction}.sha256" "${provenance}" "${provenance}.sha256"
  predictor_end_ns="$(date +%s%N)"
  printf '{"session":%s,"phase":"prediction_lock","start_epoch_ns":%s,"end_epoch_ns":%s,"elapsed_seconds":%s,"prediction_sha256":"%s","provenance_sha256":"%s","status":"locked"}\n' \
    "${session}" "${predictor_start_ns}" "${predictor_end_ns}" \
    "$(awk -v a="${predictor_start_ns}" -v b="${predictor_end_ns}" 'BEGIN{printf "%.6f",(b-a)/1000000000}')" \
    "${prediction_hash}" "${provenance_hash}" >> "${RESULT_DIR}/phase-times.jsonl"
  printf 'prediction_locked session=%s prediction_sha256=%s provenance_sha256=%s\n' \
    "${session}" "${prediction_hash}" "${provenance_hash}" >> "${RESULT_DIR}/status-events.txt"

  printf 'session=%s phase=full seed=%s order=%s\n' "${session}" "$((seed + session * 7919 + 17))" "$(IFS=,; printf '%s' "${full_order[*]}")" >> "${RESULT_DIR}/randomized-order.txt"
  phase_start="$(date +%s%N)"
  for scale in "${full_order[@]}"; do
    run_cell "${session}" full "${scale}" "${REFERENCE_SECONDS}" || {
      printf 'full_cell_failed session=%s scale=%s\n' "${session}" "${scale}" >> "${RESULT_DIR}/status-events.txt"
      # Keep collecting the remaining locked full-reference cells.
    }
  done
  phase_end="$(date +%s%N)"
  printf '{"session":%s,"phase":"full","start_epoch_ns":%s,"end_epoch_ns":%s,"elapsed_seconds":%s,"status":"complete_with_cell_failures_possible"}\n' \
    "${session}" "${phase_start}" "${phase_end}" "$(awk -v a="${phase_start}" -v b="${phase_end}" 'BEGIN{printf "%.6f",(b-a)/1000000000}')" >> "${RESULT_DIR}/phase-times.jsonl"
done

score_rc=0
python3 "${SCORER}" --result-dir "${RESULT_DIR}" --manifest "${MANIFEST}" \
  --output-json "${RESULT_DIR}/fpm8-throughput-score.json" \
  > "${RESULT_DIR}/scoring.stdout.txt" 2> "${RESULT_DIR}/scoring.stderr.txt" || score_rc=$?
SCORE_EXIT_CODE="${score_rc}"
if [[ "${score_rc}" != 0 ]]; then
  printf 'scoring_incomplete exit_code=%s\n' "${score_rc}" >> "${RESULT_DIR}/status-events.txt"
  RUN_STATUS="collection_complete_analysis_incomplete"
else
  RUN_STATUS="complete"
fi
printf 'completed_utc=%s\n' "$(date -u +%FT%TZ)" >> "${RESULT_DIR}/run-metadata.txt"
printf 'Evidence: %s\n' "${RESULT_DIR}"
