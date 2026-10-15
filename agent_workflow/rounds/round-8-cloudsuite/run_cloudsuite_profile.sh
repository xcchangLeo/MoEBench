#!/usr/bin/env bash
set -Eeuo pipefail

ROUND_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULT_DIR="${ROUND_DIR}/results"
mkdir -p "${RESULT_DIR}"

PREFIX="bsr8-$(date -u +%H%M%S)"
NETWORK="${PREFIX}-net"
DB="${PREFIX}-mysql"
MEM="${PREFIX}-memcache"
CREATED=()
NETWORK_CREATED=0

cleanup() {
  local name
  for name in "${CREATED[@]}"; do
    docker logs "${name}" > "${RESULT_DIR}/${name}.final.log" 2>&1 || true
    docker inspect "${name}" > "${RESULT_DIR}/${name}.final.inspect.json" 2>/dev/null || true
    docker diff "${name}" > "${RESULT_DIR}/${name}.diff.txt" 2>/dev/null || true
    docker rm -f "${name}" >/dev/null 2>&1 || true
  done
  if [[ "${NETWORK_CREATED}" == 1 ]]; then docker network rm "${NETWORK}" >/dev/null 2>&1 || true; fi
}
trap cleanup EXIT INT TERM

docker network create "${NETWORK}" > "${RESULT_DIR}/network-id.txt"
NETWORK_CREATED=1

if docker image inspect cloudsuite/round8-db-seeded:ready >/dev/null 2>&1; then
  docker run -d --name "${DB}" --network "${NETWORK}" --network-alias mysql_server \
    --user mysql --entrypoint /usr/sbin/mysqld cloudsuite/round8-db-seeded:ready \
    > "${RESULT_DIR}/database-container-id.txt"
else
  docker run -d --name "${DB}" --network "${NETWORK}" --network-alias mysql_server \
    --entrypoint /bin/bash cloudsuite/web-serving:db_server \
    -lc '/etc/entrypoint.sh; exec tail -f /dev/null' \
    > "${RESULT_DIR}/database-container-id.txt"
fi
CREATED+=("${DB}")

db_ready=0
for _ in $(seq 1 1200); do
  if docker exec "${DB}" mysql -uroot -proot -Nse 'SELECT 1' >/dev/null 2>&1; then db_ready=1; break; fi
  if [[ "$(docker inspect -f '{{.State.Running}}' "${DB}" 2>/dev/null || true)" != true ]]; then break; fi
  sleep 2
done
docker logs "${DB}" > "${RESULT_DIR}/database.log" 2>&1 || true
if [[ "${db_ready}" != 1 ]]; then printf 'database_not_ready\n' > "${RESULT_DIR}/status.txt"; exit 2; fi
docker exec "${DB}" mysql -uroot -proot -e 'SELECT 1 AS authenticated_query; SHOW DATABASES;' \
  > "${RESULT_DIR}/database-sql-check.txt"
docker exec "${DB}" mysql -uroot -proot -e 'SELECT COUNT(*) AS elgg_tables FROM information_schema.tables WHERE table_schema="ELGG_DB";' \
  >> "${RESULT_DIR}/database-sql-check.txt"
printf 'database_ready\n' > "${RESULT_DIR}/status.txt"

docker run -d --name "${MEM}" --network "${NETWORK}" --network-alias memcache_server \
  cloudsuite/web-serving:memcached_server -m 512 -t 2 > "${RESULT_DIR}/memcached-container-id.txt"
CREATED+=("${MEM}")
docker inspect "${DB}" "${MEM}" > "${RESULT_DIR}/infrastructure-inspect.json"

start_web() {
  local children="$1" tag="$2" ready=0
  local web="${PREFIX}-web-${tag}"
  docker run -d --name "${web}" --network "${NETWORK}" \
    --network-alias root_server --network-alias web_server \
    cloudsuite/web-serving:web_server \
    bash -lc "/etc/bootstrap.sh http root_server mysql_server memcache_server ${children} auto; exec tail -f /dev/null" \
    > "${RESULT_DIR}/${tag}-web-container-id.txt"
  CREATED+=("${web}")
  for _ in $(seq 1 90); do
    if docker exec "${web}" curl -fsS -o /dev/null http://127.0.0.1:8080/; then ready=1; break; fi
    if [[ "$(docker inspect -f '{{.State.Running}}' "${web}" 2>/dev/null || true)" != true ]]; then break; fi
    sleep 2
  done
  docker logs "${web}" > "${RESULT_DIR}/${tag}-web.log" 2>&1 || true
  if [[ "${ready}" != 1 ]]; then printf 'web_not_ready_%s\n' "${tag}" >> "${RESULT_DIR}/status.txt"; return 1; fi
}

run_cell() {
  local tag="$1" children="$2" scale="$3" session="$4"
  local cell="${tag}-session${session}-scale${scale}" client="${PREFIX}-client-${cell}"
  local start_ns end_ns exit_code elapsed
  start_ns="$(date +%s%N)"
  set +e
  docker run --name "${client}" --network "${NETWORK}" cloudsuite/web-serving:faban_client \
    --oper=usergen\&run --ramp-up=5 --ramp-down=5 --steady=30 \
    --min=1000 --max=1500 --type=THINKTIME --dist=fixed root_server "${scale}" \
    > "${RESULT_DIR}/${cell}.stdout.log" 2>&1
  exit_code=$?
  set -e
  end_ns="$(date +%s%N)"
  elapsed="$(awk -v a="${start_ns}" -v b="${end_ns}" 'BEGIN { printf "%.6f", (b-a)/1000000000 }')"
  printf '{"cell":"%s","php_fpm_max_children":%s,"scale":%s,"session":%s,"start_epoch_ns":%s,"end_epoch_ns":%s,"elapsed_seconds":%s,"exit_code":%s}\n' \
    "${cell}" "${children}" "${scale}" "${session}" "${start_ns}" "${end_ns}" "${elapsed}" "${exit_code}" \
    >> "${RESULT_DIR}/runs.jsonl"
  docker cp "${client}:/usr/src/faban/output" "${RESULT_DIR}/${cell}-faban-output" >/dev/null 2>&1 || true
  docker logs "${client}" > "${RESULT_DIR}/${cell}.container.log" 2>&1 || true
  docker inspect "${client}" > "${RESULT_DIR}/${cell}.inspect.json" 2>/dev/null || true
  CREATED+=("${client}")
  if [[ "${exit_code}" != 0 ]]; then printf 'workload_failed_%s_exit_%s\n' "${cell}" "${exit_code}" >> "${RESULT_DIR}/status.txt"; return 1; fi
}

: > "${RESULT_DIR}/runs.jsonl"
printf 'started_utc=%s\n' "$(date -u +%FT%TZ)" > "${RESULT_DIR}/run-metadata.txt"
docker version --format '{{.Server.Version}}' >> "${RESULT_DIR}/run-metadata.txt"
docker image inspect cloudsuite/web-serving:web_server cloudsuite/web-serving:db_server \
  cloudsuite/web-serving:memcached_server cloudsuite/web-serving:faban_client \
  --format '{{index .RepoTags 0}} {{.Id}} {{json .RepoDigests}}' > "${RESULT_DIR}/image-digests.txt"

for session in 1 2 3 4; do
  for spec in $(printf 'fpm4:4\nfpm8:8\n' | shuf); do
    tag="${spec%%:*}"
    children="${spec#*:}"
    printf 'session=%s config=%s scale_order=' "${session}" "${tag}" >> "${RESULT_DIR}/randomized-order.txt"
    load_order="$(shuf -e 20 40 60 80 100 | paste -sd, -)"
    printf '%s\n' "${load_order}" >> "${RESULT_DIR}/randomized-order.txt"
    start_web "${children}" "${tag}-session${session}"
    for scale in ${load_order//,/ }; do run_cell "${tag}" "${children}" "${scale}" "${session}"; done
    docker logs "${PREFIX}-web-${tag}-session${session}" \
      > "${RESULT_DIR}/${tag}-session${session}-web-final.log" 2>&1 || true
    docker rm -f "${PREFIX}-web-${tag}-session${session}" >/dev/null
  done
done

printf 'completed_utc=%s\n' "$(date -u +%FT%TZ)" >> "${RESULT_DIR}/run-metadata.txt"
printf 'complete\n' >> "${RESULT_DIR}/status.txt"
