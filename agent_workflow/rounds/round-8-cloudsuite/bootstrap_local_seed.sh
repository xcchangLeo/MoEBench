#!/usr/bin/env bash
set -Eeuo pipefail

ROUND_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCHIVE="${1:-${ROUND_DIR}/seed/ELGG_DB.tar.gz}"
EXPECTED_SHA256="7ca3c78076089f96a8f514e5a588b746e2d61e930373df45d28f9b948502307c"
SEEDED_IMAGE="cloudsuite/round8-db-seeded:ready"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)-$$"
CONTAINER="bsr8-seed-${STAMP}"
RESULT_DIR="${ROUND_DIR}/results/local-seed-${STAMP}"
STAGE_DIR=""
EXPECTED_TABLES_FILE="${ROUND_DIR}/expected-elgg-tables.txt"

mkdir -p "${RESULT_DIR}"

cleanup() {
  local status=$?
  if docker inspect "${CONTAINER}" >/dev/null 2>&1; then
    docker logs "${CONTAINER}" > "${RESULT_DIR}/database-container.log" 2>&1 || true
    docker inspect "${CONTAINER}" > "${RESULT_DIR}/database-container.inspect.json" 2>/dev/null || true
    docker diff "${CONTAINER}" > "${RESULT_DIR}/database-container.diff.txt" 2>/dev/null || true
    docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${STAGE_DIR}" && -d "${STAGE_DIR}" ]]; then rm -rf -- "${STAGE_DIR}"; fi
  exit "${status}"
}
trap cleanup EXIT INT TERM

if [[ ! -f "${ARCHIVE}" ]]; then
  printf 'Missing seed archive: %s\n' "${ARCHIVE}" >&2
  exit 2
fi
if docker image inspect "${SEEDED_IMAGE}" >/dev/null 2>&1; then
  printf 'Refusing to overwrite existing image %s\n' "${SEEDED_IMAGE}" >&2
  exit 2
fi
docker image inspect cloudsuite/web-serving:db_server --format '{{.Id}} {{json .RepoDigests}}' \
  > "${RESULT_DIR}/base-db-image.txt"

ACTUAL_SHA256="$(sha256sum "${ARCHIVE}" | awk '{print $1}')"
printf '%s  %s\n' "${ACTUAL_SHA256}" "${ARCHIVE}" > "${RESULT_DIR}/archive.sha256"
if [[ "${ACTUAL_SHA256}" != "${EXPECTED_SHA256}" ]]; then
  printf 'Unexpected seed archive SHA-256: %s\n' "${ACTUAL_SHA256}" >&2
  exit 2
fi

LISTING="${RESULT_DIR}/archive-files.txt"
tar -tzf "${ARCHIVE}" > "${LISTING}"
if ! awk 'BEGIN { ok=1; n=0 } /^backup\// { n++; if ($0 ~ /(^|\/)\.\.(\/|$)/ || $0 ~ /^\//) ok=0; next } { ok=0 } END { exit !(ok && n>0) }' "${LISTING}"; then
  printf 'Seed archive has an unexpected or unsafe path layout\n' >&2
  exit 2
fi
if ! rg -q '^backup/ELGG_DB/' "${LISTING}"; then
  printf 'Seed archive is missing backup/ELGG_DB\n' >&2
  exit 2
fi

STAGE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/bsr8-seed-${STAMP}.XXXXXX")"
tar -xzf "${ARCHIVE}" -C "${STAGE_DIR}" --no-same-owner --no-same-permissions
if [[ ! -f "${STAGE_DIR}/backup/xtrabackup_checkpoints" ]]; then
  printf 'Extracted backup is missing xtrabackup_checkpoints\n' >&2
  exit 2
fi
cp "${STAGE_DIR}/backup/xtrabackup_checkpoints" "${RESULT_DIR}/xtrabackup_checkpoints.txt"

docker run -d --name "${CONTAINER}" --network none \
  --mount "type=bind,source=${STAGE_DIR}/backup,target=/round8-seed/backup" \
  --entrypoint /bin/bash cloudsuite/web-serving:db_server -lc 'exec sleep infinity' \
  > "${RESULT_DIR}/database-container-id.txt"
IDENTITY="$(docker inspect --format '{{.Name}} {{.HostConfig.NetworkMode}}' "${CONTAINER}")"
printf '%s\n' "${IDENTITY}" > "${RESULT_DIR}/database-container-identity.txt"
if [[ "${IDENTITY}" != "/${CONTAINER} none" || "${CONTAINER}" != bsr8-seed-* ]]; then
  printf 'Refusing to modify unexpected container: %s\n' "${IDENTITY}" >&2
  exit 3
fi
docker exec "${CONTAINER}" bash -lc \
  'find /var/lib/mysql -mindepth 1 -maxdepth 2 -printf "%P %y %s %m %u:%g\\n" | sort' \
  > "${RESULT_DIR}/initial-datadir-manifest.txt"

cp "${EXPECTED_TABLES_FILE}" "${RESULT_DIR}/expected-elgg-tables.txt"

docker exec "${CONTAINER}" bash -lc '
  set -Eeuo pipefail
  command -v mariabackup >/dev/null
  install -d -o mysql -g mysql /var/lib/mysql /var/run/mysqld
  if mysqladmin ping --silent >/dev/null 2>&1; then
    echo "refusing to clear a live MySQL datadir" >&2
    exit 3
  fi
  find /var/lib/mysql -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +
  if find /var/lib/mysql -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
    echo "failed to clear disposable base-image datadir" >&2
    exit 3
  fi
  mariabackup --prepare --target-dir=/round8-seed/backup
  mariabackup --copy-back --target-dir=/round8-seed/backup --datadir=/var/lib/mysql
  chown -R mysql:mysql /var/lib/mysql
  if [[ -x /etc/init.d/mariadb ]]; then
    service mariadb start
  elif [[ -x /etc/init.d/mysql ]]; then
    service mysql start
  else
    echo "neither mariadb nor mysql init service exists" >&2
    exit 4
  fi
' > "${RESULT_DIR}/restore.log" 2>&1

ready=0
for _ in $(seq 1 120); do
  if docker exec "${CONTAINER}" mysql -uroot -proot -Nse 'SELECT 1' >/dev/null 2>&1; then
    ready=1
    break
  fi
  if [[ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER}" 2>/dev/null || true)" != true ]]; then break; fi
  sleep 2
done
docker logs "${CONTAINER}" > "${RESULT_DIR}/database-ready.log" 2>&1 || true
if [[ "${ready}" != 1 ]]; then
  printf 'Seeded database did not accept authenticated root query\n' >&2
  exit 3
fi

TABLE_COUNT="$(docker exec "${CONTAINER}" mysql -uroot -proot -Nse \
  'SELECT COUNT(*) FROM information_schema.tables WHERE table_schema="ELGG_DB";')"
printf 'database=ELGG_DB\ntable_count=%s\n' "${TABLE_COUNT}" > "${RESULT_DIR}/database-validation.txt"
docker exec "${CONTAINER}" mysql -uroot -proot -Nse \
  'SELECT table_name FROM information_schema.tables WHERE table_schema="ELGG_DB" ORDER BY table_name;' \
  > "${RESULT_DIR}/actual-elgg-tables.txt"
sort "${EXPECTED_TABLES_FILE}" > "${RESULT_DIR}/expected-elgg-tables.sorted.txt"
sort "${RESULT_DIR}/actual-elgg-tables.txt" > "${RESULT_DIR}/actual-elgg-tables.sorted.txt"
if [[ ! "${TABLE_COUNT}" =~ ^[0-9]+$ || "${TABLE_COUNT}" -ne 18 ]] || \
   ! diff -u "${RESULT_DIR}/expected-elgg-tables.sorted.txt" \
      "${RESULT_DIR}/actual-elgg-tables.sorted.txt" \
     > "${RESULT_DIR}/elgg-table-schema.diff"; then
  printf 'ELGG_DB schema mismatch: expected 18 archived tables, found %s\n' "${TABLE_COUNT}" >&2
  exit 3
fi
docker exec "${CONTAINER}" mysql -uroot -proot -e \
  'SELECT COUNT(*) AS elgg_entities_rows FROM ELGG_DB.elgg_entities; SELECT COUNT(*) AS elgg_metadata_rows FROM ELGG_DB.elgg_metadata;' \
  > "${RESULT_DIR}/application-row-validation.txt"
ENTITY_COUNT="$(docker exec "${CONTAINER}" mysql -uroot -proot -Nse \
  'SELECT COUNT(*) FROM ELGG_DB.elgg_entities;')"
METADATA_COUNT="$(docker exec "${CONTAINER}" mysql -uroot -proot -Nse \
  'SELECT COUNT(*) FROM ELGG_DB.elgg_metadata;')"
if [[ ! "${ENTITY_COUNT}" =~ ^[0-9]+$ || "${ENTITY_COUNT}" -lt 1 ||
      ! "${METADATA_COUNT}" =~ ^[0-9]+$ || "${METADATA_COUNT}" -lt 1 ]]; then
  printf 'ELGG application row validation failed: entities=%s metadata=%s\n' \
    "${ENTITY_COUNT}" "${METADATA_COUNT}" >&2
  exit 3
fi
printf 'table_count=%s\nelgg_entities_rows=%s\nelgg_metadata_rows=%s\n' \
  "${TABLE_COUNT}" "${ENTITY_COUNT}" "${METADATA_COUNT}" \
  >> "${RESULT_DIR}/database-validation.txt"

docker exec "${CONTAINER}" mysqladmin -uroot -proot shutdown > "${RESULT_DIR}/database-stop.log" 2>&1
stopped=0
for _ in $(seq 1 30); do
  if ! docker exec "${CONTAINER}" mysqladmin -uroot -proot ping --silent >/dev/null 2>&1; then
    stopped=1
    break
  fi
  sleep 1
done
if [[ "${stopped}" != 1 ]]; then
  printf 'MySQL did not stop cleanly; refusing to commit live datadir\n' >&2
  exit 3
fi
printf 'mysql_stopped_cleanly=true\n' >> "${RESULT_DIR}/database-stop.log"
docker commit "${CONTAINER}" "${SEEDED_IMAGE}" > "${RESULT_DIR}/seeded-image-id.txt"
docker image inspect "${SEEDED_IMAGE}" --format '{{.Id}} {{json .RepoDigests}}' \
  > "${RESULT_DIR}/seeded-image-inspect.txt"
printf 'status=seeded_and_validated\nimage=%s\ntable_count=%s\nelgg_entities_rows=%s\nelgg_metadata_rows=%s\n' \
  "${SEEDED_IMAGE}" "${TABLE_COUNT}" "${ENTITY_COUNT}" "${METADATA_COUNT}" \
  > "${RESULT_DIR}/status.txt"
printf 'Seeded %s with %s ELGG_DB tables and %s entities. Evidence: %s\n' \
  "${SEEDED_IMAGE}" "${TABLE_COUNT}" "${ENTITY_COUNT}" "${RESULT_DIR}"
