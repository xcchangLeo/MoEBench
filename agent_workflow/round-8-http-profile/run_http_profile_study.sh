#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULT_DIR="${ROOT_DIR}/results"
PORT=18081
URL="http://127.0.0.1:${PORT}/"
LEVELS=(20 40 60 80 100)
SESSIONS=4
FULL_SECONDS=20
PROBE_SECONDS=4
ACTIVE_CONF=""
ACTIVE_PIDFILE=""
ACTIVE_NGINX_PID=""
mkdir -p "${RESULT_DIR}"

stop_active_nginx() {
    if [[ -n "${ACTIVE_CONF}" ]]; then
        local pid=""
        if [[ -f "${ACTIVE_PIDFILE}" ]]; then pid="$(<"${ACTIVE_PIDFILE}")"; fi
        nginx -s quit -c "${ACTIVE_CONF}" >/dev/null 2>&1 || true
        for _ in {1..50}; do
            [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null || break
            sleep 0.1
        done
        for _ in {1..50}; do
            ss -ltnH "sport = :${PORT}" | rg -q . || break
            sleep 0.1
        done
    fi
    ACTIVE_CONF=""
    ACTIVE_PIDFILE=""
    ACTIVE_NGINX_PID=""
}
cleanup() {
    stop_active_nginx
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if ss -ltnH "sport = :${PORT}" | rg -q .; then
    printf 'Refusing to start: loopback port %s is already in use.\n' "${PORT}" >&2
    exit 1
fi

: > "${RESULT_DIR}/runs.jsonl"
: > "${RESULT_DIR}/run-order.txt"
: > "${RESULT_DIR}/failures.txt"
{
    printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'host_label=local_host_redacted_in_manuscript\n'
    printf 'kernel='; uname -srmo
    printf 'nginx='; nginx -v 2>&1
    printf 'wrk_package_version='; dpkg-query -W -f='${Version}\n' wrk
    printf 'url=%s\nport=%s\n' "${URL}" "${PORT}"
    printf 'sessions=%s\nlevels=%s\nfull_seconds=%s\nprobe_seconds=%s\n' \
        "${SESSIONS}" "${LEVELS[*]}" "${FULL_SECONDS}" "${PROBE_SECONDS}"
    printf '\n--- CPU ---\n'; lscpu
    printf '\n--- Memory / Load ---\n'; free -h; uptime
} > "${RESULT_DIR}/host-context.txt" 2>&1

run_measurement() {
    local config="$1" workers="$2" session="$3" connections="$4" phase="$5"
    local duration log start end elapsed status timeout_seconds
    duration="${FULL_SECONDS}s"
    timeout_seconds=35
    if [[ "${phase}" == probe ]]; then duration="${PROBE_SECONDS}s"; timeout_seconds=12; fi
    log="${RESULT_DIR}/${config}-session${session}-connections${connections}-${phase}.log"
    start="$(date +%s%N)"
    {
        printf 'config=%s workers=%s session=%s connections=%s threads=4 phase=%s duration=%s\n' \
            "${config}" "${workers}" "${session}" "${connections}" "${phase}" "${duration}"
        printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        printf 'command=timeout %ss wrk -t 4 -c %s -d %s --latency --timeout 5s %s\n' \
            "${timeout_seconds}" "${connections}" "${duration}" "${URL}"
        set +e
        /usr/bin/time -f 'elapsed_real_seconds=%e' \
            timeout --signal=TERM "${timeout_seconds}s" \
            wrk -t 4 -c "${connections}" -d "${duration}" --latency --timeout 5s "${URL}"
        status=$?
        set -e
        printf 'exit_code=%s\nended_utc=%s\n' "${status}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    } > "${log}" 2>&1
    end="$(date +%s%N)"
    elapsed="$(awk -v a="${start}" -v b="${end}" 'BEGIN { printf "%.6f", (b-a)/1000000000 }')"
    printf '{"config":"%s","workers":%s,"session":%s,"connections":%s,"phase":"%s","nginx_pid":%s,"start_epoch_ns":%s,"end_epoch_ns":%s,"elapsed_wall_seconds":%s,"exit_code":%s,"raw_log":"%s"}\n' \
        "${config}" "${workers}" "${session}" "${connections}" "${phase}" \
        "${ACTIVE_NGINX_PID}" "${start}" "${end}" "${elapsed}" "${status}" "${log}" >> "${RESULT_DIR}/runs.jsonl"
    if [[ "${status}" != 0 ]]; then
        printf 'measurement_failed config=%s session=%s load=%s phase=%s exit=%s\n' \
            "${config}" "${session}" "${connections}" "${phase}" "${status}" \
            >> "${RESULT_DIR}/failures.txt"
        return 1
    fi
}

start_server() {
    local config="$1" workers="$2" session="$3" phase="$4"
    local conf="${RESULT_DIR}/${config}-session${session}-${phase}.conf"
    local pidfile="${RESULT_DIR}/${config}-session${session}-${phase}.pid"
    local errorlog="${RESULT_DIR}/${config}-session${session}-${phase}-nginx-error.log"
    if ss -ltnH "sport = :${PORT}" | rg -q .; then
        printf 'Refusing to start phase: loopback port %s is still in use.\n' "${PORT}" >&2
        return 1
    fi
    sed -e "s|@PID_FILE@|${pidfile}|g" \
        -e "s|@ERROR_LOG@|${errorlog}|g" \
        -e "s|@WORKERS@|${workers}|g" \
        -e "s|@DOCROOT@|${ROOT_DIR}/www|g" \
        "${ROOT_DIR}/nginx.template.conf" > "${conf}"
    nginx -t -c "${conf}" > "${RESULT_DIR}/${config}-session${session}-${phase}-nginx-test.log" 2>&1
    ACTIVE_CONF="${conf}"
    ACTIVE_PIDFILE="${pidfile}"
    nginx -c "${conf}"
    for _ in {1..50}; do
        if curl --fail --silent --show-error --max-time 1 "${URL}" >/dev/null; then
            ACTIVE_NGINX_PID="$(<"${pidfile}")"
            return 0
        fi
        sleep 0.1
    done
    printf 'http_smoke_failed config=%s session=%s\n' "${config}" "${session}" >&2
    return 1
}

for session in $(seq 1 "${SESSIONS}"); do
    for spec in $(printf 'workers1:1\nworkers4:4\n' | shuf); do
        config="${spec%%:*}"
        workers="${spec#*:}"
        probe_levels="$(shuf -e "${LEVELS[@]}" | paste -sd, -)"
        full_levels="$(shuf -e "${LEVELS[@]}" | paste -sd, -)"
        printf 'session=%s config=%s workers=%s phase=probe order=%s\n' \
            "${session}" "${config}" "${workers}" "${probe_levels}" >> "${RESULT_DIR}/run-order.txt"
        start_server "${config}" "${workers}" "${session}" probe
        for connections in ${probe_levels//,/ }; do
            run_measurement "${config}" "${workers}" "${session}" "${connections}" probe
        done
        stop_active_nginx

        printf 'session=%s config=%s workers=%s phase=full order=%s\n' \
            "${session}" "${config}" "${workers}" "${full_levels}" >> "${RESULT_DIR}/run-order.txt"
        start_server "${config}" "${workers}" "${session}" full
        for connections in ${full_levels//,/ }; do
            run_measurement "${config}" "${workers}" "${session}" "${connections}" full
        done
        stop_active_nginx
    done
done

printf 'finished_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${RESULT_DIR}/host-context.txt"
