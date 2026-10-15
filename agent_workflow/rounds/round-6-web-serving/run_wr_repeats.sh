#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
URL="http://127.0.0.1:18080/"
PORT=18080
BASE_SEED=20261012
CONCURRENCY_LEVELS=(20 100 200 500 1000)
RUNS=5
TEST_DURATION=20s

mkdir -p "$ROOT_DIR/www"
if [[ -f "$ROOT_DIR/nginx.pid" ]] && kill -0 "$(<"$ROOT_DIR/nginx.pid")" 2>/dev/null; then
    printf 'Refusing to start: an experiment Nginx PID already exists.\n' >&2
    exit 1
fi

cleanup() {
    if [[ -f "$ROOT_DIR/nginx.pid" ]]; then
        local nginx_pid
        nginx_pid="$(<"$ROOT_DIR/nginx.pid")"
        if kill -0 "$nginx_pid" 2>/dev/null; then
            kill -QUIT "$nginx_pid"
            for _ in {1..50}; do
                kill -0 "$nginx_pid" 2>/dev/null || break
                sleep 0.1
            done
        fi
        rm -f "$ROOT_DIR/nginx.pid"
    fi
}
trap cleanup EXIT INT TERM

{
    printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'host=%s\n' "$(hostname)"
    printf 'kernel='; uname -srmo
    printf 'nginx='; nginx -v 2>&1
    printf 'wrk_package_version='; dpkg-query -W -f='${Version}\n' wrk
    printf 'port=%s\nurl=%s\n' "$PORT" "$URL"
    printf 'rounds=%s\nduration_per_profile=%s\nbase_seed=%s\n' "$RUNS" "$TEST_DURATION" "$BASE_SEED"
    printf '\n--- CPU ---\n'; lscpu
    printf '\n--- Memory / Load ---\n'; free -h; uptime
} > "$ROOT_DIR/host_context.txt" 2>&1

nginx -t -c "$ROOT_DIR/nginx.conf" > "$ROOT_DIR/nginx-config-test.log" 2>&1
nginx -c "$ROOT_DIR/nginx.conf"
for _ in {1..50}; do
    if curl --fail --silent --show-error --max-time 1 "$URL" >/dev/null; then
        break
    fi
    sleep 0.1
done
curl --fail --silent --show-error --max-time 5 --output /dev/null \
    --write-out 'http_status=%{http_code} bytes=%{size_download} time_total=%{time_total}\n' \
    "$URL" > "$ROOT_DIR/http-smoke-check.txt"

for ((run = 1; run <= RUNS; run++)); do
    seed=$((BASE_SEED + run))
    mapfile -t levels < <(python -c 'import random,sys; x=[20,100,200,500,1000]; random.Random(int(sys.argv[1])).shuffle(x); print("\n".join(map(str,x)))' "$seed")
    printf 'run=%s seed=%s order=%s\n' "$run" "$seed" "${levels[*]}" >> "$ROOT_DIR/run-order.txt"

    for connections in "${levels[@]}"; do
        log="$ROOT_DIR/run-${run}-connections-${connections}.log"
        {
            printf 'run=%s connections=%s threads=4 duration=%s seed=%s\n' "$run" "$connections" "$TEST_DURATION" "$seed"
            printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
            printf 'command=wrk -t 4 -c %s -d %s --latency --timeout 5s %s\n' "$connections" "$TEST_DURATION" "$URL"
            /usr/bin/time -f 'elapsed_real_seconds=%e' \
                wrk -t 4 -c "$connections" -d "$TEST_DURATION" --latency --timeout 5s "$URL"
            printf 'ended_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        } > "$log" 2>&1
    done
done

printf 'finished_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$ROOT_DIR/host_context.txt"
