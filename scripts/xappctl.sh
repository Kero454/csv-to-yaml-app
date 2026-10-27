#!/usr/bin/env bash
#
# xappctl.sh - terminal CLI for the Near-RT RIC xApp lifecycle.
#
# Every operation is a plain `curl` against the O-RAN App Manager (appmgr) or
# the xApp onboarder REST API, following the same pattern as the upstream
# tools:
#   - ric-plt-appmgr/scripts/appmgrcli   (deploy / undeploy / status via curl)
#   - oran-ric/dep/demos/bronze/xapp-hw.sh (onboard -> list -> deploy -> poll pods)
#
# Usage:
#   xappctl.sh [--mode portforward|incluster|direct] <command> [args]
#
# Commands:
#   onboard  <config-file.json> <schema.json>   push xApp chart to chartmuseum
#   charts                                       list onboarded (deployable) xApps
#   deploy   <name> [--version V] [--namespace NS] [--override JSON]
#   status   [name]                              query all deployed xApps, or one
#   undeploy <name>                              delete a deployed xApp
#   health                                       appmgr health (alive + ready)
#   subscribe <targetUrl> [eventType] [maxRetries] [retryTimer]
#   subscriptions                                list subscriptions
#   wait     <name> [timeoutSec]                 wait for the xApp pod to be Running
#   testpod  start|stop|exec <curl args...>      manage the in-cluster curl test pod
#   help
#
# Connection modes (env XAPPCTL_MODE or --mode):
#   portforward  (default) kubectl port-forward the ricplt service to localhost
#   incluster    run curl from a test pod inside namespace ricplt, hitting the
#                ClusterIP service DNS directly (service-ricplt-appmgr-http:8080)
#   direct       use APPMGR_URL / ONBOARDER_URL as given (e.g. Kong ingress
#                http://<host>:32080/appmgr and http://<host>:32080/onboard)
#
# Environment overrides:
#   RIC_NAMESPACE        platform namespace            (default ricplt)
#   XAPP_NAMESPACE       xApp namespace                (default ricxapp)
#   APPMGR_SERVICE       appmgr service name           (default service-ricplt-appmgr-http)
#   APPMGR_PORT          appmgr service port           (default 8080)
#   ONBOARDER_SERVICE    onboarder service name        (default service-ricplt-xapp-onboarder-http)
#   ONBOARDER_PORT       onboarder service port        (default 8888)
#   APPMGR_URL           direct-mode appmgr base URL
#   ONBOARDER_URL        direct-mode onboarder base URL
#   TESTPOD_NAME         in-cluster curl pod name      (default xapp-curl)
#   TESTPOD_IMAGE        in-cluster curl pod image     (default curlimages/curl:8.5.0)
#   KUBECTL              kubectl command               (default kubectl)
#
# Output: response body on stdout, "HTTP <code>" on stderr.
# Exit code: 0 on 2xx, 1 otherwise.

set -o pipefail

MODE="${XAPPCTL_MODE:-portforward}"
RIC_NAMESPACE="${RIC_NAMESPACE:-ricplt}"
XAPP_NAMESPACE="${XAPP_NAMESPACE:-ricxapp}"
APPMGR_SERVICE="${APPMGR_SERVICE:-service-ricplt-appmgr-http}"
APPMGR_PORT="${APPMGR_PORT:-8080}"
ONBOARDER_SERVICE="${ONBOARDER_SERVICE:-service-ricplt-xapp-onboarder-http}"
ONBOARDER_PORT="${ONBOARDER_PORT:-8888}"
TESTPOD_NAME="${TESTPOD_NAME:-xapp-curl}"
TESTPOD_IMAGE="${TESTPOD_IMAGE:-curlimages/curl:8.5.0}"
KUBECTL="${KUBECTL:-kubectl}"
VERBOSE="${XAPPCTL_VERBOSE:-0}"

APPMGR_BASE="/ric/v1"
ONBOARD_BASE="/api/v1"

PF_PID=""
HTTP_CODE=""

log()  { echo "[xappctl] $*" >&2; }
vlog() { [ "$VERBOSE" = "1" ] && echo "[xappctl] $*" >&2; return 0; }
die()  { log "ERROR: $*"; exit 1; }

usage() { sed -n '2,50p' "$0" | sed 's/^# \{0,1\}//'; exit 0; }

cleanup() {
  if [ -n "$PF_PID" ]; then
    kill "$PF_PID" 2>/dev/null
    wait "$PF_PID" 2>/dev/null
    PF_PID=""
  fi
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

# Pick a free local port in 21000-21999 for port-forward mode.
free_port() {
  local p
  for _ in $(seq 1 20); do
    p=$((21000 + RANDOM % 1000))
    if ! (echo > /dev/tcp/127.0.0.1/$p) 2>/dev/null; then echo "$p"; return; fi
  done
  echo 21999
}

# start_portforward <service> <remotePort>  -> sets LOCAL_URL
start_portforward() {
  local svc="$1" rport="$2" lport
  lport=$(free_port)
  vlog "$KUBECTL port-forward -n $RIC_NAMESPACE svc/$svc $lport:$rport"
  $KUBECTL port-forward -n "$RIC_NAMESPACE" "svc/$svc" "$lport:$rport" >/dev/null 2>&1 &
  PF_PID=$!
  # wait until the tunnel accepts connections (max ~10s)
  for _ in $(seq 1 20); do
    if (echo > /dev/tcp/127.0.0.1/$lport) 2>/dev/null; then break; fi
    sleep 0.5
  done
  LOCAL_URL="http://127.0.0.1:$lport"
}

testpod_running() {
  [ "$($KUBECTL get pod -n "$RIC_NAMESPACE" "$TESTPOD_NAME" -o jsonpath='{.status.phase}' 2>/dev/null)" = "Running" ]
}

testpod_start() {
  if testpod_running; then
    log "test pod $TESTPOD_NAME already running in $RIC_NAMESPACE"
    return 0
  fi
  log "starting test pod $TESTPOD_NAME ($TESTPOD_IMAGE) in namespace $RIC_NAMESPACE"
  $KUBECTL run "$TESTPOD_NAME" -n "$RIC_NAMESPACE" --image="$TESTPOD_IMAGE" \
    --restart=Never --command -- sleep 86400 >/dev/null
  $KUBECTL wait -n "$RIC_NAMESPACE" --for=condition=Ready "pod/$TESTPOD_NAME" --timeout=120s >/dev/null \
    || die "test pod did not become Ready"
  log "test pod ready"
}

testpod_stop() {
  $KUBECTL delete pod -n "$RIC_NAMESPACE" "$TESTPOD_NAME" --ignore-not-found >/dev/null
  log "test pod $TESTPOD_NAME removed"
}

# rest <target: appmgr|onboarder> <METHOD> <path> [data]
# Prints body to stdout, sets HTTP_CODE. Mirrors appmgrcli's rest() helper.
rest() {
  local target="$1" method="$2" path="$3" data="${4:-}"
  local svc port base_url out body

  case "$target" in
    appmgr)    svc="$APPMGR_SERVICE";    port="$APPMGR_PORT";    base_url="${APPMGR_URL:-}" ;;
    onboarder) svc="$ONBOARDER_SERVICE"; port="$ONBOARDER_PORT"; base_url="${ONBOARDER_URL:-}" ;;
    *) die "unknown target $target" ;;
  esac

  local -a curl_args=(--silent --show-error --connect-timeout 20 --max-time 300
                      --header "Content-Type: application/json" -X "$method"
                      -w '\n##%{http_code}')

  case "$MODE" in
    incluster)
      testpod_running || testpod_start
      local url="http://${svc}:${port}${path}"
      vlog "$KUBECTL exec -n $RIC_NAMESPACE $TESTPOD_NAME -- curl -X $method $url"
      if [ -n "$data" ]; then
        out=$(printf '%s' "$data" | $KUBECTL exec -i -n "$RIC_NAMESPACE" "$TESTPOD_NAME" -- \
              curl "${curl_args[@]}" --data-binary @- "$url")
      else
        out=$($KUBECTL exec -n "$RIC_NAMESPACE" "$TESTPOD_NAME" -- curl "${curl_args[@]}" "$url")
      fi
      ;;
    direct)
      [ -n "$base_url" ] || die "direct mode needs APPMGR_URL / ONBOARDER_URL"
      local url="${base_url}${path}"
      vlog "curl -X $method $url"
      if [ -n "$data" ]; then
        out=$(curl "${curl_args[@]}" --data "$data" "$url")
      else
        out=$(curl "${curl_args[@]}" "$url")
      fi
      ;;
    portforward|*)
      start_portforward "$svc" "$port"
      local url="${LOCAL_URL}${path}"
      vlog "curl -X $method $url"
      if [ -n "$data" ]; then
        out=$(curl "${curl_args[@]}" --data "$data" "$url")
      else
        out=$(curl "${curl_args[@]}" "$url")
      fi
      cleanup
      ;;
  esac

  HTTP_CODE=$(printf '%s' "$out" | sed -n 's/^##\([0-9]\{3\}\)$/\1/p' | tail -n1)
  body=$(printf '%s' "$out" | sed '/^##[0-9]\{3\}$/d')
  [ -n "$body" ] && printf '%s\n' "$body"
  echo "HTTP ${HTTP_CODE:-000}" >&2
  case "$HTTP_CODE" in 2*) return 0 ;; *) return 1 ;; esac
}

# ---------------------------------------------------------------------------
# JSON helpers (python3 preferred, jq fallback)
# ---------------------------------------------------------------------------

build_onboard_payload() {
  local cfg="$1" schema="$2"
  [ -f "$cfg" ]    || die "descriptor not found: $cfg"
  [ -f "$schema" ] || die "schema not found: $schema"
  if command -v python3 >/dev/null 2>&1; then
    python3 - "$cfg" "$schema" <<'PY'
import json, sys
cfg, schema = sys.argv[1], sys.argv[2]
print(json.dumps({"config-file.json": json.load(open(cfg)),
                  "controls-schema.json": json.load(open(schema))}))
PY
  elif command -v jq >/dev/null 2>&1; then
    jq -n --slurpfile c "$cfg" --slurpfile s "$schema" \
      '{"config-file.json": $c[0], "controls-schema.json": $s[0]}'
  else
    die "need python3 or jq to build the onboard payload"
  fi
}

# ---------------------------------------------------------------------------
# Commands (same vocabulary as appmgrcli)
# ---------------------------------------------------------------------------

cmd_onboard() {
  [ $# -ge 2 ] || die "usage: onboard <config-file.json> <schema.json>"
  local payload
  payload=$(build_onboard_payload "$1" "$2") || exit 1
  log "POST ${ONBOARD_BASE}/onboard  ($(basename "$1") + $(basename "$2"))"
  rest onboarder POST "${ONBOARD_BASE}/onboard" "$payload"
  local rc=$?
  # 400 "already exists" is acceptable (idempotent re-onboard)
  if [ $rc -ne 0 ] && [ "$HTTP_CODE" = "400" ]; then
    log "chart may already exist in chartmuseum (400) - continuing"
    return 0
  fi
  return $rc
}

cmd_charts() {
  log "GET ${APPMGR_BASE}/xapps/list  (deployable charts in helm repo)"
  rest appmgr GET "${APPMGR_BASE}/xapps/list"
}

cmd_deploy() {
  local name="" version="" ns="$XAPP_NAMESPACE" override='{"image_pull_policy":"Never"}'
  while [ $# -gt 0 ]; do
    case "$1" in
      --version)   version="$2"; shift 2 ;;
      --namespace) ns="$2"; shift 2 ;;
      --override)  override="$2"; shift 2 ;;
      *) [ -z "$name" ] && name="$1" || die "unexpected argument $1"; shift ;;
    esac
  done
  [ -n "$name" ] || die "usage: deploy <name> [--version V] [--namespace NS] [--override JSON]"
  local data
  data=$(printf '{"xappName":"%s","helmVersion":"%s","releaseName":"%s","namespace":"%s","overrideFile":%s}' \
         "$name" "$version" "$name" "$ns" "$override")
  log "POST ${APPMGR_BASE}/xapps  $data"
  rest appmgr POST "${APPMGR_BASE}/xapps" "$data"
}

cmd_status() {
  if [ -n "${1:-}" ]; then
    log "GET ${APPMGR_BASE}/xapps/$1"
    rest appmgr GET "${APPMGR_BASE}/xapps/$1"
  else
    log "GET ${APPMGR_BASE}/xapps  (all deployed xApps)"
    rest appmgr GET "${APPMGR_BASE}/xapps"
  fi
}

cmd_undeploy() {
  [ -n "${1:-}" ] || die "usage: undeploy <name>"
  log "DELETE ${APPMGR_BASE}/xapps/$1"
  rest appmgr DELETE "${APPMGR_BASE}/xapps/$1"
}

cmd_health() {
  log "GET ${APPMGR_BASE}/health/alive"
  rest appmgr GET "${APPMGR_BASE}/health/alive"
  log "GET ${APPMGR_BASE}/health/ready"
  rest appmgr GET "${APPMGR_BASE}/health/ready"
}

cmd_subscribe() {
  [ -n "${1:-}" ] || die "usage: subscribe <targetUrl> [eventType] [maxRetries] [retryTimer]"
  local data
  data=$(printf '{"Data":{"TargetUrl":"%s","EventType":"%s","MaxRetries":%s,"RetryTimer":%s}}' \
         "$1" "${2:-all}" "${3:-5}" "${4:-10}")
  log "POST ${APPMGR_BASE}/subscriptions  $data"
  rest appmgr POST "${APPMGR_BASE}/subscriptions" "$data"
}

cmd_subscriptions() {
  log "GET ${APPMGR_BASE}/subscriptions"
  rest appmgr GET "${APPMGR_BASE}/subscriptions"
}

# Poll like xapp-hw.sh: wait until a pod for the release is Running.
cmd_wait() {
  [ -n "${1:-}" ] || die "usage: wait <name> [timeoutSec]"
  local name="$1" timeout="${2:-180}" waited=0 pod
  log "waiting for pod of $name in $XAPP_NAMESPACE (timeout ${timeout}s)"
  while [ $waited -lt $timeout ]; do
    pod=$($KUBECTL get pods -n "$XAPP_NAMESPACE" -l "release=$name" \
          -o jsonpath='{range .items[*]}{.metadata.name} {.status.phase}{"\n"}{end}' 2>/dev/null | grep Running | head -n1)
    if [ -n "$pod" ]; then
      echo "$pod"
      $KUBECTL get pods -n "$XAPP_NAMESPACE" -l "release=$name" >&2
      return 0
    fi
    sleep 5; waited=$((waited + 5)); echo -n "." >&2
  done
  echo >&2
  $KUBECTL get pods -n "$XAPP_NAMESPACE" -l "release=$name" >&2
  die "pod for $name not Running after ${timeout}s"
}

cmd_testpod() {
  case "${1:-}" in
    start) testpod_start ;;
    stop)  testpod_stop ;;
    exec)  shift; testpod_running || testpod_start
           $KUBECTL exec -i -n "$RIC_NAMESPACE" "$TESTPOD_NAME" -- "$@" ;;
    *) die "usage: testpod start|stop|exec <cmd...>" ;;
  esac
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

while [ $# -gt 0 ]; do
  case "$1" in
    --mode)  MODE="$2"; shift 2 ;;
    -v|--verbose) VERBOSE=1; shift ;;
    -h|--help|help) usage ;;
    *) break ;;
  esac
done

[ $# -ge 1 ] || { log "missing command"; usage; }
cmd="$1"; shift

case "$cmd" in
  onboard)            cmd_onboard "$@" ;;
  charts|list)        cmd_charts ;;
  deploy|dep)         cmd_deploy "$@" ;;
  status|stat|query)  cmd_status "$@" ;;
  undeploy|undep|delete) cmd_undeploy "$@" ;;
  health|heal)        cmd_health ;;
  subscribe|sub)      cmd_subscribe "$@" ;;
  subscriptions|subs) cmd_subscriptions ;;
  wait)               cmd_wait "$@" ;;
  testpod)            cmd_testpod "$@" ;;
  *) die "unknown command '$cmd' (run: $0 help)" ;;
esac
