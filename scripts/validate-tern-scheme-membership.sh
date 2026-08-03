#!/bin/sh
set -eu

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 GENERATED_NQUADS_OR_SPARQL_ENDPOINT" >&2
    exit 2
fi

KURRA_BIN="${KURRA_BIN:-/Users/leskneebone/.local/bin/kurra}"
QUERY_PATH="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)/validation/tern-scheme-membership.rq"
REPO_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
if [ -x "$REPO_ROOT/.venv/bin/python" ]; then
    DEFAULT_PYTHON="$REPO_ROOT/.venv/bin/python"
else
    DEFAULT_PYTHON="python3"
fi
TARGET="$1"

case "$TARGET" in
    http://*|https://*)
        QUERY_CONTENT="$(sed -n '1,$p' "$QUERY_PATH")"
        "$KURRA_BIN" db sparql "$TARGET" "$QUERY_CONTENT"
        ;;
    *)
        "${PYTHON_BIN:-$DEFAULT_PYTHON}" "$REPO_ROOT/scripts/validate_tern_scheme_membership.py" "$TARGET"
        ;;
esac
