#!/bin/sh
# Optional real network delay for the High-latency client (PRD: tc netem).
# Needs cap_add NET_ADMIN. Set NETEM_DELAY (e.g. "800ms") to enable.
if [ -n "$NETEM_DELAY" ]; then
  tc qdisc add dev eth0 root netem delay "$NETEM_DELAY" "${NETEM_JITTER:-100ms}" \
    || echo "warning: could not apply tc netem (missing NET_ADMIN?)"
fi
exec "$@"
