#!/bin/sh
# Serve map.html on :8777, replacing any server already holding the port.
cd "$(dirname "$0")"
pids=$(lsof -ti:8777) && kill $pids
exec python3 -m http.server 8777
