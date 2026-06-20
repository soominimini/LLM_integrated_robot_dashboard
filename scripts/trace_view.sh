#!/usr/bin/env bash
# Live step-by-step viewer for the running web server.
# Follows src/logs/trace.log and renders each "work" as an indented block:
#   a WORK header, its internal steps indented beneath, then the DONE/timing line.
# Usage:  ./scripts/trace_view.sh            # follow everything
#         ./scripts/trace_view.sh quiz       # only works whose lines match 'quiz'
TRACE="/home/qtrobot/tutorials-master/demos/version_1_llm_gemini/src/logs/trace.log"
FILTER="${1:-}"

tail -n 0 -F "$TRACE" 2>/dev/null | awk -v f="$FILTER" '
  function strip(s){ gsub(/^[0-9:]+ \[[0-9a-f-]+\] /,"",s); return s }
  {
    line=$0
    if (f != "" && index(line, f)==0 && line !~ /WORK|DONE/) next
    if (index(line,"WORK")) { printf "\n\033[1;36m%s\033[0m\n", line }
    else if (index(line,"DONE")) { printf "\033[1;32m%s\033[0m\n", line }
    else { printf "      \342\224\224 %s\n", strip(line) }
    fflush()
  }'
