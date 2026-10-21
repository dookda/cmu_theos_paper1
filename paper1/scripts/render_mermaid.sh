#!/bin/bash
# Render a mermaid file to a high-resolution PNG with mermaid-cli, using the local Chrome.
#   bash paper1/scripts/render_mermaid.sh paper1/analysis/figure/fig_workflow.mmd paper1/analysis/figure/fig_workflow.png
set -euo pipefail
IN=$1; OUT=$2; SCALE=${3:-5}
CFG=$(mktemp -t pptr).json
echo '{"executablePath": "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "args": ["--no-sandbox"]}' > "$CFG"
PUPPETEER_SKIP_DOWNLOAD=1 npx -y -p @mermaid-js/mermaid-cli@11 mmdc -i "$IN" -o "$OUT" \
  -p "$CFG" -b white -s "$SCALE" -w 1100
rm -f "$CFG"
