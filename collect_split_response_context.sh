#!/bin/bash
set -u

ROOT="${1:-/global/workdir/jkim068/orbit}"
OUT="${2:-/global/workdir/jkim068/orbit_split_response_context.txt}"

cd "$ROOT" || {
    echo "ERROR: Cannot enter $ROOT"
    exit 1
}

: > "$OUT"

section() {
    {
        echo
        echo "================================================================"
        echo "$1"
        echo "================================================================"
    } >> "$OUT"
}

section "REPOSITORY"
{
    echo "root=$ROOT"
    echo "date=$(date -Is)"
    echo
    pwd
    git status --short 2>&1 || true
    git log -1 --oneline 2>&1 || true
} >> "$OUT"

section "PYPROJECT"
sed -n '1,260p' pyproject.toml >> "$OUT" 2>&1 || true

section "PACKAGE FILE TREE"
find src/orbit tests \
    -type f \
    \( -name '*.py' -o -name '*.toml' \) \
    | sort >> "$OUT" 2>&1 || true

section "CLI HELP"
{
    echo '$ orbit --version'
    orbit --version 2>&1 || true

    echo
    echo '$ orbit --help'
    orbit --help 2>&1 || true

    echo
    echo '$ orbit path-response --help'
    orbit path-response --help 2>&1 || true
} >> "$OUT"

section "RESPONSE REFERENCES"
grep -RInE \
    --include='*.py' \
    --include='*.toml' \
    --include='*.md' \
    'path-response|path_response|path response|polarization_calculations|ph_calculations|run_response|run_ph|run_polarization|qe\.ph|qe\.polarization' \
    src tests pyproject.toml README.md \
    >> "$OUT" 2>&1 || true

FILES="$(
    {
        printf '%s\n' \
            pyproject.toml \
            README.md \
            src/orbit/__init__.py \
            src/orbit/__main__.py \
            src/orbit/cli.py \
            src/orbit/config.py \
            src/orbit/project.py

        grep -RIlE \
            --include='*.py' \
            'path-response|path_response|polarization_calculations|ph_calculations|run_response|run_ph_array|run_polarization_array|qe\.ph|qe\.polarization' \
            src/orbit tests 2>/dev/null || true
    } | awk 'NF && !seen[$0]++'
)"

for FILE in $FILES; do
    [[ -f "$FILE" ]] || continue

    section "FILE: $FILE"
    sed -n '1,3000p' "$FILE" >> "$OUT"
done

PROJECT="/global/workdir/jkim068/BaBiO3"

if [[ -f "$PROJECT/orbit.toml" ]]; then
    section "BABI03 GAPFLOW CONFIG"
    sed -n '1,240p' "$PROJECT/orbit.toml" >> "$OUT"
fi

for RESPONSE in \
    "$PROJECT/calculations/helpers/Bi1/20260815T205203Z-70574432/iteration_010/response" \
    "$PROJECT/calculations/helpers/Bi2/20260817T054011Z-a44a1631/iteration_006/response"
do
    [[ -d "$RESPONSE" ]] || continue

    section "GENERATED RESPONSE DIRECTORY: $RESPONSE"
    find "$RESPONSE" -maxdepth 2 -type f -printf '%P\n' \
        | sort >> "$OUT"

    for NAME in \
        preparation.json \
        ph_calculations.txt \
        polarization_calculations.txt \
        all_calculations.txt \
        run_ph_array.sh \
        run_polarization_array.sh \
        run_response_array.sh \
        submit_ph.sh \
        submit_polarization.sh \
        submit_all.sh
    do
        FILE="$RESPONSE/$NAME"
        [[ -f "$FILE" ]] || continue

        section "GENERATED FILE: $FILE"
        sed -n '1,1200p' "$FILE" >> "$OUT"
    done
done

section "END"
echo "Context collection completed." >> "$OUT"

echo "Wrote:"
echo "  $OUT"
wc -l -c "$OUT"
