#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${1:-$PWD}"
PROJECT_DIR="${2:-$(dirname "$REPO_DIR")/STO2x2x1}"
OUTPUT_FILE="${3:-$REPO_DIR/polarization_plot_context.txt}"

REPO_DIR="$(cd "$REPO_DIR" && pwd)"
PROJECT_DIR="$(cd "$PROJECT_DIR" && pwd)"

section() {
    {
        echo
        echo "================================================================"
        echo "$1"
        echo "================================================================"
    } >> "$OUTPUT_FILE"
}

append_file() {
    local file="$1"
    local max_lines="${2:-2400}"

    if [[ -f "$file" ]]; then
        {
            echo
            echo "----- FILE: $file -----"
            sed -n "1,${max_lines}p" "$file"
            echo "----- END FILE: $file -----"
        } >> "$OUTPUT_FILE"
    fi
}

: > "$OUTPUT_FILE"

{
    echo "Polarization plot patch context"
    echo "Generated: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "Repository: $REPO_DIR"
    echo "Test project: $PROJECT_DIR"
    echo "Python: $(python --version 2>&1 || true)"
    echo "ORBIT executable: $(command -v orbit || true)"
} >> "$OUTPUT_FILE"

section "GIT AND PACKAGE STATE"

(
    cd "$REPO_DIR"
    git status --short 2>&1 || true
    git log -1 --oneline 2>&1 || true
) >> "$OUTPUT_FILE"

append_file "$REPO_DIR/pyproject.toml"
append_file "$REPO_DIR/src/orbit/__init__.py"

section "REPOSITORY FILE TREE"

(
    cd "$REPO_DIR"
    find src tests -maxdepth 5 -type f \
        \( -name '*.py' -o -name '*.toml' -o -name '*.json' \) \
        -print 2>/dev/null | sort
) >> "$OUTPUT_FILE"

section "POLARIZATION-RELATED SOURCE MATCHES"

SOURCE_MATCHES="$(
    cd "$REPO_DIR"
    grep -RIlE \
        'polarization|Berry|born.?charge|effective.?charge|Z[_*]?star|branch|oxidation|polarization.?quantum|ph\.x' \
        src tests . \
        --include='*.py' \
        --include='*.toml' \
        --include='*.md' \
        --exclude='polarization_plot_context.txt' \
        --exclude-dir='.git' \
        --exclude-dir='.pytest_cache' \
        --exclude-dir='__pycache__' \
        2>/dev/null |
    sort -u
)"

printf '%s\n' "$SOURCE_MATCHES" >> "$OUTPUT_FILE"

section "FULL RELEVANT SOURCE FILES"

while IFS= read -r relative_file; do
    [[ -n "$relative_file" ]] || continue

    if [[ "$relative_file" = /* ]]; then
        append_file "$relative_file" 4000
    else
        append_file "$REPO_DIR/$relative_file" 4000
    fi
done <<< "$SOURCE_MATCHES"

section "CLI HELP"

{
    orbit --help 2>&1 || true
    echo
    orbit path-response --help 2>&1 || true
    echo
    orbit plot --help 2>&1 || true
} >> "$OUTPUT_FILE"

section "STO2x2x1 TOP-LEVEL CONFIGURATION"

append_file "$PROJECT_DIR/orbit.toml"
append_file "$PROJECT_DIR/SrTiO3_2x2x1.cif"
append_file "$PROJECT_DIR/STO2x2x1.cif"

find "$PROJECT_DIR" -maxdepth 1 -type f \
    \( -name '*.cif' -o -name '*.toml' \) \
    -print 2>/dev/null | sort >> "$OUTPUT_FILE"

section "STO2x2x1 RELEVANT TREE"

find "$PROJECT_DIR" -maxdepth 8 -type f \
    \( \
        -iname '*polar*' -o \
        -iname '*response*' -o \
        -iname '*berry*' -o \
        -iname '*born*' -o \
        -name 'run.json' -o \
        -name 'path.json' -o \
        -name 'path.csv' -o \
        -name 'extraction.json' \
    \) \
    -printf '%p  %s bytes\n' 2>/dev/null |
sort >> "$OUTPUT_FILE"

section "PATH METADATA"

while IFS= read -r file; do
    append_file "$file" 1200
done < <(
    find "$PROJECT_DIR/paths" "$PROJECT_DIR/calculations/paths" \
        -type f \
        \( \
            -name 'run.json' -o \
            -name 'path.json' -o \
            -name 'path.csv' -o \
            -name 'extraction.json' \
        \) \
        2>/dev/null |
    sort |
    head -80
)

section "POLARIZATION/RESPONSE METADATA AND TABLES"

while IFS= read -r file; do
    append_file "$file" 1800
done < <(
    find "$PROJECT_DIR" -type f \
        \( \
            -iname '*polar*.json' -o \
            -iname '*polar*.csv' -o \
            -iname '*polar*.dat' -o \
            -iname '*response*.json' -o \
            -iname '*response*.csv' -o \
            -iname '*berry*.json' -o \
            -iname '*berry*.csv' \
        \) \
        2>/dev/null |
    sort |
    head -100
)

section "QE POLARIZATION OUTPUT EXCERPTS"

while IFS= read -r file; do
    {
        echo
        echo "----- QE OUTPUT: $file -----"
        grep -nEi -B 5 -A 14 \
            'polarization|berry phase|electronic dipole|ionic dipole|modulo|P[[:space:]]*=|C/m|e/bohr|phase' \
            "$file" 2>/dev/null |
        head -500 || true
        echo "----- END QE EXCERPT -----"
    } >> "$OUTPUT_FILE"
done < <(
    find "$PROJECT_DIR/calculations/paths" \
        -type f \
        \( -name '*.out' -o -name '*.pwo' -o -name 'espresso.pwo' \) \
        -path '*response*' \
        2>/dev/null |
    sort |
    head -12
)

section "CURRENT FILE CHECKSUMS"

while IFS= read -r file; do
    sha256sum "$file" 2>/dev/null || true
done < <(
    {
        printf '%s\n' "$SOURCE_MATCHES" |
            while IFS= read -r file; do
                [[ -n "$file" ]] || continue
                if [[ "$file" = /* ]]; then
                    printf '%s\n' "$file"
                else
                    printf '%s\n' "$REPO_DIR/$file"
                fi
            done

        find "$REPO_DIR" -maxdepth 2 -type f \
            -iname '*polar*.py' 2>/dev/null
    } |
    sort -u
) >> "$OUTPUT_FILE"

echo "[done] wrote $OUTPUT_FILE"
echo "[next] attach or paste $OUTPUT_FILE"
