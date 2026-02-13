#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────────────
# MLHP Chapter 1 — Stitch all animations into a single video
#
# Usage:
#   cd /lfs/skampere2/0/sttruong/mlhp
#   bash animations/ch1/stitch.sh                          # crossfade (default)
#   bash animations/ch1/stitch.sh --simple                 # hard-cut (faster)
#   bash animations/ch1/stitch.sh --music animations/music/chopin_nocturne_op9_no2.mp3
#
# Music options:
#   --music <file>       Path to a background music file (mp3/wav/flac)
#   --music-volume 0.12  Music volume level (0.0–1.0, default: 0.12)
#
# Output: animations/ch1/chapter1.mp4
# ────────────────────────────────────────────────────────────────────
set -euo pipefail

export PATH="/lfs/local/0/sttruong/miniconda3/bin:$PATH"

# ── parse arguments ────────────────────────────────────────────────
SIMPLE=false
MUSIC_FILE=""
MUSIC_VOL="0.12"   # default: soft background level

while [[ $# -gt 0 ]]; do
    case "$1" in
        --simple)       SIMPLE=true; shift ;;
        --music)        MUSIC_FILE="$2"; shift 2 ;;
        --music-volume) MUSIC_VOL="$2"; shift 2 ;;
        *)              echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -n "$MUSIC_FILE" && ! -f "$MUSIC_FILE" ]]; then
    echo "ERROR: Music file not found: $MUSIC_FILE"
    exit 1
fi

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MEDIA="$ROOT/media/ch1/videos"
OUT="$ROOT/animations/ch1/chapter1.mp4"
TMPDIR="$(mktemp -d)"

# ── clip list (in narrative order from script.md) ───────────────────
# Format: directory/resolution/filename
CLIPS=(
    "section_titles/1080p60/ChapterOpening.mp4"

    "section_titles/1080p60/Part1Title.mp4"
    "response_matrix/1080p60/ResponseMatrixSort.mp4"

    "section_titles/1080p60/Part2Title.mp4"
    "rasch_to_bt/1080p60/RaschToBradleyTerry.mp4"

    "section_titles/1080p60/Part3Title.mp4"
    "ackley_sampling/1080p60/AckleySampling.mp4"

    "section_titles/1080p60/Part4Title.mp4"
    "softmax_choice/1080p60/SoftmaxChoice.mp4"

    "section_titles/1080p60/Part5Title.mp4"
    "red_bus_blue_bus/1080p60/RedBusBlueBus.mp4"
    "mixture_iia/1080p60/MixtureIIAViolation.mp4"

    "gp_prior_samples/1080p60/GPPriorSamples.mp4"

    "section_titles/1080p60/ChapterClosing.mp4"
)

# ── validate all clips exist ────────────────────────────────────────
echo "Checking clips..."
for clip in "${CLIPS[@]}"; do
    full="$MEDIA/$clip"
    if [[ ! -f "$full" ]]; then
        echo "ERROR: Missing $full"
        echo "  Run: manim -qh --disable_caching --media_dir media/ch1 animations/ch1/<file>.py <Scene>"
        exit 1
    fi
done
echo "All ${#CLIPS[@]} clips found."

# ── get duration of a clip ──────────────────────────────────────────
get_duration() {
    ffprobe -v quiet -show_entries format=duration \
        -of default=noprint_wrappers=1:nokey=1 "$1" 2>/dev/null
}

# ── helper: mix background music into a video file ────────────────
add_music() {
    local video_in="$1"
    local music_in="$2"
    local vol="$3"
    local video_out="$4"

    echo "Mixing background music (volume=${vol})..."
    ffmpeg -y -i "$video_in" -i "$music_in" \
        -filter_complex "[1:a]volume=${vol},afade=t=in:d=2,afade=t=out:st=$(python3 -c "
dur=$(get_duration "$video_in")
print(max(0, float(dur) - 3))
"):d=3[bg];[bg]apad[bgpad];[bgpad]atrim=0:$(get_duration "$video_in")[music]" \
        -map 0:v -map "[music]" \
        -c:v copy -c:a aac -b:a 192k \
        -shortest \
        "$video_out" 2>/dev/null
    echo "Music added."
}

# ════════════════════════════════════════════════════════════════════
#  SIMPLE MODE: hard cuts, lossless concat
# ════════════════════════════════════════════════════════════════════
if $SIMPLE; then
    echo ""
    echo "=== Simple mode: hard-cut concatenation ==="
    CONCAT_LIST="$TMPDIR/concat.txt"
    for clip in "${CLIPS[@]}"; do
        echo "file '$MEDIA/$clip'" >> "$CONCAT_LIST"
    done

    if [[ -n "$MUSIC_FILE" ]]; then
        # Concat video first, then mix music
        SILENT_OUT="$TMPDIR/silent.mp4"
        ffmpeg -y -f concat -safe 0 -i "$CONCAT_LIST" \
            -c copy "$SILENT_OUT" 2>/dev/null
        add_music "$SILENT_OUT" "$MUSIC_FILE" "$MUSIC_VOL" "$OUT"
    else
        ffmpeg -y -f concat -safe 0 -i "$CONCAT_LIST" \
            -c copy "$OUT" 2>/dev/null
    fi

    echo "Done: $OUT"
    total_dur=$(get_duration "$OUT")
    echo "Total duration: ${total_dur}s"
    rm -rf "$TMPDIR"
    exit 0
fi

# ════════════════════════════════════════════════════════════════════
#  DEFAULT MODE: crossfade between clips
# ════════════════════════════════════════════════════════════════════
echo ""
echo "=== Crossfade mode ==="

FADE=0.5  # crossfade duration in seconds

# Get all durations
declare -a DURS
for clip in "${CLIPS[@]}"; do
    d=$(get_duration "$MEDIA/$clip")
    DURS+=("$d")
done

N=${#CLIPS[@]}

# Build the ffmpeg filter_complex for chained xfade
INPUTS=""
for clip in "${CLIPS[@]}"; do
    INPUTS="$INPUTS -i $MEDIA/$clip"
done

FILTER=""
running_offset=0

for (( i=0; i<N-1; i++ )); do
    dur_i=${DURS[$i]}
    offset=$(python3 -c "print(round($running_offset + $dur_i - $FADE, 4))")

    if (( i == 0 )); then
        in_a="[0:v]"
    else
        in_a="[v${i}]"
    fi
    in_b="[$((i+1)):v]"

    j=$((i+1))
    if (( j == N-1 )); then
        out_tag="[vout]"
    else
        out_tag="[v${j}]"
    fi

    FILTER="${FILTER}${in_a}${in_b}xfade=transition=fade:duration=${FADE}:offset=${offset}${out_tag};"

    running_offset=$(python3 -c "print(round($running_offset + $dur_i - $FADE, 4))")
done

# Remove trailing semicolon
FILTER="${FILTER%;}"

echo "Building crossfade with ${N} clips, ${FADE}s fade..."
echo "Filter has $((N-1)) xfade stages."

# Run ffmpeg
if [[ -n "$MUSIC_FILE" ]]; then
    SILENT_OUT="$TMPDIR/silent_xfade.mp4"
    eval ffmpeg -y $INPUTS \
        -filter_complex "\"${FILTER}\"" \
        -map '"[vout]"' \
        -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p \
        -r 60 \
        "\"$SILENT_OUT\"" 2>/dev/null
    add_music "$SILENT_OUT" "$MUSIC_FILE" "$MUSIC_VOL" "$OUT"
else
    eval ffmpeg -y $INPUTS \
        -filter_complex "\"${FILTER}\"" \
        -map '"[vout]"' \
        -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p \
        -r 60 \
        "\"$OUT\"" 2>/dev/null
fi

echo ""
echo "Done: $OUT"
total_dur=$(get_duration "$OUT")
echo "Total duration: ${total_dur}s"

# Print segment breakdown
echo ""
echo "Segment breakdown:"
echo "──────────────────────────────────────────"
printf "%-35s %8s\n" "Clip" "Duration"
echo "──────────────────────────────────────────"
for (( i=0; i<N; i++ )); do
    name=$(python3 -c "print('${CLIPS[$i]}'.split('/')[-1])")
    printf "%-35s %7.1fs\n" "$name" "${DURS[$i]}"
done
echo "──────────────────────────────────────────"
printf "%-35s %7.1fs\n" "Total (with crossfades)" "$total_dur"

rm -rf "$TMPDIR"
