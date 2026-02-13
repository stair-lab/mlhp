#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────────────
# MLHP Chapter 1 — Build narrated video
#
# Combines per-section narration audio with animation clips.
# When narration is longer than animation, the last frame freezes.
# When there's no animation (opening/closing), extends the title card.
#
# Usage:
#   bash animations/ch1/stitch_narrated.sh
#   bash animations/ch1/stitch_narrated.sh --music animations/music/chopin_nocturne_op9_no2.mp3
#
# Output: animations/ch1/chapter1_narrated.mp4
# ────────────────────────────────────────────────────────────────────
set -euo pipefail

export PATH="/lfs/local/0/sttruong/miniconda3/bin:$PATH"

# ── parse arguments ──────────────────────────────────────────────
MUSIC_FILE=""
MUSIC_VOL="0.06"  # lower default for narrated video

while [[ $# -gt 0 ]]; do
    case "$1" in
        --music)        MUSIC_FILE="$2"; shift 2 ;;
        --music-volume) MUSIC_VOL="$2"; shift 2 ;;
        *)              echo "Unknown option: $1"; exit 1 ;;
    esac
done

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MEDIA="$ROOT/media/ch1/videos"
NAR="$ROOT/animations/ch1/narration"
TITLES="$MEDIA/section_titles/1080p60"
OUT="$ROOT/animations/ch1/chapter1_narrated.mp4"
TMPDIR="$(mktemp -d)"

get_duration() {
    ffprobe -v quiet -show_entries format=duration \
        -of default=noprint_wrappers=1:nokey=1 "$1" 2>/dev/null
}

# ── Section definitions ──────────────────────────────────────────
# Format: "section_id  title_card_video  animation_video  narration_audio"
# Use "NONE" if no animation/title for that section.
SECTIONS=(
    "opening        $TITLES/ChapterOpening.mp4    NONE                                                              $NAR/part1_opening.mp3"
    "part1          $TITLES/Part1Title.mp4         $MEDIA/response_matrix/1080p60/ResponseMatrixSort.mp4             $NAR/part1_response_matrix.mp3"
    "part2          $TITLES/Part2Title.mp4         $MEDIA/rasch_to_bt/1080p60/RaschToBradleyTerry.mp4               $NAR/part2_rasch_to_bt.mp3"
    "part3          $TITLES/Part3Title.mp4         $MEDIA/ackley_sampling/1080p60/AckleySampling.mp4                $NAR/part3_ackley.mp3"
    "part4          $TITLES/Part4Title.mp4         $MEDIA/softmax_choice/1080p60/SoftmaxChoice.mp4                  $NAR/part4_softmax.mp3"
    "part5a         $TITLES/Part5Title.mp4         $MEDIA/red_bus_blue_bus/1080p60/RedBusBlueBus.mp4                $NAR/part5_red_bus.mp3"
    "part5b         NONE                           $MEDIA/mixture_iia/1080p60/MixtureIIAViolation.mp4              $NAR/part5_mixture.mp3"
    "part5c         NONE                           $MEDIA/gp_prior_samples/1080p60/GPPriorSamples.mp4              $NAR/part5_gp.mp3"
    "closing        $TITLES/ChapterClosing.mp4     NONE                                                              $NAR/part6_closing.mp3"
)

echo "Building narrated chapter video..."
echo ""

SEGMENT_FILES=()

for section_line in "${SECTIONS[@]}"; do
    read -r sec_id title_vid anim_vid nar_audio <<< "$section_line"
    echo "── Section: $sec_id"

    nar_dur=$(get_duration "$nar_audio")
    echo "   Narration: ${nar_dur}s"

    segment_out="$TMPDIR/${sec_id}.mp4"

    if [[ "$anim_vid" == "NONE" && "$title_vid" != "NONE" ]]; then
        # No animation — extend title card to match narration length
        title_dur=$(get_duration "$title_vid")
        echo "   Title card: ${title_dur}s (extending to ${nar_dur}s)"

        ffmpeg -y -i "$title_vid" -i "$nar_audio" \
            -filter_complex "[0:v]tpad=stop_mode=clone:stop_duration=$(python3 -c "print(max(0, $nar_dur - $title_dur))")[v]" \
            -map "[v]" -map 1:a \
            -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p -r 60 \
            -c:a aac -b:a 192k \
            -shortest \
            "$segment_out" 2>/dev/null

    elif [[ "$title_vid" != "NONE" && "$anim_vid" != "NONE" ]]; then
        # Title card + animation — concat video, then overlay narration
        anim_dur=$(get_duration "$anim_vid")
        title_dur=$(get_duration "$title_vid")
        total_vid=$(python3 -c "print($title_dur + $anim_dur)")
        echo "   Title: ${title_dur}s + Animation: ${anim_dur}s = ${total_vid}s"

        # First concat title + animation
        concat_list="$TMPDIR/${sec_id}_concat.txt"
        echo "file '$title_vid'" > "$concat_list"
        echo "file '$anim_vid'" >> "$concat_list"
        concat_vid="$TMPDIR/${sec_id}_concat.mp4"

        ffmpeg -y -f concat -safe 0 -i "$concat_list" \
            -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p -r 60 \
            "$concat_vid" 2>/dev/null

        concat_dur=$(get_duration "$concat_vid")

        # Extend with freeze-frame if narration is longer
        extra=$(python3 -c "print(max(0, $nar_dur - $concat_dur))")
        echo "   Extending by ${extra}s for narration"

        ffmpeg -y -i "$concat_vid" -i "$nar_audio" \
            -filter_complex "[0:v]tpad=stop_mode=clone:stop_duration=${extra}[v]" \
            -map "[v]" -map 1:a \
            -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p -r 60 \
            -c:a aac -b:a 192k \
            -shortest \
            "$segment_out" 2>/dev/null

    elif [[ "$title_vid" == "NONE" && "$anim_vid" != "NONE" ]]; then
        # Animation only (no title card) — extend with freeze-frame
        anim_dur=$(get_duration "$anim_vid")
        extra=$(python3 -c "print(max(0, $nar_dur - $anim_dur))")
        echo "   Animation: ${anim_dur}s (extending by ${extra}s)"

        ffmpeg -y -i "$anim_vid" -i "$nar_audio" \
            -filter_complex "[0:v]tpad=stop_mode=clone:stop_duration=${extra}[v]" \
            -map "[v]" -map 1:a \
            -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p -r 60 \
            -c:a aac -b:a 192k \
            -shortest \
            "$segment_out" 2>/dev/null
    fi

    seg_dur=$(get_duration "$segment_out")
    echo "   Output: ${seg_dur}s"
    echo ""
    SEGMENT_FILES+=("$segment_out")
done

# ── Concatenate all segments ─────────────────────────────────────
echo "Concatenating ${#SEGMENT_FILES[@]} segments..."
FINAL_CONCAT="$TMPDIR/final_concat.txt"
for seg in "${SEGMENT_FILES[@]}"; do
    echo "file '$seg'" >> "$FINAL_CONCAT"
done

if [[ -n "$MUSIC_FILE" ]]; then
    SILENT_OUT="$TMPDIR/narrated_silent.mp4"
    ffmpeg -y -f concat -safe 0 -i "$FINAL_CONCAT" \
        -c copy "$SILENT_OUT" 2>/dev/null

    # Mix background music at low volume under narration (loop music to cover full duration)
    vid_dur=$(get_duration "$SILENT_OUT")
    echo "Adding background music (volume=${MUSIC_VOL}, looped)..."
    ffmpeg -y -i "$SILENT_OUT" -stream_loop -1 -i "$MUSIC_FILE" \
        -filter_complex \
        "[0:a]volume=1.0[voice];[1:a]atrim=0:${vid_dur},volume=${MUSIC_VOL},afade=t=in:d=3,afade=t=out:st=$(python3 -c "print(max(0, float($vid_dur) - 4))"):d=4[music];[voice][music]amix=inputs=2:duration=first[aout]" \
        -map 0:v -map "[aout]" \
        -c:v copy -c:a aac -b:a 192k \
        "$OUT" 2>/dev/null
else
    ffmpeg -y -f concat -safe 0 -i "$FINAL_CONCAT" \
        -c copy "$OUT" 2>/dev/null
fi

total_dur=$(get_duration "$OUT")
echo ""
echo "════════════════════════════════════════════"
echo "Done: $OUT"
echo "Total duration: ${total_dur}s ($(python3 -c "m,s=divmod(int($total_dur),60); print(f'{m}:{s:02d}')"))"
echo "════════════════════════════════════════════"

# Segment breakdown
echo ""
echo "Segment breakdown:"
echo "──────────────────────────────────────────"
for seg in "${SEGMENT_FILES[@]}"; do
    d=$(get_duration "$seg")
    printf "  %-30s %6.1fs\n" "$(basename "$seg" .mp4)" "$d"
done
echo "──────────────────────────────────────────"
printf "  %-30s %6.1fs\n" "Total" "$total_dur"

rm -rf "$TMPDIR"
