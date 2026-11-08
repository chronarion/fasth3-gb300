#!/usr/bin/env bash
# Download the FastH3 weights at the revisions this release was measured with into $HF_HOME/hub, then build the composed
# model directories the serve/ configs point at ($HF_HOME/composed/*, relative symlinks into the hub snapshots).
#
#   HF_HOME=/path/to/hf tools/download_models.sh [t2v] [qad] [omni]      (no argument: all three)
#
#   t2v   FastVideo-FastH3-8-Step-V2 (text encoder, transformer, VAEs; ~138 GB) + the NVFP4 VAE decoders (~1.6 GB)
#   qad   + FastH3-QAD-v2's NVFP4 transformer (~38 GB), for profiles/qad-nvfp4.env and profiles/max.env
#   omni  + FastH3-Omni-8-Step-V1 (~66 GB) and FastH3-Omni-NVFP4-QAD's transformer (~37 GB), for the omni service
#
# Needs huggingface_hub's `hf` CLI (pip install -U "huggingface_hub[cli]"), ideally in a container or venv.
set -euo pipefail
: "${HF_HOME:?set HF_HOME to the Hugging Face cache directory (the compose file mounts it as FASTH3_HF)}"
export HF_HOME

V2=FastVideo/FastVideo-FastH3-8-Step-V2;        V2_REV=3da2ddfe1954d9cda4c05b643dc0f26007a655c5
DEC=FastVideo/FastH3-NVFP4-VAE-Decoder;         DEC_REV=bd0c248f2a0f57ca643b3df2ac1f375fa55101fb
QAD=FastVideo/FastH3-QAD-v2;                    QAD_REV=a0dc05a76acdba7733c196b67b424889b6a2272b
OMNI=FastVideo/FastVideo-FastH3-Omni-8-Step-V1; OMNI_REV=53e0d26a1691aa473b6262d8196a8feb2c297df0
OQAD=FastVideo/FastH3-Omni-NVFP4-QAD;           OQAD_REV=f01e6a23ac6066aa9eb622cf94b6259442e70ec2

want() { [ $# -eq 0 ] && return 0; for x in "${ARGS[@]}"; do [ "$x" = "$1" ] && return 0; done; return 1; }
ARGS=("$@"); [ ${#ARGS[@]} -eq 0 ] && ARGS=(t2v qad omni)

snap() { echo "../../hub/models--${1//\//--}/snapshots/$2"; }    # relative to $HF_HOME/composed/<dir>

link_all() {   # link_all DIR REPO REV ENTRY...: symlink each ENTRY of REPO@REV into composed/DIR
    local dir=$HF_HOME/composed/$1 src; src=$(snap "$2" "$3"); shift 3
    mkdir -p "$dir"
    for entry in "$@"; do ln -sfn "$src/$entry" "$dir/$entry"; done
}

# Every component of the t2v model; the QAD and Omni dirs reuse them.
hf download "$V2" --revision "$V2_REV"
hf download "$DEC" --revision "$DEC_REV" --include "quality-36L/*" "fast-8L/*" "README.md"
V2_ENTRIES=(audio_scheduler audio_vae checkpoint_content.json checkpoint_metadata.json fastvideo_inference.json LICENSE
            LICENSE-Qwen3-VL modular_model_index.json NOTICE processor README.md scheduler text_encoder tokenizer vae)

if want qad; then
    hf download "$QAD" --revision "$QAD_REV" --include "transformer/*" "README.md"
    link_all fasth3-qad-v2-fullvae "$V2" "$V2_REV" "${V2_ENTRIES[@]}"
    link_all fasth3-qad-v2-fullvae "$QAD" "$QAD_REV" transformer
fi

if want omni; then
    hf download "$OMNI" --revision "$OMNI_REV"
    hf download "$OQAD" --revision "$OQAD_REV" --include "transformer_ref/*" "*.json" "README.md"
    for dir in fasth3-omni-8step-v1 fasth3-omni-qad; do
        link_all $dir "$V2" "$V2_REV" audio_vae processor text_encoder tokenizer vae
        link_all $dir "$OMNI" "$OMNI_REV" audio_scheduler scheduler fastvideo_inference.json modular_model_index.json
    done
    link_all fasth3-omni-8step-v1 "$OMNI" "$OMNI_REV" transformer_ref
    link_all fasth3-omni-qad "$OQAD" "$OQAD_REV" transformer_ref
fi

# Every link must resolve.
bad=0
for link in "$HF_HOME"/composed/*/*; do [ -e "$link" ] || { echo "broken: $link" >&2; bad=1; }; done
[ $bad = 0 ] && echo "models ready in $HF_HOME"
exit $bad
