#!/bin/sh

# Cleanup script for TokAlign pipeline
# Removes intermediate files to free up storage space after training

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export MAIN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd ${MAIN_DIR}

# Cleanup mode: safe, moderate, or aggressive
CLEANUP_MODE="${1:-safe}"

echo "=========================================="
echo "TokAlign Cleanup Script"
echo "=========================================="
echo "Mode: ${CLEANUP_MODE}"
echo "Main directory: ${MAIN_DIR}"
echo "=========================================="
echo ""

# Function to calculate directory size
get_size() {
    if [ -d "$1" ]; then
        du -sh "$1" 2>/dev/null | cut -f1
    else
        echo "0"
    fi
}

# Function to ask for confirmation
confirm() {
    read -p "$1 (y/N): " -n 1 -r
    echo
    [[ $REPLY =~ ^[Yy]$ ]]
}

# SAFE CLEANUP: Only removes obviously safe files
safe_cleanup() {
    echo "=== SAFE CLEANUP ==="
    echo "Removing:"
    echo "  - GloVe build artifacts"
    echo "  - Temporary GloVe files"
    echo "  - Old log files"
    echo ""
    
    # GloVe build artifacts (if GloVe directory exists)
    if [ -d "../GloVe/build" ]; then
        echo "Cleaning GloVe build artifacts..."
        rm -rf ../GloVe/build/*.o 2>/dev/null
        echo "  ✓ Removed GloVe object files"
    fi
    
    # GloVe temporary files
    if [ -d "../GloVe" ]; then
        echo "Cleaning GloVe temporary files..."
        cd ../GloVe
        rm -f vocab.src.txt cooccurrence.src.bin cooccurrence.shuf.src.bin 2>/dev/null
        rm -f vocab.*.txt cooccurrence.*.bin 2>/dev/null
        cd ${MAIN_DIR}
        echo "  ✓ Removed GloVe temporary files"
    fi
    
    # Old log files (keep recent ones)
    if [ -d "./log" ]; then
        echo "Cleaning old log files..."
        find ./log -name "*.log" -mtime +7 -delete 2>/dev/null
        echo "  ✓ Removed log files older than 7 days"
    fi
    
    echo ""
    echo "✓ Safe cleanup complete!"
}

# MODERATE CLEANUP: Removes intermediate training files
moderate_cleanup() {
    echo "=== MODERATE CLEANUP ==="
    safe_cleanup
    
    echo ""
    echo "Removing:"
    echo "  - Chunk files (~176GB)"
    echo "  - Tokenized chunks (~200-400GB)"
    echo "  - Original corpus file (~176GB)"
    echo "  - Intermediate training checkpoints (keeping final)"
    echo "  - HuggingFace cache"
    echo "  - GloVe training corpus (already tokenized)"
    echo ""
    
    # Remove chunk files (already tokenized and concatenated)
    if [ -d "./data/pretrain-corpus/chunks" ]; then
        CHUNK_SIZE=$(get_size "./data/pretrain-corpus/chunks")
        echo "Chunk files size: ${CHUNK_SIZE}"
        if confirm "Remove chunk files? (already tokenized and concatenated)"; then
            rm -rf ./data/pretrain-corpus/chunks 2>/dev/null
            echo "  ✓ Removed chunk files (freed ~${CHUNK_SIZE})"
        fi
    fi
    
    # Remove tokenized chunks (already concatenated)
    if [ -d "./data/pretrain-dataset/chunks-tokenized" ]; then
        TOKENIZED_CHUNK_SIZE=$(get_size "./data/pretrain-dataset/chunks-tokenized")
        echo "Tokenized chunks size: ${TOKENIZED_CHUNK_SIZE}"
        if confirm "Remove tokenized chunks? (already concatenated into final dataset)"; then
            rm -rf ./data/pretrain-dataset/chunks-tokenized 2>/dev/null
            echo "  ✓ Removed tokenized chunks (freed ~${TOKENIZED_CHUNK_SIZE})"
        fi
    fi
    
    # Remove original corpus (already tokenized)
    if [ -f "./data/pretrain-corpus/pile-corpus.jsonl" ]; then
        CORPUS_SIZE=$(get_size "./data/pretrain-corpus/pile-corpus.jsonl")
        echo "Original corpus size: ${CORPUS_SIZE}"
        if confirm "Remove original corpus file? (already tokenized)"; then
            rm -f ./data/pretrain-corpus/pile-corpus.jsonl 2>/dev/null
            echo "  ✓ Removed original corpus file (freed ~${CORPUS_SIZE})"
        fi
    fi
    
    # Remove intermediate checkpoints, keep final ones
    if [ -d "./log" ]; then
        echo "Cleaning intermediate checkpoints..."
        # Keep final checkpoints (checkpoint-2500), remove others
        INTERMEDIATE_SIZE=$(du -sh ./log/*/checkpoint-* 2>/dev/null | grep -v "checkpoint-2500" | awk '{sum+=$1} END {print sum}' || echo "0")
        find ./log -type d -name "checkpoint-*" ! -name "checkpoint-2500" -exec rm -rf {} + 2>/dev/null
        echo "  ✓ Removed intermediate checkpoints (kept checkpoint-2500)"
    fi
    
    # HuggingFace cache (can be re-downloaded) - THIS IS LIKELY HUGE!
    if [ -d "./data/cache" ]; then
        CACHE_SIZE=$(get_size "./data/cache")
        echo "HuggingFace cache size: ${CACHE_SIZE}"
        echo "  WARNING: Cache can be very large (5TB+). This is safe to delete but will require re-downloading datasets if you need them again."
        if confirm "Remove HuggingFace cache? (will free ~${CACHE_SIZE}, can be re-downloaded)"; then
            rm -rf ./data/cache/* 2>/dev/null
            echo "  ✓ Removed HuggingFace cache (freed ~${CACHE_SIZE})"
        fi
    fi
    
    # GloVe training corpus (raw text, already tokenized)
    if [ -f "./data/pretrain-corpus/glove-corpus-1B.jsonl" ]; then
        CORPUS_SIZE=$(get_size "./data/pretrain-corpus/glove-corpus-1B.jsonl")
        echo "GloVe corpus size: ${CORPUS_SIZE}"
        if confirm "Remove GloVe training corpus? (already tokenized)"; then
            rm -f ./data/pretrain-corpus/glove-corpus-1B.jsonl
            echo "  ✓ Removed GloVe training corpus"
        fi
    fi
    
    echo ""
    echo "✓ Moderate cleanup complete!"
}

# AGGRESSIVE CLEANUP: Removes everything except final outputs
aggressive_cleanup() {
    echo "=== AGGRESSIVE CLEANUP ==="
    echo "WARNING: This will remove most intermediate files!"
    echo ""
    echo "KEEPING:"
    echo "  - Final trained models (log/*/checkpoint-2500)"
    echo "  - Final tokenized dataset (data/pretrain-dataset/pile00-gemma-7b-tokenized)"
    echo "  - Alignment matrices (data/pythia2gemma-7b/align_matrix.json)"
    echo "  - GloVe vectors (data/vec-*.txt)"
    echo ""
    echo "REMOVING:"
    echo "  - Chunk files (~176GB)"
    echo "  - Tokenized chunks (~176GB)"
    echo "  - Original corpus (~176GB)"
    echo "  - HuggingFace cache (~5TB)"
    echo "  - All intermediate checkpoints"
    echo "  - All logs"
    echo ""
    
    if ! confirm "Are you sure you want to proceed with aggressive cleanup?"; then
        echo "Cleanup cancelled."
        exit 0
    fi
    
    moderate_cleanup
    
    echo ""
    echo "Removing chunk files..."
    # Remove raw chunk files (already tokenized and concatenated)
    if [ -d "./data/pretrain-corpus/chunks" ]; then
        CHUNK_SIZE=$(get_size "./data/pretrain-corpus/chunks")
        echo "Chunk files size: ${CHUNK_SIZE}"
        if confirm "Remove chunk files? (already tokenized and concatenated)"; then
            rm -rf ./data/pretrain-corpus/chunks 2>/dev/null
            echo "  ✓ Removed chunk files"
        fi
    fi
    
    echo "Removing tokenized chunk directories..."
    # Remove tokenized chunks (already concatenated)
    if [ -d "./data/pretrain-dataset/chunks-tokenized" ]; then
        TOKENIZED_CHUNK_SIZE=$(get_size "./data/pretrain-dataset/chunks-tokenized")
        echo "Tokenized chunks size: ${TOKENIZED_CHUNK_SIZE}"
        if confirm "Remove tokenized chunks? (already concatenated into final dataset)"; then
            rm -rf ./data/pretrain-dataset/chunks-tokenized 2>/dev/null
            echo "  ✓ Removed tokenized chunks"
        fi
    fi
    
    echo "Removing original corpus file..."
    # Remove original corpus (already tokenized)
    if [ -f "./data/pretrain-corpus/pile-corpus.jsonl" ]; then
        CORPUS_SIZE=$(get_size "./data/pretrain-corpus/pile-corpus.jsonl")
        echo "Original corpus size: ${CORPUS_SIZE}"
        if confirm "Remove original corpus file? (already tokenized)"; then
            rm -f ./data/pretrain-corpus/pile-corpus.jsonl 2>/dev/null
            echo "  ✓ Removed original corpus file"
        fi
    fi
    
    echo "Removing other tokenized datasets (if not needed)..."
    # Keep directory structure but remove large tokenized datasets
    if [ -d "./data/pretrain-dataset" ]; then
        DATASET_SIZE=$(get_size "./data/pretrain-dataset")
        echo "Tokenized datasets directory size: ${DATASET_SIZE}"
        echo "  (Keeping final dataset: pile00-gemma-7b-tokenized)"
        # Only remove if user confirms - be careful here
    fi
    
    echo "Removing all logs..."
    if [ -d "./log" ]; then
        find ./log -name "*.log" -delete 2>/dev/null
        echo "  ✓ Removed all log files"
    fi
    
    echo ""
    echo "✓ Aggressive cleanup complete!"
}

# Show current disk usage
show_usage() {
    echo "=== CURRENT DISK USAGE ==="
    echo ""
    
    if [ -d "./data" ]; then
        echo "Data directory:"
        du -sh ./data/* 2>/dev/null | sort -h
        echo ""
    fi
    
    if [ -d "./log" ]; then
        echo "Log directory:"
        du -sh ./log/* 2>/dev/null | sort -h
        echo ""
    fi
    
    TOTAL=$(du -sh . 2>/dev/null | cut -f1)
    echo "Total project size: ${TOTAL}"
    echo ""
}

# Main execution
case "$CLEANUP_MODE" in
    safe)
        show_usage
        safe_cleanup
        ;;
    moderate)
        show_usage
        moderate_cleanup
        ;;
    aggressive)
        show_usage
        aggressive_cleanup
        ;;
    usage|help)
        echo "Usage: $0 [safe|moderate|aggressive|usage]"
        echo ""
        echo "Modes:"
        echo "  safe       - Remove only build artifacts and old logs (safest)"
        echo "  moderate   - Remove intermediate checkpoints and caches"
        echo "  aggressive - Remove everything except final models (most space saved)"
        echo "  usage      - Show current disk usage"
        echo ""
        echo "Default: safe"
        exit 0
        ;;
    *)
        echo "Unknown mode: $CLEANUP_MODE"
        echo "Usage: $0 [safe|moderate|aggressive|usage]"
        exit 1
        ;;
esac

echo ""
echo "=== FINAL DISK USAGE ==="
show_usage

echo "Cleanup complete!"
echo ""
echo "To see what was kept:"
echo "  - Final models: ./log/*/checkpoint-2500"
echo "  - Alignment matrix: ./data/pythia2gemma-7b/align_matrix.json"
echo "  - GloVe vectors: ./data/vec-*.txt"

