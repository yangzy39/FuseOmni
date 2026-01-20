@echo off
REM REAP-OMNI Pruning Pipeline for Qwen3-Omni-30B-A3B (Windows)
REM
REM This script orchestrates three pruning operations:
REM 1. Vision Modality Stripping - Remove vision encoder and projector
REM 2. REAP Expert Pruning - Prune MoE experts based on audio affinity
REM 3. Layer Similarity Pruning - Remove redundant transformer layers
REM
REM Usage:
REM   run_pruning.bat [OPTIONS]
REM
REM Author: REAP-OMNI Implementation

setlocal enabledelayedexpansion

REM ============================================================================
REM Default Configuration
REM ============================================================================

set "SCRIPT_DIR=%~dp0"
set "MODEL_PATH=%SCRIPT_DIR%..\models\Qwen3-Omni-30B-A3B-Instruct"
set "OUTPUT_DIR=%SCRIPT_DIR%..\models"

REM Pruning parameters
set "RETENTION_RATE=0.5"
set "MAX_LAYERS=8"
set "SIMILARITY_THRESHOLD=0.9"
set "COMPONENT=thinker"

REM Flags
set "RUN_VISION=1"
set "RUN_REAP=1"
set "RUN_LAYER=1"
set "DRY_RUN="
set "VERBOSE=--verbose"

REM ============================================================================
REM Parse Arguments
REM ============================================================================

:parse_args
if "%~1"=="" goto :done_args

if /i "%~1"=="--model-path" (
    set "MODEL_PATH=%~2"
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--output-dir" (
    set "OUTPUT_DIR=%~2"
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--vision-only" (
    set "RUN_VISION=1"
    set "RUN_REAP="
    set "RUN_LAYER="
    shift
    goto :parse_args
)
if /i "%~1"=="--reap-only" (
    set "RUN_VISION="
    set "RUN_REAP=1"
    set "RUN_LAYER="
    shift
    goto :parse_args
)
if /i "%~1"=="--layer-only" (
    set "RUN_VISION="
    set "RUN_REAP="
    set "RUN_LAYER=1"
    shift
    goto :parse_args
)
if /i "%~1"=="--retention-rate" (
    set "RETENTION_RATE=%~2"
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--max-layers" (
    set "MAX_LAYERS=%~2"
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--dry-run" (
    set "DRY_RUN=--dry-run"
    shift
    goto :parse_args
)
if /i "%~1"=="--help" (
    goto :show_help
)
if /i "%~1"=="-h" (
    goto :show_help
)

echo Unknown option: %~1
echo Use --help for usage information
exit /b 1

:done_args

REM ============================================================================
REM Banner
REM ============================================================================

echo.
echo ============================================================
echo   REAP-OMNI Pruning Pipeline
echo   Qwen3-Omni-30B-A3B Model Compression
echo ============================================================
echo.

REM ============================================================================
REM Pre-flight checks
REM ============================================================================

echo [INFO] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.8+.
    exit /b 1
)

echo [INFO] Checking model path: %MODEL_PATH%
if not exist "%MODEL_PATH%" (
    echo [ERROR] Model path not found: %MODEL_PATH%
    exit /b 1
)
if not exist "%MODEL_PATH%\config.json" (
    echo [ERROR] config.json not found in model path
    exit /b 1
)

echo [SUCCESS] Pre-flight checks passed
echo.

REM Create output directory
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

REM ============================================================================
REM Step 1: Vision Modality Stripping
REM ============================================================================

if defined RUN_VISION (
    echo ------------------------------------------------------------
    echo   Step 1: Vision Modality Stripping
    echo ------------------------------------------------------------
    
    set "VISION_OUTPUT=%OUTPUT_DIR%\Qwen3-Omni-30B-A3B-Vision-Stripped"
    
    echo [INFO] Input:  %MODEL_PATH%
    echo [INFO] Output: !VISION_OUTPUT!
    
    python "%SCRIPT_DIR%vision_strip.py" ^
        --model-path "%MODEL_PATH%" ^
        --output-path "!VISION_OUTPUT!" ^
        %DRY_RUN% %VERBOSE%
    
    if errorlevel 1 (
        echo [ERROR] Vision stripping failed
        exit /b 1
    )
    
    if not defined DRY_RUN (
        echo [SUCCESS] Vision stripping completed
        set "MODEL_PATH=!VISION_OUTPUT!"
    )
    echo.
)

REM ============================================================================
REM Step 2: REAP Expert Pruning
REM ============================================================================

if defined RUN_REAP (
    echo ------------------------------------------------------------
    echo   Step 2: REAP Expert Pruning
    echo ------------------------------------------------------------
    
    set "REAP_OUTPUT=%OUTPUT_DIR%\Qwen3-Omni-30B-A3B-REAP-Pruned"
    
    echo [INFO] Input:  %MODEL_PATH%
    echo [INFO] Output: !REAP_OUTPUT!
    echo [INFO] Retention Rate: %RETENTION_RATE%
    
    python "%SCRIPT_DIR%reap_expert_pruning.py" ^
        --model-path "%MODEL_PATH%" ^
        --output-path "!REAP_OUTPUT!" ^
        --component "%COMPONENT%" ^
        --retention-rate %RETENTION_RATE% ^
        %DRY_RUN% %VERBOSE%
    
    if errorlevel 1 (
        echo [ERROR] REAP pruning failed
        exit /b 1
    )
    
    if not defined DRY_RUN (
        echo [SUCCESS] REAP expert pruning completed
        set "MODEL_PATH=!REAP_OUTPUT!"
    )
    echo.
)

REM ============================================================================
REM Step 3: Layer Similarity Pruning
REM ============================================================================

if defined RUN_LAYER (
    echo ------------------------------------------------------------
    echo   Step 3: Layer Similarity Pruning
    echo ------------------------------------------------------------
    
    set "LAYER_OUTPUT=%OUTPUT_DIR%\Qwen3-Omni-30B-A3B-Layer-Pruned"
    
    echo [INFO] Input:  %MODEL_PATH%
    echo [INFO] Output: !LAYER_OUTPUT!
    echo [INFO] Max Layers: %MAX_LAYERS%
    
    python "%SCRIPT_DIR%layer_similarity_pruning.py" ^
        --model-path "%MODEL_PATH%" ^
        --output-path "!LAYER_OUTPUT!" ^
        --component "%COMPONENT%" ^
        --max-layers %MAX_LAYERS% ^
        --similarity-threshold %SIMILARITY_THRESHOLD% ^
        %DRY_RUN% %VERBOSE%
    
    if errorlevel 1 (
        echo [ERROR] Layer pruning failed
        exit /b 1
    )
    
    if not defined DRY_RUN (
        echo [SUCCESS] Layer similarity pruning completed
    )
    echo.
)

REM ============================================================================
REM Summary
REM ============================================================================

echo ------------------------------------------------------------
echo   Pipeline Complete
echo ------------------------------------------------------------

if defined DRY_RUN (
    echo [INFO] This was a DRY RUN. No files were modified.
) else (
    echo [INFO] Pruned models saved to: %OUTPUT_DIR%
    echo.
    echo Output models:
    if defined RUN_VISION echo   - Vision-stripped: %OUTPUT_DIR%\Qwen3-Omni-30B-A3B-Vision-Stripped
    if defined RUN_REAP echo   - REAP-pruned:     %OUTPUT_DIR%\Qwen3-Omni-30B-A3B-REAP-Pruned
    if defined RUN_LAYER echo   - Layer-pruned:    %OUTPUT_DIR%\Qwen3-Omni-30B-A3B-Layer-Pruned
)

echo.
echo ============================================================
echo   REAP-OMNI Pruning Pipeline Finished
echo ============================================================
echo.

exit /b 0

REM ============================================================================
REM Help
REM ============================================================================

:show_help
echo REAP-OMNI Pruning Pipeline for Qwen3-Omni-30B-A3B
echo.
echo Usage: %~nx0 [OPTIONS]
echo.
echo Options:
echo   --model-path PATH       Path to original model
echo   --output-dir PATH       Base output directory
echo   --vision-only           Only run vision stripping
echo   --reap-only             Only run REAP expert pruning
echo   --layer-only            Only run layer pruning
echo   --retention-rate RATE   Expert retention rate (default: 0.5)
echo   --max-layers NUM        Maximum layers to prune (default: 8)
echo   --dry-run               Analyze without making changes
echo   --help                  Show this help message
echo.
echo Examples:
echo.
echo   # Run all pruning steps
echo   %~nx0 --model-path .\models\Qwen3-Omni-30B-A3B-Instruct
echo.
echo   # Only strip vision modality
echo   %~nx0 --vision-only
echo.
echo   # REAP pruning with 60%% expert retention
echo   %~nx0 --reap-only --retention-rate 0.6
echo.
echo   # Dry run to preview changes
echo   %~nx0 --dry-run
echo.
exit /b 0
