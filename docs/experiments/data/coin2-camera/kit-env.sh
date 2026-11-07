# Per-job operational environment; no credentials or global lane changes.
export SIMLAB_KIT_ARGS="--ext-folder /home/haptics/dojo-data/wave-1106-c3moved/extscache --/app/extensions/registryEnabled=0"
export SIMLAB_PHYSICS_KIT_ARGS="$SIMLAB_KIT_ARGS"
export INTEG_WATCHDOG_S=600 INTEG_WATCHDOG_HARD_S=600 DOJO_MIN_FREE_GB=50
