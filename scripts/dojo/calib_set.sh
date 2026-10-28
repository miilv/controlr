# source me after DOJO_BG_FOLDS is set: the perception service's ONE frozen calibration set for every episode
# (dojo.perception.calib_set; legality audit 2026-10-28, P1-1: the set used to be chosen per episode from the layout id).
# The default set is built from the calibration-only ids 100040-49 (D-C10b), which are never evaluated. A run on a
# legacy D0 half is a labelled choice: DOJO_CALIB_SET=fold0 (constant too, but its manifest is unknown: rows carry
# calib_disjoint null).
export DOJO_CALIB_SET="${DOJO_CALIB_SET:-frozen_d0_100040_49}"
calib_set_check() {
  if [[ -f "$DOJO_BG_FOLDS/$DOJO_CALIB_SET/cam_head.png" ]]; then
    return 0
  fi
  echo "[calib] no calibration set $DOJO_BG_FOLDS/$DOJO_CALIB_SET: build it (python -m dojo.perception.calib_set build" \
       "--d0 <D0 root> --out $DOJO_BG_FOLDS; journal 2026-10-28-legality-calib-fold) or name a legacy set" \
       "explicitly (DOJO_CALIB_SET=fold0)" >&2
  return 1
}
