# 2026-10-30: pour_balls' coin flip after the pour; `ported_pour_untilt` "capped" (stream H)

## The official failure

S's pour_balls dev 100000 replays (isr-lab `~/data2/ilia/bridge/pbreplay`) show the official run failing and the
integ run succeeding.
- Both pours reach ~126 deg, all balls in.
- `pour_p1`'s raise (`move_delta` +6 cm) is IK-unreachable there. The 46-step hover `move_to` then rolls the cup
  upright while carrying it back.
- In the official run the cup leaves the jaws in that move's first 5 steps (tilt 125 → 75 deg) and lands on its side.
  The integ run is near-identical (perceived cup yaw 5.0 vs 6.9 deg; held 14 mm further out) and the cup stays in.

## Two variants (`RuntimeConfig.ported_pour_untilt`, `Skills`, G3; default "direct")

- **"staged":** after the pour, retrace the profile's reached poses backwards to upright at "slow" speed. The spout
  stays over the target and each step is re-solved from the measured in-hand transform. The hover then only
  translates.
- **"capped":** the profile ends at `PHI_CAP` = 112 deg, then the staged untilt.
  - G3 mode B, every pour that emptied the cup: all balls were in the vase by 94-106 deg.
  - The 118-130 deg increments only add in-hand slip (4.5 → 6.8 deg), and the untilt then starts from there.

## G3 screen

- Mode B (the cup believed off until held) on pour_balls dev 100000-04.
- 20 cup-pose perturbations per layout:
  - yaw ±3 / 4 deg;
  - xy ±3 mm, and 3 mm with yaw ±2 / 4 deg;
  - along y +7 / +11 / +14 / -7 / -11 mm, and +11 mm with 5 deg;
  - x ±7 mm.
- 100 episodes per arm, the same pairs. 0 LLM, compute3, one job at a time.

| arm | success | cup drops: during the pour (slip stop) | cup drops: after the pour | no grasp / other | mean steps |
|---|---|---|---|---|---|
| direct | 53 | 18 | 1 | 20 / 8 | 274 |
| staged | 53 (0 up / 0 down) | 18 | 1 | 20 / 8 | 327 |
| capped | 54 (1 up / 0 down) | 18 | 0 | 20 / 8 | 300 |

- **G3 reproduces the official class once:** perturbation y +3 mm on 100000.
  - Direct: the raise is IK-unreachable, then the 46-step hover. The cup turns 125 → 82 deg in the jaws within 6
    steps and falls.
  - Staged: the same pour to 130 deg. The cup slips already on the first slow step back (130 → 124 deg: tilt 68).
    Speed is not the lever.
  - Capped: the pour ends at 112 deg, the untilt holds, success.
- **The 18 drops during the pour** stop on "in-hand slip" at 88-100 deg, before any cap. They are unchanged by both
  variants. They are G3 mode B's own class; the camera board is 5/5.
- **Rate.** Capped turns the one reproduced case and costs no success. The class is rare in G3 (1 in 100), so this
  does not show a rate.
- **Next check:** the camera 5-pair no-regression check on pour_balls (no pair loses, per the lead's pre-registered
  bar).
