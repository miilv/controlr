"""Stream M: the 54 scoreboard cells, best skill-ceiling estimate vs latest perception, recoverable Average points."""
import json, time, collections, sys
from state import options, load_rows
ARGS = options()
sys.path.insert(0, str(ARGS.repo))
from dojo.scoreboard.board import cells

D = str(ARGS.state_dir)
# D-C19 mode (supersedes J5; lead 29 Oct): the joint sealed block 100100-100119, generated after both freezes, scored by the official
# client behind Misha's strict v2. Tranche A = the first 5 stable ids per cell, tranche B = the next 5 (pre-registered). Same 54-cell
# convention (unmeasured cells 0), n >= 5 per cell, same family rules, no D-C10c exclusions; outputs *_dc19a.* / *_dc19b.*
DC19 = None  # sealed measurements deliberately unsupported by this research board
J5 = DC19 is not None                # (internal name kept: "J5" = the held-out block mode)
SFX = f"_dc19{DC19.lower()}" if DC19 else ""
import subprocess
import re
_ANC = {}


def post_boundary(sha, boundary="69890e2"):       # is boundary an ancestor of sha (git, local controlr checkout)?
    sha = str(sha or "")[:7]
    key = (sha, boundary)
    if key not in _ANC:
        _ANC[key] = (subprocess.run(["git", "-C", str(ARGS.repo), "merge-base", "--is-ancestor", boundary, sha],
                                    capture_output=True).returncode == 0) if sha and sha != "None" else False
    return _ANC[key]


# ---------------------------------------------------------------- skill-ceiling candidates (ours: mode gt / G3 mode A)
C = collections.defaultdict(list)


def add(task, date, kind, src, sha, lay, n, succ, score, note=""):
    C[task].append(dict(date=date, kind=kind, src=src, sha=sha, lay=lay, n=n, succ=succ, score=score, note=note))


# my mode-gt rows (proxy, 08b0dd7, submit_v3, verify bundle, 0 LLM)
gt = collections.defaultdict(list)
for r in load_rows(ARGS.state_dir / "gt_rows.json", optional=True):
    gt[r["task"]].append(r)
for t, v in gt.items():
    add(t, "10-28 13", "mode gt (proxy)", "m95 gt", "08b0dd7", "100000-04", len(v), sum(x["success"] for x in v),
        round(100 * sum(x["score"] for x in v) / len(v)))
# G3 mode A references (breadth3 r2a column; m2 = 42a4b6b, m3 = eeb94de; 100000-09)
for t, s, n, sc, ref in [("build_tower", 10, 10, 100, "42a4b6b"), ("deposit_coin", 7, 10, 74, "42a4b6b"),
                         ("fasten_screws", 10, 10, 100, "42a4b6b"), ("fill_egg_holder", 0, 5, 25, "42a4b6b"),
                         ("fill_pen_holder", 1, 10, 20, "42a4b6b"), ("fold_clothes", 8, 10, 84, "42a4b6b"),
                         ("hang_mugs", 4, 10, 57, "42a4b6b"), ("imitate_sorting_sequence", 2, 3, 72, "eeb94de"),
                         ("insert_key", 0, 6, 15, "eeb94de"), ("insert_tubes", 10, 10, 100, "42a4b6b"),
                         ("make_kong", 10, 10, 100, "eeb94de"), ("organize_table", 0, 9, 44, "eeb94de"),
                         ("play_Xylophone", 10, 10, 100, "42a4b6b"), ("play_stacking_toy", 9, 10, 96, "eeb94de"),
                         ("play_tic_tac_toe", 10, 10, 100, "eeb94de"), ("plug_in_charger", 8, 10, 80, "42a4b6b"),
                         ("press_by_number", 10, 10, 100, "eeb94de"), ("stack_blocks", 10, 10, 100, "eeb94de"),
                         ("store_laptop_and_headphones", 0, 10, 0, "42a4b6b"), ("store_tools_in_toolbox", 2, 10, 57, "eeb94de"),
                         ("swap_blocks", 9, 10, 90, "eeb94de"), ("sweep_blocks", 0, 10, 0, "42a4b6b")]:
    add(t, "10-27 04" if ref == "42a4b6b" else "10-27 05", "G3 mode A", "breadth3 ref (integ3 g3 *ref_*)", ref,
        "100000-09", n, s, sc)
# G3 mode A from the streams' journals / run dirs (newest per task)
add("deposit_coin", "10-27 15", "G3 mode A", "journal precision (post-merge main)", "13a28b6", "100000-24", 25, 20, 82)
add("insert_tubes", "10-27 15", "G3 mode A", "journal precision (post-merge main)", "13a28b6", "100000-24", 25, 24, 98)
add("fold_clothes", "10-27 12", "G3 mode A", "journal gennew", "421d5ea", "100000-24", 25, 15, 62)
add("make_toast", "10-27 10", "G3 mode A", "journal tools (raw records)", "6dd647c", "100000-09", 10, 9, 92)
add("match_and_pick_from_conveyor", "10-27 14", "G3 mode A", "journal memory (#177 follow-up)", "475bf80", "100000-24", 25, 24, 96)
add("insert_key", "10-27 14", "G3 mode A", "journal pegs (main + rest-handover fix)", "main~10-27", "14 settled of 100000-24", 14, 10, 75)
add("plug_in_charger", "10-27 09", "G3 mode A", "integ3 g3 reg_plug", "13aaf9a", "100000-09", 10, 9, 90)
add("stack_blocks_by_language", "10-25", "G3 mode A", "skills3 g3 by_language_v1", "68ea915", "100000-09", 10, 10, 100, "stale")
add("pick_from_conveyor_by_image", "10-26", "G3 mode A", "skills6 g3 img_v3", "51b3094", "100000-09", 10, 0, 0, "stale")
add("pick_from_conveyor_by_image", "10-28 19", "G3 mode A", "C g3_v13 (compute4 imitate_c/convimg)", "4309013 (+ #262 demo_gen only)",
    "v13 100000-04 + 100020-24", 10, 0, 0, "grasp_slip 10/10: skill-blocked at the basket lift")
add("imitate_sorting_sequence", "10-27 02", "G3 mode A", "integ3 g3 b3ref", "1f7e965", "100002-20 (demo set)", 10, 8, 82,
    "demo layouts; the eeb94de ref (2/3, 72) ran on 100000-09, demos only on 100002 / 100004")
add("hang_mugs_random", "10-25", "G3 mode A", "skills4 g3 hang_rand_v4", "bcbe853", "110000-09", 10, 0, 20, "stale")
add("make_toast_random", "10-25", "G3 mode A", "skills4 g3 toast_rand_sync_v5", "5887754", "100000-24 (16 stable)", 16, 9, 77, "stale")
add("store_tools_in_toolbox", "10-27 14", "G3 mode A", "journal longh (#169)", "#169", "100000-19", 20, 7, 68)
add("swap_blocks", "10-27 08", "G3 mode A", "journal memory-swap (main)", "main~10-27", "100000-09", 10, 9, 90)
add("store_laptop_and_headphones", "10-27 19", "G3 mode A", "journal integ-c4", "22f6339", "100000-09", 10, 0, 0)
add("sweep_blocks", "10-27 20", "G3 mode A", "journal gennew v10", "001fe61", "100000-09", 10, 1, 10)
# G3 ceiling rows (g3 run, GT poses, physics-only, single plan, 0 LLM): stream M's (compute3 m95/g3c, 173acfa) and P / N / H's
# (compute7 ~/dojo-data/g3c/<task>_<sha>/); one candidate per (task, run dir), dated by its file time, measurement slice only
G3C = collections.defaultdict(list)
# dev rows on the training block (D-C10c: 100030-49 never measures): shown as upside only, never a ceiling
DEV_UPSIDE = {"align_blocks": "dev upside (training block, not a ceiling): N 20/20 on v1 draws of 100030-49, 094d219",
              "solve_equation": "dev upside (training block, not a ceiling): N 18/18 on v1 draws of 100030-49, 004b501"}
# 02d67ce = #282 head + one unmerged tiles commit (N): its rows stand for the merged routes on align / dolls (the tiles commit
# touches tiles only), but are pending for the tile cells (solve, arrange) until the tiles PR merges
# camera paths the lead ruled blocked (the G3 ceiling stands, but the camera system cannot reach it): ranked last for levers
BLOCKED = {"make_kong": "blocked: draw read (D-C17 (3)): the skill must read the face of the tile it draws from the wall; "
                        "S's held-tile reader is not usable (head top-1 0.44, present-to-wrist pose parked)"}
PENDING_CEIL_BY_TASK = {"02d67ce": {"solve_equation", "arrange_largest_number", "arrange_largest_number_random"}}
PENDING_CEIL = {"725221b": "P ported_classify=library (parked: by_language 0/5 vs v3b 3/5)",
                }   # 955afcf (ported_swap=reach, PR #265) joined the candidate (lead's ruling via P, 28 Oct)    # ceilings of unmerged variants: upside, not the candidate's ceiling
G3C_FLAGS = {"e783de43": "N --routes (merged main e783de43)", "f5b7a96": "ported_routes=on + ported_pickup=library (P, PR #289 -> e783de4)", "9440d7a": "ported_pour=full (P, PR #279 -> 643041f)", "7a4e033": "ported_pour=full (P, PR #279 -> 643041f)",
             "dd0f9bf": "new_routes (N, PR #268 head)","5bfe623": "ported_routes=on (P, PR #259 -> 00a85af)", "955afcf": "ported_routes=on + ported_swap=reach (P, PR #265)",
             "0aa98d5": "classify port (P, PR2)", "a615b0a": "classify port (P, PR2)"}
try:
    for r in json.load(open(f"{D}/g3c_rows.json")):
        if 100030 <= r["seed"] <= 100049 or r["seed"] >= 100100 or not r.get("stable", True):
            continue
        G3C[(r["task"], r.get("box", "compute3"), r.get("dir", r["task"] + "_" + r["sha"]))].append(r)
except FileNotFoundError:
    pass
for (t, box, d), v in G3C.items():
    none = sum(1 for x in v if x["plan_none"])
    skills = sorted({x["skill"] for x in v if x["skill"]})
    noplan = none + sum(1 for x in v if x["stage"] == "plan" and not x["plan_none"])
    stages = sorted({x["stage"] for x in v if x["stage"] not in ("plan", "done")})
    note = ("Plan none" if none == len(v) else ("skill " + "/".join(skills)) if skills
            else "skill ran (fails at " + "/".join(stages) + ")" + (f", no plan in {noplan}/{len(v)}" if noplan else "")) \
        + (", 1 ep" if len(v) == 1 else "")
    seeds = sorted(x["seed"] for x in v)
    m = max(x.get("mtime", 0) for x in v)
    src = "m95 g3c" if box == "compute3" else f"g3c {box.replace('compute3g', 'compute3')}:{d}"
    flags = G3C_FLAGS.get(v[0]["sha"])
    if flags:
        note += f", {flags}"
    add(t, time.strftime("%m-%d %H:%M", time.gmtime(m)) if m else "10-28 18", "G3 mode gt", src, v[0]["sha"],
        f"{seeds[0]}-{str(seeds[-1])[-2:]}" if len(seeds) > 1 else str(seeds[0]),
        len(v), sum(bool(x["success"]) for x in v), round(sum(x["score"] for x in v) / len(v)), note)
# P's classify v3b ceilings (merged #266, the default; compute7 cls2 / cls3, not readable while compute7 is off: from P's message)
add("classify_objects", "10-28 19:00", "G3 mode gt", "P cls2/cls3 (compute7, via P)", "a615b0a/0aa98d5", "100000-09", 10, 2, 40,
    "classify v3b (#266), ported_routes=on")
add("classify_objects_by_language", "10-28 19:00", "G3 mode gt", "P cls2/cls3 (compute7, via P)", "a615b0a/0aa98d5",
    "100000-09 (3 lost to OOM / stop)", 7, 3, 56, "classify v3b (#266), ported_routes=on")
# older proxy mode-gt rows (integ runs A_*, GT state injected)
add("stack_bowls", "10-25", "mode gt (proxy)", "integ A_bowls_off_1a", "2855f2e", "100000-04 (unsettled)", 5, 1, 20, "stale")
add("put_bottles_into_dustbin", "10-25", "mode gt (proxy)", "integ A_bottles_off_1a", "2855f2e", "100000-03", 4, 0, 32, "stale")
# robot-harness ledger, tier A_gt, ilia (our G3, 25 Oct)
for t, n, s, sc in [("make_kong", 25, 25, 100), ("press_by_number", 25, 25, 100), ("build_tower", 25, 25, 100),
                    ("stack_blocks", 25, 25, 100), ("swap_blocks", 25, 24, 96)]:
    add(t, "10-25", "tier A_gt (rh ledger, ours)", "robot-harness results/ledger.jsonl", "feat/dojo-skills", "100000-24", n, s, sc, "stale")
# robot-harness ledger, tier A_gt, misha (OTHER system: planning hints only; SEEN = published-seed rows, D-C9)
OTHER = collections.defaultdict(list)
for l in []:  # missing historical ledger: other-system hints omitted
    if not l.strip():
        continue
    r = json.loads(l)
    if r.get("tier") == "A_gt" and r.get("side") == "misha" and "libero" not in str(r.get("task")) and r.get("n", 0) >= 4:
        seen = "seed" in str(r.get("layouts")) or "SEEN" in str(r.get("notes"))
        OTHER[r["task"]].append(dict(date=r["date"][5:], n=r["n"], succ=r["success"], score=r["score"],
                                     lay=("their dev slice 100010-14" if "generated ids" in str(r["layouts"]) else str(r["layouts"])[:24]), sha=str(r.get("commit"))[-7:], seen=seen))

# ---------------------------------------------------------------- latest perception (ours)
P = {}
# breadth3 r2a (genuine 5.1, submit_v1, eeb94de, 100000-04)
for l in []:  # missing historical breadth ledger: never substitute for strict rows     # J5: only rows on 100100-104
    r = json.loads(l)
    P[r["task"]] = dict(date="10-27", src="breadth3 r2a (5.1)", sha=r["commit"].split("@")[-1][-7:], n=r["n"],
                        succ=r["success"], score=round(r["score"]))
# Integration's c4b arms (genuine 5.1, submit_v3): rows from fetch_rows.sh (compute4 runs2, compute7 runs2)
DEVSET_RULING_TS = 1793222046     # lead, 28 Oct 21:14Z: 100000-04 = dev / planning only; measurement = 100005-09 + 100020-29
def is_v15(r):     # Integration's regenerated measurement layouts (generator v15): layout_gen field or a v15 tag in the run name
    g = str(r.get("layout_gen") or "")
    return g.lstrip("v") == "15" or "v15" in r["run"].split("__")[0]


# one candidate row from two families (I): match_and_pick's v15 candidate = matchvetosg_off_v15 (100005-24) + matchpicksg_v15 (100025-29)
FAM_ALIAS = {"v15:c4b_matchpicksg_v15c": "v15:c4b_matchvetosg_off",
             "v15:c4b_insertkeysg_v15c": "v15:c4b_insertkeysg_keyoff",
             "v15:c4b_putbottlessg_v15c": "v15:c4b_putbottlessg_ubgoff"}   # put_bottles v15 candidate = ubgoff + the remaining v15 ids   # insert_key v15 candidate = keyoff (100005-24) + v15 (100025-29)
V15_CELLS = {"push_T", "stack_blocks", "stack_bowls", "build_tower", "put_bottles_into_dustbin", "pack_objects_into_box_random",
             "fill_pen_holder", "insert_key", "plug_in_charger", "store_tools_in_toolbox", "swap_blocks",
             "match_and_pick_from_conveyor", "pick_from_conveyor_by_image"}
# fill_egg_holder stays on its old v6 rows: v15 settling drops the eggs (1/15 usable), a generator artefact (lead, 29 Oct)
KONG_REVERT_SHA = "d7610a7"          # lead (29 Oct): v3 (#315) rejected; S restores "place" = v2 on main (v3 -> "place3"); set the sha
KONG_V3_OTHER = []
EXCLUDE_RUNS = {"c4b_tttcur_gc": "#241 bug (ttt board label shadowed), fixed"}
# GRIPFIX_SHA = PR #278's merge sha (ported_drag "turn"; push_T rows count only from it, with ported_drag == "turn")
# correction (lead, 28 Oct): the root cause is a push_T turn-direction issue (yaws ~180 deg apart -> a wrapped-wrist plan is
# refused), not a grasp read; only push_T / push_T_random rows wait for P's fix sha, other ported cells count normally
PORTED = {"push_T", "push_T_random"}
PORTED_ALL = {"cover_blocks", "push_T", "push_T_random", "swap_T", "general_pickup", "pour_balls_into_vase", "classify_objects",
              "classify_objects_by_language"}       # #278 also adds a bounded camera-mode grasp settle: tag rows pre-#278
GRIPFIX_SHA = "4417ed5"        # PR #278 merged (lead, 28 Oct)
# lead, 28 Oct ~22:10Z: catalogue gaps. Cells blind on every box until W's sync (set CAT_SYNCED to its index md5), and cells
# blind on compute4 before 371abdc7. A blind row is not the candidate's number: dropped (the cell counts 0 until a clean row).
CAT_GAP_ALL = {"fold_clothes_random", "hang_mugs_random", "make_toast_random", "store_laptop_and_headphones_random",
               "sweep_blocks_random"}   # not sweep_blocks: its 2 missing brooms are KNOWN_UNSHIPPED (no layout holds them; I)
CAT_SYNCED = ("2cd7455c",)     # W's 26 gap assets synced (compute4 = compute3, I, 28 Oct): the five gap cells count from it
CAT_GAP_C4 = {"solve_equation", "arrange_largest_number", "arrange_largest_number_random", "classify_objects",
              "classify_objects_by_language", "sort_nesting_dolls_by_size", "sort_nesting_dolls_by_size_random",
              "pour_by_language", "pour_liquid_into_cup", "pour_liquid_into_cup_random"}
GLYPH_REPORT_SHA = "30db5ff"   # #284: the slimmed records keep glyph_read from here on (W)
GLYPH_ON_SHA = "14e25ed"       # glyph_read "on" in the candidate (#280)


def glyph_status(r):         # "ran" (marker seen), "unknown" (14e25ed..30db5ff in an sg job: the reader ran, records stripped), "pre"
    cf = r.get("cfg") or {}
    if cf.get("glyph_ran"):
        return "ran"
    if post_boundary(r.get("head"), GLYPH_REPORT_SHA):
        return "pre"         # post-#284 with no marker: the classifier did not run on the numerals
    run_f = r["run"].split("__")[0].rsplit("_", 1)[0]
    if post_boundary(r.get("head"), GLYPH_ON_SHA) and run_f.rsplit("_", 1)[0].endswith("sg"):
        return "unknown"
    return "pre"


PICKUP_SHA = "e783de4"         # PR #289 merged: ported_pickup "library" (general_pickup: ours first, then v1)
TILES_SHA = "3ad78f2"          # #290 (glyph reads on slot / stacked tiles + operators) + #291 (solve empty-mat placeholder)
N_SHA = "a8349f7"              # PR #282 merged: N's routes; N's cells count from it with new_routes (+ glyph for solve / arrange)
N_CELLS = {"solve_equation", "arrange_largest_number", "arrange_largest_number_random", "align_blocks",
           "sort_nesting_dolls_by_size", "sort_nesting_dolls_by_size_random"}
GLYPH_CELLS = {"solve_equation", "arrange_largest_number", "arrange_largest_number_random"}
POUR_SHA = "643041f"           # PR #279 merged: ported_pour "full" (provisional until P's G3 no-spill check)
EXCLUDE_ROWS = {("c4", "c4b_coverblocks_port_100005"): "pre-seg outlier on compute4 (d868478; Integration)",
                ("c4", "c4b_pusht_port_100005"): "pre-seg outlier on compute4 (d868478; Integration)"}
CONVEYOR = {"match_and_pick_from_conveyor", "pick_from_conveyor_by_image",     # no settled records exist: raw is the slice
            "make_toast", "make_toast_random"}   # make_toast(_random): the toaster rarely settles at rest: raw is its slice too
# fill_egg raw-is-the-slice reverted (lead, 29 Oct): raw 100005 / 07 fail at reset ("Unstable envs"), never an episode
# cell-specific minimum n for a candidate row (lead, 29 Oct): the generator settles only 4 of 15 measurement ids for fill_egg
# S's rehearsal (30 Oct): our integ runner seeds the env with the LAYOUT ID, the official path (upstream SeedManager) with the POSITION
# INDEX in the call: task-internal draws differ (make_kong's discard group, ttt's opponent cell, every task's instruction wording);
# layout geometry is unaffected. Note only; switching to index seeding is a protocol change the lead decides later
SEED_PROTO_CELLS = {"make_kong", "play_tic_tac_toe"}
REBASE_PRE = {"c4b_swapblockssg", "c4b_buildtowersg", "c4b_pluginchargersg", "c4b_fillpensg", "c4b_storetoolssg", "c4b_fastenscrewssg",
              "c4b_organizetablesg", "c4b_hangmugssg", "c4b_playtictactoesg"}   # all 46 re-baseline rows started before 03:08:48Z (I)
SEED_FOOTNOTE = ("Seed protocol: the integ runner seeds the env with the layout id, the official path (upstream SeedManager) with the "
                 "position index within the call; task-internal draws differ (instruction wording for every task; make_kong's discard "
                 "group; play_tic_tac_toe's opponent cell); layout geometry is identical. No number changes until the lead decides.")
MIN_N = {} if J5 else {"fill_egg_holder": (4, "n = 4: generator settles only 4 of 15 ids")}
ARTEFACTS = collections.defaultdict(list)
DIVERGE = collections.defaultdict(lambda: collections.defaultdict(list))
# rows that do not measure the system (setup artefacts): shown, kept out of 'measured now' (their dimension's mean is taken
# over the comparable cells) until a valid row replaces them. Applies only while the named source is the selected row.
NOT_COMPARABLE = {
    "imitate_sorting_sequence": ("breadth3 r2a", "not comparable: most layouts have no dev demonstration, the support arm's "
                                 "stub never moves (C); C re-measures on 15 demo layouts"),
    "pick_from_conveyor_by_image": ("c4b_imgbelt2", "not comparable: the old basket / board placement (basket inside the "
                                    "conveyor); v13 layouts exist only in compute3 skills6 (C's audit)"),
}
ROWS = []
for fn in ("rows_c4.json", "rows_c7.json", "rows_c1m.json", "rows_c1m4.json"):
    try:
        ROWS += [dict(x, box="c1" if "c1m" in fn else fn[5:-5]) for x in load_rows(ARGS.state_dir / fn)]
    except FileNotFoundError:
        pass
for fn in ("norow_c4.json", "norow_c7.json", "norow_c1m.json", "norow_c1m4.json"):     # run dirs with no episode row: cut mid-run (infra)
    try:
        for d in json.load(open(f"{D}/{fn}")):
            if d.startswith("c4b_"):
                ARTEFACTS["cut mid-run, no row (infra; compute7 emergency stop 28 Oct ~19:07Z)" if fn == "norow_c7.json"
                          else "no row yet (running or cut)"].append(d)
    except FileNotFoundError:
        pass
# adopted levers developed on a measurement id: that (cell, id) never counts, in any arm (Misha's review finding P2-7, D-C10c)
# D-C10c, lever-specific (lead, 29 Oct): an id is excluded only from rows that USE the lever developed on it, plus both arms of
# that lever's own A/B (so the pair stays fair). (cell, id) -> (why, uses(row), regex of the lever's A/B families or None)
_c = lambda r: r.get("cfg") or {}  # noqa: E731
L_DRAG = (lambda r: _c(r).get("ported_drag") == "turn", None)
L_PAR = (lambda r: _c(r).get("align_measure") == "parallel", r"_(par|seq)$")
L_PERSIST = (lambda r: post_boundary(r.get("head"), "9199a62"), None)          # #294 is a runtime fix (no flag): sha >= 9199a62
L_LABELS = (lambda r: _c(r).get("series_labels") == "numbered", r"_num(on|off)$")
L_PLIB = (lambda r: _c(r).get("ported_pickup") == "library", None)
L_COIN = (lambda r: _c(r).get("slot_read") in ("on", True), r"slotsg_(on|off)$")   # slot_settle_merged is perception-side (B arm)
L_COIN6 = (lambda r: _c(r).get("slot_read") in ("on", True) or _c(r).get("disc_handover_read") == "on", r"slotsg_(on|off)$")
L_330 = (lambda r: post_boundary(r.get("head"), "384a28d"), None)              # #330 runtime fix: sha >= 384a28df
L_AFIX = (lambda r: _c(r).get("tile_grasp") == "prefer" or _c(r).get("tile_recover") in (True, "on"), r"_afix(on|off)(_v15)?$")
TUNED = {("push_T", 100005): ("ported_drag turn (developed on push_T 100005)", *L_DRAG),
         ("push_T_random", 100005): ("ported_drag turn (developed on push_T 100005)", *L_DRAG),
         ("align_blocks", 100020): ("align_measure parallel (developed on align_blocks 100020)", *L_PAR),
         # audit (29 Oct, lead): class (a) = developed / debugged directly on a measurement episode
         ("solve_equation", 100005): ("read_persist #294 (debugged on its records)", *L_PERSIST),
         **{(t_, i): ("series_labels #299 (replayed on these planning records)", *L_LABELS)
            for t_ in ("classify_objects", "classify_objects_by_language") for i in range(100005, 100009)},
         ("general_pickup", 100006): ("ported_pickup library (motivated by it; its only rescue)", *L_PLIB),
         ("general_pickup", 100008): ("ported_pickup library (motivated by it)", *L_PLIB),
         # C's self-audit for the coin A/B (lead, 29 Oct): the A/B is judged on 100008 + 100020-29
         ("deposit_coin", 100005): ("slot_settle_merged (designed from rb_100005)", *L_COIN),
         ("deposit_coin", 100006): ("slot_read offline screen (100006 render / GT stand pose); disc_handover_read (rb_100006)", *L_COIN6),
         ("deposit_coin", 100007): ("slot_read classes confirmed on it (strict reading)", *L_COIN),
         ("deposit_coin", 100009): ("slot_read classes confirmed on it (strict reading)", *L_COIN),
         # N's self-audit (lead, 29 Oct)
         **{("sweep_blocks_random", i): ("#330 divergence fix (debugged on it)", *L_330) for i in (100005, 100007, 100009)},
         **{(t_, i): ("tile_grasp / tile_recover (afix; designed from arrange_largest_number 100005-07)", *L_AFIX)
            for t_ in ("arrange_largest_number", "arrange_largest_number_random") for i in (100005, 100006, 100007)}}
TUNED_ON = {k: v[0] for k, v in TUNED.items()}
_tg = collections.defaultdict(lambda: collections.defaultdict(list))
for (t_, i), why in sorted(TUNED_ON.items()):
    _tg[why][t_].append(i)
_ids = lambda v: (f"{v[0]}-{str(v[-1])[-2:]}" if len(v) > 1 and v == list(range(v[0], v[-1] + 1)) else " + ".join(str(x) for x in v))  # noqa: E731
TUNED_GROUPED = {}
for why, cs in _tg.items():
    by_ids = collections.defaultdict(list)
    for t_, v in cs.items():
        by_ids[_ids(v)].append(t_)
    for ids, ts in by_ids.items():
        TUNED_GROUPED[f"{' + '.join(ts)} {ids}"] = why
HEADLINE_CAVEAT = ("measured on a slice that was also used for failure analysis (optimistic); clean estimate pending a fresh "
                   "held-out block. "
                   "Perception-model selection used frames that include measurement ids (before the 29 Oct 18:10Z rule). seg v2b's "
                   "adoption (28 Oct) was decided on frame-0 scenes 100000-09 + 100020-24, and seg_propose 'add' was made default on "
                   "bake sets of which 35 of 64 episode ids are measurement ids. Training data was clean (100030-49). Affects every "
                   "cell's measurement rows (optimistic); D-C19's sealed block is the clean measure. "
                   "[sources: S; journals 2026-10-28-seg-twin-v2, plan-none-perception]")
# D-C19 tranches: the pre-registered manifest (dc19_tranches.json: {task: {"A": [ids], "B": [ids]}}) when it exists; else the first 5
# / next 5 ids per cell among stable rows in 100100-100119 (fallback, flagged)
DC19_TRANCHE, DC19_SOURCE = {}, None
if J5:
    try:
        DC19_TRANCHE = {t: tuple(v.get(DC19, ())) for t, v in json.load(open(f"{D}/dc19_tranches.json")).items()}
        DC19_SOURCE = "manifest"
    except FileNotFoundError:
        _st = collections.defaultdict(set)
        for r in ROWS:
            if 100100 <= (r.get("layout_id") or 0) <= 100119 and not r.get("unstable") and r.get("success") is not None:
                _st[r["task"]].add(r["layout_id"])
        DC19_TRANCHE = {t: tuple(sorted(v)[:5] if DC19 == "A" else sorted(v)[5:10]) for t, v in _st.items()}
        DC19_SOURCE = "fallback: first 5 / next 5 stable ids per cell (no manifest)"
fam = collections.defaultdict(dict)          # (task, family) -> {layout_id: row}, a __retry replaces its original
fam_all = collections.defaultdict(list)      # (task, family) -> every counted row (all shas), for boundary watches
EXCL_TUNED = collections.defaultdict(dict)   # (task, family) -> {layout_id: row} dropped by D-C10c (shown un-excluded, never counted)
for r in ROWS:
    lid = r.get("layout_id")
    cf = r.get("cfg") or {}
    if r["task"] in N_CELLS and (not post_boundary(r.get("head"), N_SHA) or not cf.get("new_routes")
                                 or (r["task"] in GLYPH_CELLS and (glyph_status(r) != "ran"
                                                                   or not post_boundary(r.get("head"), TILES_SHA)))):
        ARTEFACTS["N cells before #282 (a8349f7) or without new_routes (+ glyph_read for solve / arrange): not candidate"].append(
            f"{r['box']}:{r['run']}@{r.get('head')}")
        continue
    if r["task"] == "pour_balls_into_vase" and (not post_boundary(r.get("head"), POUR_SHA)
                                                 or (r.get("cfg") or {}).get("ported_pour") != "full"):
        ARTEFACTS["pour_balls before #279 (643041f, ported_pour full): superseded (the believed-empty stop)"].append(
            f"{r['box']}:{r['run']}@{r.get('head')}")
        continue
    if r["task"] in PORTED and (GRIPFIX_SHA is None or not post_boundary(r.get("head"), GRIPFIX_SHA)
                                or (r.get("cfg") or {}).get("ported_drag") != "turn"):
        ARTEFACTS["pre-fix push_T rows (turn-direction / wrapped-wrist refusal; re-run on P's fix sha), not candidate"].append(
            f"{r['box']}:{r['run']}@{r.get('head')}")
        continue
    if 100000 <= lid <= 100004 and r.get("mtime", 0) > DEVSET_RULING_TS:
        ARTEFACTS["dev rows: 100000-04 camera rows after the lead's ruling (dev / planning only), not measurement"].append(
            f"{r.get('box')}:{r['run']}")
        continue
    cat = str((r.get("cfg") or {}).get("catalog_index_md5") or "")
    if r["task"] in CAT_GAP_ALL and not any(cat.startswith(m) for m in CAT_SYNCED):
        ARTEFACTS["catalogue gap (perception-blind to some assets on every box until W's sync): N/C"].append(
            f"{r.get('box')}:{r['run']}@{cat[:8] or '?'}")
        continue
    if r["task"] in CAT_GAP_C4 and cat.startswith("6e90c3"):
        ARTEFACTS["catalogue gap on compute4 before 371abdc7 (W's overlay missing): N/C"].append(f"{r.get('box')}:{r['run']}")
        continue
    if (r["task"] == "make_kong" and post_boundary(r.get("head"), "a21eaee")
            and (KONG_REVERT_SHA is None or not post_boundary(r.get("head"), KONG_REVERT_SHA))
            and (r.get("cfg") or {}).get("kong_draw_read") == "place"):
        ARTEFACTS["make_kong v3 window (a21eaee1 .. S's revert): 'place' is the rejected v3, not candidate"].append(r["run"])
        if "kongon3" not in r["run"]:
            KONG_V3_OTHER.append(f"{r.get('box')}:{r['run']}@{r.get('head')}")
        continue
    if "_dev_" in r["run"] or r["run"].split("__")[0].rsplit("_", 1)[0].endswith("_dev"):   # Integration's dev-row tag: never rows
        ARTEFACTS["dev-tagged runs (_dev_), never counted"].append(f"{r.get('box')}:{r['run']}")
        continue
    if "." in r["run"]:              # renamed dirs: *.badcat (gapped catalogue, I), *.v1killed, ...: never rows
        ARTEFACTS["renamed run dirs (.badcat / .v1killed ...), not rows"].append(f"{r.get('box')}:{r['run']}")
        continue
    if r["task"] == "imitate_sorting_sequence" and ("o2owait" in r["run"]) and str(r.get("head") or "").startswith("c86dc91"):
        ARTEFACTS["imitate o2owait / o2owaitlook on c86dc914: invalid (runtime TypeError in #322 / #323), re-run on W's fix"].append(r["run"])
        continue
    if (r.get("box"), r["run"]) in EXCLUDE_ROWS:
        ARTEFACTS["excluded outliers"].append(f"{r['box']}:{r['run']}")
        continue
    diverged = r.get("category") == "error" and "ivergence" in str(r.get("why"))
    if diverged:   # lead: infra only if PhysX diverged before our robot touched anything (first grip close vs the end step)
        fc = r.get("first_close")
        during = fc is not None and r.get("steps") is not None and r["steps"] > fc
        DIVERGE[r["task"]]["during manipulation (counts 0)" if during else "before any touch (infra)"].append(
            f"{r.get('box')}:{r['run']} close@{fc} end@{r.get('steps')}")
        if during:
            r = dict(r, success=False, score=0.0)
    if (r.get("category") == "infra_failure" or (r.get("steps") == 0 and not r.get("success"))
            or (diverged and not during) or (r.get("category") == "error" and "service exited" in str(r.get("why")))):
        # PhysX divergence: RoboDojo's official client abandons the layout and refills it from a spare, so it is not counted
        ARTEFACTS["infra (watchdog / killed), not counted"].append(r["run"])
        continue
    if r.get("unstable") or r.get("success") is None or lid is None:
        continue
    if J5 != (100100 <= lid <= 100119) or 100030 <= lid <= 100049 or lid >= 100120:   # D-C19 mode: the block only; main: never 100100+
        continue
    if J5 and lid not in DC19_TRANCHE.get(r["task"], ()):
        continue
    f = r["run"].split("__")[0].rsplit("_", 1)[0]
    if f in EXCLUDE_RUNS or r.get("mode") != "perception":
        continue
    if r.get("layout") and not r["layout"].endswith(".settled.json") and r["task"] not in CONVEYOR:
        ARTEFACTS["raw layout (no settled record), episode dropped"].append(f"{r['run']}")
        continue
    c1 = f.endswith("_c1")              # A/B families moved to compute1 (4090 class, no seg / glyph servers): lever deltas only
    if c1:
        f = f[:-3]
    if is_v15(r):                       # A/B pairs and layout dedup never mix layout generators (lead)
        base = f[:-4] if f.endswith("_v15") else f
        # c4b_<fam>_v15 (no arm) = the v15 candidate row -> suffix "v15c"; c4b_<fam>_<arm>_v15 -> the arm's own suffix
        f = "v15:" + (base + "_v15c" if base.rsplit("_", 1)[-1].endswith("sg") else base)
        f = FAM_ALIAS.get(f, f)
    if c1:
        f = "c1:" + f                   # never a candidate row; pairs only within compute1
    _tz = TUNED.get((r["task"], lid))       # measurement integrity (Misha's review P2-7, D-C10c): lever-specific
    if _tz and (_tz[1](r) or (_tz[2] and re.search(_tz[2], f))):
        ARTEFACTS["tuned on this measurement id (lever developed there), not counted"].append(f"{r.get('box')}:{r['run']}")
        _o = EXCL_TUNED[(r["task"], f)].get(lid)       # kept aside: the un-excluded rate is shown next to the counted one
        if _o is None or r["mtime"] > _o["mtime"]:
            EXCL_TUNED[(r["task"], f)][lid] = r
        continue
    old = fam[(r["task"], f)].get(lid)
    if old is None or r["mtime"] > old["mtime"]:
        fam[(r["task"], f)][lid] = r
    fam_all[(r["task"], f)].append(r)
fam = {k: v for k, v in fam.items() if len(v) >= 2}
agg = lambda v: dict(n=len(v), succ=sum(bool(x["success"]) for x in v), score=round(100 * sum(x["score"] or 0 for x in v) / len(v), 1),  # noqa: E731
                     sha=sorted({str(x["head"]) for x in v})[-1], mtime=max(x["mtime"] for x in v),
                     catalogs=sorted({str(x.get("catalog_md5"))[:8] for x in v if x.get("catalog_md5")}),
                     runs=sorted((dict(box=x.get("box"), run=x.get("run"), layout_id=x.get("layout_id"), success=bool(x["success"]),
                                       score=x["score"], head=str(x.get("head"))[:8]) for x in v), key=lambda d: d["layout_id"] or 0))
# measured now = the CANDIDATE system (lead, 28 Oct): submit_v3 + grip_check on, DOJO_GRIP_SOURCE=v2 (arm 8 ON settings),
# no other lever on. Per task the newest candidate family; else the newest flag-off control (marked: not the candidate)
CANDIDATE = ("submit_v3 + grip_check on (grip_source v2) + ported_routes on + ported_swap reach"
             " + new_routes [equation, largest_number, align, row_by_size] (#268 merged dcc5e7a)"
             " + uniform color_map (#267; perception default from main 69890e2 on)"
             " + seg_propose add (#286) + glyph servers digit_v2 / kong_v1b (#284) + catalogue gate (#285), from c65a9ea"
             " + series_count seg (seg_twin v2b server, counting only) + h95 [put_into_opts spot/patient/sequence, drop_verdict"
             " release, refine_recentre_guard vis] (conditional on M's write audit); Integration runs it on compute4 from 20:05Z"
             " at 69890e2 (families c4b_<cell>sg_*)")
CANDIDATE_CFG = {"sha": "b4cd6c2 for not-started lines from ~06:2xZ (#303, vocab_found_needs_score true adopted; started A/B pairs keep their sha); 9199a62 for the queued lines from ~04:00Z (I), every candidate line with align_measure parallel; 3ad78f2 (lead): + #290 glyph on slot / stacked tiles + operators, #291 solve empty-mat placeholder; #275 pour_order merged but default off (not in the candidate); e783de4: + ported_pickup library (#289); 32dc2a9: + seg_add_refit_poor on (#288); cf3ed7f: + seg_add_move_weak on (#287); before it c65a9ea: + seg_propose \"add\" (#286) + both glyph servers (digit_v2, kong_v1b; #284 routing) + catalogue gate (#285); earlier rows (14e25ed, 69890e2) stay valid as near-equivalent",
                 "ported_pickup": "library (PR #289 merged e783de4; general_pickup rows count from it, superseding earlier ones at n >= 5)",
                 "vocab_found_needs_score": "true (#303, adopted; b4cd6c2)",
                 "boundaries": {"b4cd6c2": "#303 vocab_found_needs_score true", "3ad78f2": "#290 tile / operator glyph reads, #291 solve placeholder (solve / arrange rows count from here with the glyph marker)", "e783de4": "#289 ported_pickup library", "32dc2a9": "#288 seg_add_refit_poor (default on)", "cf3ed7f": "#287 seg_add_move_weak (default on)", "c65a9ea": "#284 glyph routing + report digit fields, #285 catalogue gate, #286 seg_propose add",
                                "14e25ed": "#280 glyph_read on", "69890e2": "#267 uniform color_map"},
                 "seg_propose": "add (#286, adopted)", "seg_add_move_weak": "on (#287, default)", "seg_add_refit_poor": "on (#288, default)",
                 "make_kong_note": "glyph_read on reads slot tiles as unknown until W / S route the glyph model by asset category", "h95": {"put_into_opts": ["spot", "patient", "sequence"], "drop_verdict": "release", "refine_recentre_guard": "vis"}, "h95_status": "adopted: M's write audit on the h95 pairs clean (28 Oct 20:43Z; hon 23 harmful / 22 helpful vs hoff 31 / 37 over 44 episodes)",
                 "perception_env": {"DOJO_PERCEPTION_CFG": {"series_count": "seg"}, "DOJO_SEG_ADDR": "per-job seg_twin v2b server"},
                 "families": "c4b_<cell>sg_<port|portoff|rb|gcon|tdc17|a1off>_<id>; treatments _toff (select_ties off), _a1on (A1 apply)",
                 "glyph_read_marker": "per-row _perception.glyph_read visible only from #284 (30db5ff); sg rows on 14e25ed..30db5ff = glyph-unknown (accepted); make_kong tiles go to DOJO_GLYPH_ADDR_KONG (kong_v1b side server)",
                 "glyph_read": "on (PR #280 merged 14e25ed, confirmed by the lead; S's digit_v2 server at DOJO_GLYPH_ADDR, model sha256 a9c341e949d0...; "
                               "numerals only: solve_equation, arrange_largest_number (+_random); per-row marker "
                               "_perception.glyph_read on the frame-0 numeral entries, else pre-glyph)",
                 "ported_pour": "full (PR #279 merged 643041f; confirmed: P's G3 check 3/5 = v1 3/5, no spill / tip; pour_balls rows count from it)",
                 "ported_drag": "turn (PR #278 merged 4417ed5; push_T rows count only from it, with ported_drag == turn)",
                 "series_count_default_flip": "#276 merged 4ad59e1 (seg only with the seg server; sg families)",
                 "preset": "submit_v3", "grip_check": "on", "grip_source": "v2", "ported_routes": "on", "ported_swap": "reach",
                 "new_routes": ["equation", "largest_number", "align", "row_by_size", "pour_order"],
                 "new_routes_pour_order": "joins from b16707e (lead): instruction-matched (pour + colour clauses, each a bottle + a bowl); "
                                          "conditional on the guard: pour_liquid_into_cup(+_random) must run pour_bottle, never pour_in_order",
                 "joint_limits": "the submit_v3 preset value (\"wide\"); H's \"measured\" is NOT adopted (lead, 29 Oct)",
                 "pad_row_extend": "true (perception; adopted 29 Oct, #298 53dd9ed8+: 5/5 vs 4/5 at n=5; 100020-24 pairs as confirmation)",
                 "kong_draw_read_v3": "REJECTED (#315 a21eaee1: 8 vs 9 on 15 layouts); #317 (d7610a7c) restores place = v2, v3 behind place3; rows a21eaee1 <= sha < d7610a7c with place excluded",
                 "kong_draw_read": "place v2 (adopted 29 Oct; #311 merged b131264e: 5/8 vs 0/8, +62.5; S's v3 comes as kongon3 vs v2)",
                 "imitate": "imitate_pairing one_to_one + imitate_wait origin (adopted 29 Oct, 17d97d6d+, lead ruling below the 10-pt bar: wait removes the early-start query-0 failure, D 3 wins / 0 losses over 8 layouts, 0.65 vs 0.20); imitate_look OFF",
                 "belt_wrist_veto": "on (adopted 30 Oct by its rule; weak evidence: 1 differing pair of 10 (+1, 0 losers); rule registered after 8 / 10 pairs were known; reaches match_and_pick + pick_from_conveyor_by_image (parked, ~0) via interactive.pick_from_belt; CANDIDATE md5 5509d7be from 08:23:50Z)",
                 "imitate_bundle": "REJECT (lead, 2026-10-30 ~08:40Z machine time): guard premise error (support robot present on make_kong); not amended because a kong score had been seen (M's gate-field inspection printed 100005: A / B / C 0 / 0 / 0); block stopped; arm C rejected with it; re-test with a newly registered score-based kong guard after D-C19 (parked rows stay unrun)",
                 "slot_read": "NOT adopted (lead, 30 Oct): coin slot rule REJECT at 11 pairs (on 29.1 vs off 20.0, net +1, but 1 success loser 100024); ttt / keyslot / blocksslot parked unrun for a later freeze",
                 "watch_relocate": "NOT adopted (lead, 30 Oct): bar not met: all 9/15 vs 6/15 (+20), unseen 2W / 1L (100021); parked; no re-test on the same ids",
                 "peg_base": "peg_base_refine local + peg_base_tries 8 (adopted 30 Oct by its pre-registered rule, #349 / #351, 27c7a5b+: pbrB 62.0 vs pbrA 41.0 over 10 pairs, worst pairs exactly -0.3 = pass; peg_stuck_confirm (pbrC) rejected: 47.8 vs 65.6, 3 pairs lose > 0.3)",
                 "ported_classify": "p2 (adopted 29 Oct, #312 7a48ac9d+: by_language +10 at 7 pairs, 0 losing layouts on success (0/0/15 ties), score 6 W / 1 L; remaining pairs = confirmation, GUARD ALERT on any success loss); p1 before it (ddf5faa8+)",
                 "series_labels": "numbered (#299, adopted 29 Oct: the belief dropped repeated labels; classify +30 at n=6, by_language 0 at n=5, sweep guards 0 / 0)",
                 "align_measure": "parallel (adopted by the lead's pre-registered rule at 10/10 pairs, 29 Oct ~04:10Z: par 7 vs seq 2, 0 seq-only wins; 91eb98f)",
                 "parked_ab": "tdc17 / toff legality variants, bowls a1, the port guards, bowls h95 on/off (lead, 29 Oct): not pending",
                 "ported_pour_aim": "b10 (adopted 29 Oct, #293 b16707e1+: pour_liquid +33.3 (3/6 vs 1/6), pour_balls guard 0, pour_liquid_random flat (misrouted; W fixing)); remaining pairs = confirmation",
                 "n_cells_from": "PR #282 merged a8349f7: N's cells count from it with new_routes (+ glyph_read for solve / arrange)",
                 "new_routes_from": "PR #268 merged dcc5e7a (lead, 28 Oct); N's 100030-49 rows stay dev upside, ceilings from v0 ids",
                 "perception": {"color_map": "all (uniform), #267", "boundary_sha": "69890e2",
                                "rule": "rows from shas >= 69890e2 are the candidate's perception; earlier rows stay valid as "
                                        "near-equivalent (offline -3 of 1023 objects), tagged pre-#267",
                                "series_count": "seg (seg_twin v2b server running; counting only)",
                                "series_count_boundary": "W's default-flip PR sha or Integration's job env (pending); classify / "
                                                         "arrange camera rows without it are pre-seg"},
                 "levers_off": ["put_into_opts", "refine_recentre_guard", "post_place_read", "a1", "wrist_check", "bg_perception",
                                "place_surface", "seg_propose (perception)", "twin_pairing (perception)"],
                 "rule": "route flags (ported_routes; later N's new_routes) fire only where the base planner says Plan none: rows "
                         "with or without them are equivalent elsewhere once the guard cells pass (lead, 28 Oct)"}
ROUTE_FLAGS = ("ported_routes", "new_routes")
# the cells where G3 printed Plan none at 173acfa (stream M g3c): only a route-flag-on row is the candidate there
PLAN_NONE = {t for (t, box, d), v in G3C.items() if box == "compute3" and all(x["plan_none"] for x in v)}
# family-name suffixes: CONTROL_SUFFIX may be the candidate (config decides); every other suffix is a treatment arm
# (perception-side flags are invisible in RuntimeConfig, so a name not on this list never counts as the candidate)
CONTROL_SUFFIX = ("off", "_gcon", "_rb", "_port", "_portr", "_hon")   # _hon: h95 adopted (M's write audit clean, 28 Oct 20:43Z)
PAIRS = {"gcon": ("gcoff", "gc"), "rs": ("rsoff", "rs"), "seg": ("segoff", "seg"), "hon": ("hoff", "h"), "emb": ("off", "emb"),
         "stereo": ("off", "stereo"), "fm": ("fmoff", "fm"), "tdc17": ("toff", "select_ties (D-C17)"),
         "jlm": ("jlw", "joint_limits measured (H; compute1, 4090 class, no seg server)")}
UNCLASSIFIED = set()
H95 = {"put_into_opts": ["spot", "patient", "sequence"], "drop_verdict": "release", "refine_recentre_guard": "vis"}
LEVER_DEFAULTS = {"put_into_opts": (None, [], (), H95["put_into_opts"]), "refine_recentre_guard": (None, "centre", "vis"),
                  "drop_verdict": (None, "belief", "release"),   # "belief" = main's default "post_place_read": (None, "off"),
                  "a1": (None, "off"), "wrist_check": (None, "off"), "bg_perception": (None, "off"), "place_surface": (None, "bbox")}
# Integration's candidate families from 28 Oct 20:05Z (compute4): c4b_<cell>sg_<suffix>_<id>; "sg" = series_count seg with the
# seg_twin v2b server checked per job. Candidate suffixes, and treatment suffixes paired with their candidate twin:
SG_CAND = ("port", "portoff", "rb", "gcon", "tdc17", "a1off", "off", "pturn", "pfull", "hon", "glyph", "plib", "seq", "aimoff",
           "numoff", "padoff", "tileoff", "v15c", "kongoff", "p2off", "paceoff", "yawoff", "keyoff", "ubgoff", "unionoff",
           "ovloff", "afixoff", "bowls3off", "pbrB", "relocoff", "bundleoff", "rsroff", "fpliftA", "rskA", "dc19", "untiltoff")   # pbrB: peg_base_refine adopted (lead, 30 Oct); pbrA retired   # every A arm = the candidate (lead, 29 Oct)   # dc19: assumed D-C19 family suffix (to agree when the block is scheduled)   # hon: h95 adopted; afixoff: the combined arrange A arm (= candidate, Integration 29 Oct)
SG_TREAT = {"toff": ("tdc17", "select_ties off"), "a1on": ("a1off", "A1 apply"), "on": ("off", None),
            "hoff": ("hon", "h95 off (re-check on the adopted candidate)"),
            "par": ("seq", "align_measure parallel (91eb98f)"),
            "aimon": ("aimoff", "ported_pour_aim b10 (#293, b16707e)"),
            "numon": ("numoff", "series_labels numbered (#299, ed44940)"),
            "padon": ("padoff", "pad_row_extend (#298, ed44940)"),
            "afixon": ("afixoff", "arrange fix: tile_grasp prefer + tile_recover + tile_look_cap + series_link_raise + glyph_upright (W, #331 2203d2c; replaces the held tile A/B)"),
            "tileon": ("tileoff", "tile_grasp prefer + tile_recover + series_link_raise (dc84e2f, #307)"),
            "nump1": ("numon", "ported_classify p1 (on the candidate, vs numon; ddf5faa)"),
            "nest": ("nestoff", "nest (H)"),
            "p2on": ("p2off", "ported_classify p2 (7a48ac9)"),
            "yawon": ("yawoff", "edge_yaw (perception; #318, 8c78457)"),
            "keyon": ("keyoff", "key_handover v2 + key_lean (#320, 7ab74f6)"),
            "ubgon": ("ubgoff", "put_into unblock + gentle (#321, e321d3d)"),
            "unionon": ("unionoff", "twin_union_pool (perception; #325, ed1a773)"),
            "bundlewid": ("bundleon", "wrist_identity support_masked + imitate_repair, over the bundle (W, #369; arms on 01c1a2b; pre-registered rule)"),
            "bundleon": ("bundleoff", "imitate bundle: support_mask + wrist_pool support_masked + imitate_wait home + imitate_keep_sides (W; arms on 01c1a2b; pre-registered rule)"),
            "rskB": ("rskA", "rehover_siblings keep + sibling_order far + press_repeat cross (H; arms on c10d765; make_toast + guards; pre-registered rule)"),
            "fpliftC": ("fpliftB", "fill_joint_limits measured, over fpliftB (H; arms on be7a798; fill_pen_holder v15; pre-registered rule)"),
            "fpliftB": ("fpliftA", "fill_move_lift clear (H, #360; arms on be7a798; fill_pen_holder v15; pre-registered rule)"),
            "untilton": ("untiltoff", "ported_pour_untilt capped (lead pre-registered 12:30Z 30 Oct; pour_balls; arms on 62f43590)"),
            "rsron": ("rsroff", "row_size_read look_union (N, #358; arms on 4654086; dolls x2; pre-registered rule)"),
            "pbrC": ("pbrB", "peg_stuck_confirm on, over pbrB (S, #355, 27c7a5b; pre-registered rule)"),
            "pbrB": ("pbrA", "peg_base_refine local + peg_base_tries 8 (S, #349 / #351; arms on 27c7a5b; pre-registered rule)"),
            "bowls3on": ("bowls3off", "stack_reach grasp + nest_height measured + nest_verify support (H, #339 680ab86; pre-registered rule)"),
            "relocon": ("relocoff", "watch_relocate (W, #337 5194d42; make_kong; class c: report split)"),
            "exp": ("vetonly", "track_expire on (#335 7722b0b; vs vetonly = candidate + belt_wrist_veto, same sha)"),
            "ovlres": ("ovloff", "ported_pour_pace overlap on #332 (4f90f97, pour last untilt; B only, vs ovloff on 53f1043, a documented cross-sha exception)"),
            "ovlon": ("ovloff", "ported_pour_pace overlap (#327, 53f1043)"),
            "paceon": ("paceoff", "pour_by_language pace: b10 + lean together (H, #313; A arm without b10)"),
            "kongon": ("kongoff", "kong_draw_read place v1 (S branch 08ea836; first-version reference, not adoptable)"),
            "kongon2": ("kongoff", "kong_draw_read place v2 (#311 merged, + second wrist view; vs kongoff on 08ea836, a documented cross-sha exception)"),
            "kongon3": ("kongon2", "kong_draw_read v3 (#315 a21eaee1, in place; vs adopted v2 on the same 15 layouts)"),
            "o2o": ("gcon", "imitate o2o (#305)"), "o2otwin": ("gcon", "imitate o2o + twin (#305)"),
            "o2owait": ("gcon", "imitate o2o + wait origin (#322/#323, c86dc91; vs gcon on d2cd632, cross-sha exception)"),
            "o2owaitlook": ("gcon", "imitate o2o + wait origin + look ambiguous (#322/#323, c86dc91; vs gcon, cross-sha)")}
NOARM = {"packon": "new_routes pack (#329, 516c262; vs Plan none = 0 by construction, no A arm)"}   # treatment-only arms on Plan-none cells
SG_ON_LEVER = {"c4b_matchvetosg": "belt_wrist_veto on (3651b68)",
               **{f"c4b_{x}slotsg": "slot_read + held_thin + thin_slot_keep + slot_settle_merged on (dfb8169, #308)" for x in ("coin", "key", "ttt", "blocks")}}     # generic _on / _off pairs: lever named by the family


def sg_family(f):
    pre, suf = f.rsplit("_", 1)
    return pre.endswith("sg"), suf
GUARDS = {"swap_blocks", "stack_blocks", "put_bottles_into_dustbin"}


def plain(v):        # no RuntimeConfig lever on (perception-side arms are told apart by the family name)
    return all((r.get("cfg") or {}).get(k) in ok for r in v for k, ok in LEVER_DEFAULTS.items())


def routes_on(r):     # ported_routes: bool True / "on"; new_routes: a non-empty list of route names
    c = r.get("cfg") or {}
    return c.get("ported_routes") in (True, "on", 1, "1", "true") or bool(c.get("new_routes"))


def candidate(f, v):
    t = v[0]["task"]
    if f.startswith("c1:"):
        return False
    sg, suf = sg_family(f)
    if sg and suf not in SG_CAND:
        return False
    if not sg and not f.endswith(CONTROL_SUFFIX):
        UNCLASSIFIED.add(f) if f.rsplit("_", 1)[-1] not in PAIRS and f.rsplit("_", 1)[-1] not in ("jlw",) else None
        return False
    return (plain(v) and all(
        r.get("grip_source") == "v2" and (r.get("cfg") or {}).get("grip_check") == "on"
        and (r.get("cfg") or {}).get("preset") == "submit_v3" and (t not in PLAN_NONE or routes_on(r))
        and (t != "swap_T" or (r.get("cfg") or {}).get("ported_swap") == "reach") for r in v))   # PR #265: reach only matters there




SEG_SHA = "4ad59e1"               # W's series_count default flip (#276); seg only with the server, so not used alone
SEG_FAMILIES = set()              # Integration families whose job env set series_count=seg (DOJO_PERCEPTION_CFG)
GLYPH_CELLS = {"solve_equation", "arrange_largest_number", "arrange_largest_number_random"}
SEG_CELLS = {"classify_objects", "classify_objects_by_language", "arrange_largest_number", "arrange_largest_number_random"}


def has_seg(f, sha, rows_=()):     # the per-row record marker wins where present (W); else the sg family name
    marks = [(r.get("cfg") or {}).get("seg_counted") for r in rows_]
    if any(m is not None for m in marks):
        return all(m is not False for m in marks)
    return sg_family(f)[0] or f in SEG_FAMILIES


# series_labels "numbered" adopted (lead, 29 Oct; #299, ed44940f+): _numon is the candidate row, _numoff retires
NUMON_CELLS = {"classify_objects", "classify_objects_by_language", "sweep_blocks_random"}
P1_CELLS = {"classify_objects", "classify_objects_by_language"}
AIM_CELLS = {"pour_liquid_into_cup", "pour_liquid_into_cup_random", "pour_balls_into_vase"}   # ported_pour_aim b10 adopted     # ported_classify p1 adopted (lead, 29 Oct; ddf5faa8+)
# lead's pre-registered align rule (28-29 Oct): over the 10 pairs (100005-09 + 100020-24), ADOPT align_measure "parallel" if
# (a) par successes >= seq successes and (b) at most one layout where seq succeeds and par fails
ALIGN_TARGET = (set(range(100005, 100010)) | set(range(100020, 100025))) - {lid for (t_, lid) in TUNED_ON if t_ == "align_blocks"}
_seq, _par = fam.get(("align_blocks", "c4b_alignsg_seq"), {}), fam.get(("align_blocks", "c4b_alignsg_par"), {})
_common = sorted(set(_seq) & set(_par) & ALIGN_TARGET)
_ss = sum(bool(_seq[i]["success"]) for i in _common); _ps = sum(bool(_par[i]["success"]) for i in _common)
_lost = sum(1 for i in _common if _seq[i]["success"] and not _par[i]["success"])
ALIGN_RULE = dict(pairs=len(_common), seq=_ss, par=_ps, seq_only=_lost,
                  status=("pending" if len(_common) < len(ALIGN_TARGET) else
                          "ADOPT" if (_ps >= _ss and _lost <= 1) else "REJECT"))
ALIGN_ADOPT = True      # settled: adopted by the pre-registered rule at 10/10 (29 Oct ~04:10Z); the rule is shown on the remaining pairs
print("ALIGN RULE:", ALIGN_RULE)
# I's fidelity audit (29 Oct): measurement layouts (100005-09 + 100020-24) regenerated with today's generator (v15) differ for
# these cells; their camera rows sit on older-generator layouts (tag only; the lead decides on any re-measure)
OLDGEN = {"push_T": "v1", "stack_blocks": "v1", "stack_bowls": "v1", "build_tower": "v1", "put_bottles_into_dustbin": "v1",
          "pack_objects_into_box_random": "v1", "fill_pen_holder": "v2", "insert_key": "v2", "plug_in_charger": "v2",
          "store_tools_in_toolbox": "v2", "swap_blocks": "v2", "fill_egg_holder": "v6", "make_kong": "v7 (assets only)",
          "match_and_pick_from_conveyor": "v11 (7/10 differ)", "pick_from_conveyor_by_image": "v13 (4/10 differ)"}
# lead (29 Oct): #315 (a21eaee1) changes kong_draw_read "place" in place (v3). On the same 15 layouts: kongon3 >= kongon2 (successes)
# -> kongon3 becomes make_kong's candidate row; fewer successes -> GUARD ALERT
_k3, _k2 = fam.get(("make_kong", "c4b_makekongsg_kongon3"), {}), fam.get(("make_kong", "c4b_makekongsg_kongon2"), {})
_kc = sorted(set(_k3) & set(_k2))
KONG3 = dict(pairs=len(_kc), of=len(_k2), v3=sum(bool(_k3[i]["success"]) for i in _kc), v2=sum(bool(_k2[i]["success"]) for i in _kc))
KONG3["status"] = ("pending" if not _k2 or len(_kc) < len(_k2) else "SWITCH" if KONG3["v3"] >= KONG3["v2"] else "WORSE")
KONG3_SWITCH = False                # v3 rejected (lead, 29 Oct): kongon2 stays make_kong's candidate row
best, THIN, BOX4090 = {}, {}, {}
ALLC = collections.defaultdict(list)          # cell -> every (family, agg, rank) offered as its row


def _offer(t, f, a, rank, key):
    ALLC[t].append((f, a, rank))
    if t not in best or key > best[t][3]:
        best[t] = (f, a, rank, key)
PRECISION = {c for c, dim, w in cells() if dim == "Precision"}
for (t, f), v in fam.items():
    rows_ = list(v.values())
    if t == "imitate_sorting_sequence":
        rows_ = [x for x in rows_ if x.get("layout_id") != 100020]   # lead: imitate 100020 is unstable, never counted (any family)
    if t == "match_and_pick_from_conveyor" and f.startswith("v15:c4b_matchvetosg_"):
        # belt_wrist_veto ON adopted (lead, 30 Oct): veto-on families are the candidate (on b4cd6c2, vetonly 7722b0b); off / exp are not
        if f.rsplit("_", 1)[-1] not in ("on", "vetonly"):
            continue
        rank = 6 if len(rows_) >= 5 else 0
        a = agg(rows_); key = (rank, a["n"] >= 5, a["mtime"])
        _offer(t, f, a, rank, key)
        if rank == 0 and a["n"] < 5 and (t not in THIN or a["mtime"] > THIN[t][1]["mtime"]):
            THIN[t] = (f, a)
        continue
    # a candidate row replaces older rows once it has >= 5 episodes; a thinner candidate row is "in progress" (shown, and
    # counted only where the cell has no other row); flag-off controls rank in between
    if t == "imitate_sorting_sequence" and sg_family(f)[0] and f.rsplit("_", 1)[-1] in ("gcon", "o2o", "o2otwin", "o2owait", "o2owaitlook", "rb"):
        if f.rsplit("_", 1)[-1] != "o2owait":
            continue                        # retired: imitate_pairing one_to_one + imitate_wait origin adopted (lead, 29 Oct; 17d97d6d+)
        rows_ = [x for x in rows_ if x.get("layout_id") != 100020]     # lead: n = 8, 100020 excluded (unstable)
        rank = 5 if len(rows_) >= 5 else 0
        a = agg(rows_); key = (rank, a["n"] >= 5, a["mtime"])
        _offer(t, f, a, rank, key)
        if rank == 0 and a["n"] < 5 and (t not in THIN or a["mtime"] > THIN[t][1]["mtime"]):
            THIN[t] = (f, a)
        continue
    if t in AIM_CELLS and sg_family(f)[0] and f.rsplit("_", 1)[-1] in ("aimoff", "aimon", "pfull", "port", "rb", "rt", "unionoff"):
        suf = f.rsplit("_", 1)[-1]
        if suf == "unionoff":
            suf = "rt"                      # #325 A arm (ed1a773) = the fixed-routing candidate row, replacing _rt
        if suf not in ("aimon", "rt"):
            continue                        # retired: ported_pour_aim b10 adopted (lead, 29 Oct; #293, b16707e1+)
        if suf == "rt" and not all(post_boundary(x.get("head"), "c54ef1c") for x in rows_):
            continue
        # pour_liquid_random: rows before #319 (c54ef1c6) were misrouted to pour_by_language; _rt (fixed routing) supersedes at n >= 5
        rank = (6 if suf == "rt" else 5) if len(rows_) >= 5 else 0
        a = agg(rows_); key = (rank, a["n"] >= 5, a["mtime"])
        _offer(t, f, a, rank, key)
        if rank == 0 and a["n"] < 5 and (t not in THIN or a["mtime"] > THIN[t][1]["mtime"]):
            THIN[t] = (f, a)
        continue
    if t == "solve_equation" and sg_family(f)[0] and f.rsplit("_", 1)[-1] in ("padoff", "padon", "glyph"):
        if f.rsplit("_", 1)[-1] != "padon":
            continue                        # retired: pad_row_extend adopted (lead, 29 Oct; #298, 53dd9ed8+)
        rank = 5 if len(rows_) >= 5 else 0
        a = agg(rows_); key = (rank, a["n"] >= 5, a["mtime"])
        _offer(t, f, a, rank, key)
        if rank == 0 and a["n"] < 5 and (t not in THIN or a["mtime"] > THIN[t][1]["mtime"]):
            THIN[t] = (f, a)
        continue
    if t == "make_kong" and sg_family(f)[0] and f.rsplit("_", 1)[-1] in ("kongoff", "kongon", "kongon2", "kongon3", "rb"):
        if f.rsplit("_", 1)[-1] != ("kongon3" if KONG3_SWITCH else "kongon2"):
            continue                        # retired: kong_draw_read place v2 adopted (lead, 29 Oct; #311 b131264e)
        rank = 5 if len(rows_) >= 5 else 0
        a = agg(rows_); key = (rank, a["n"] >= 5, a["mtime"])
        _offer(t, f, a, rank, key)
        if rank == 0 and a["n"] < 5 and (t not in THIN or a["mtime"] > THIN[t][1]["mtime"]):
            THIN[t] = (f, a)
        continue
    if t in P1_CELLS and sg_family(f)[0] and f.rsplit("_", 1)[-1] in ("numon", "numoff", "nump1", "p2off", "p2on"):
        if f.rsplit("_", 1)[-1] != "p2on":
            continue                        # retired: ported_classify p2 adopted on classify x2 (lead, 29 Oct; #312 7a48ac9d+)
        rank = 5 if len(rows_) >= 5 else 0
        a = agg(rows_); key = (rank, a["n"] >= 5, a["mtime"])
        _offer(t, f, a, rank, key)
        if rank == 0 and a["n"] < 5 and (t not in THIN or a["mtime"] > THIN[t][1]["mtime"]):
            THIN[t] = (f, a)
        continue
    if t in NUMON_CELLS and sg_family(f)[0] and f.rsplit("_", 1)[-1] in ("numon", "numoff"):
        if f.rsplit("_", 1)[-1] == "numoff":
            continue                        # retired: series_labels numbered adopted (lead, 29 Oct)
        rank = 5 if len(rows_) >= 5 else 0
        a = agg(rows_); key = (rank, a["n"] >= 5, a["mtime"])
        _offer(t, f, a, rank, key)
        if rank == 0 and a["n"] < 5 and (t not in THIN or a["mtime"] > THIN[t][1]["mtime"]):
            THIN[t] = (f, a)
        continue
    if t == "align_blocks" and ALIGN_ADOPT:
        if f in ("c4b_alignsg_par", "c4b_alignsg_yawoff"):      # yawoff (8c78457, #318 A arm) = the align candidate on that sha
            rank = 5 if len(rows_) >= 5 else 0
            a = agg(rows_); key = (rank, a["n"] >= 5, a["mtime"])
            _offer(t, f, a, rank, key)
            continue
        if f in ("c4b_alignsg_seq", "c4b_alignblockssg_port") and not all(
                (x.get("cfg") or {}).get("align_measure") == "parallel" for x in rows_):
            continue                        # retired on adoption (rows without align_measure parallel)
    if candidate(f, rows_):
        rank = (3 if sg_family(f)[0] else 2.5 if f.endswith("_hon") else 2) if len(rows_) >= MIN_N.get(t, (5,))[0] else 0
        if t == "general_pickup" and rank and all(post_boundary(x.get("head"), PICKUP_SHA)
                                                  and (x.get("cfg") or {}).get("ported_pickup") == "library" for x in rows_):
            rank = 4            # #289: library pickup rows supersede earlier general_pickup rows once n >= 5
    elif (f.endswith("off") or f.endswith("_rb")) and plain(rows_) and not f.startswith("c1:"):
        rank = 1                            # compute1 families are lever-only, never even the flag-off fallback row
    else:
        continue
    if t in V15_CELLS and f.startswith("v15:") and rank >= 2:
        rank = 6                            # v15 layouts replace old-gen rows once n >= 5 (lead, 29 Oct)
    a = agg(rows_)
    if t in PRECISION and all(x.get("box") == "c1" for x in rows_):
        BOX4090[t] = (f, a)
        rank = min(rank, 0)            # 4090 class: shown, never displaces a 5090 row (lead, 28 Oct)
    key = (rank, a["n"] >= MIN_N.get(t, (5,))[0], a["mtime"])
    _offer(t, f, a, rank, key)
    if rank == 0 and a["n"] < 5 and (t not in THIN or a["mtime"] > THIN[t][1]["mtime"]):
        THIN[t] = (f, a)
# Misha's point (b), lead 29 Oct: any post-#267 candidate row with n >= 5 replaces an older row (a pre-#267, non-candidate or thin one)
_post267 = lambda a: bool(a["runs"]) and all(post_boundary(x.get("head"), "69890e2") for x in a["runs"])  # noqa: E731
# lead (29 Oct, generalised): the counted row = the NEWEST post-#267 candidate row with n >= 5, whichever way it moves the number;
# every A arm of an A/B is the candidate (relocoff, bowls3off, pbrA, smaskoff, matchveto off, coin off, ...). Retired arms (rows
# without an adopted lever) are never offered; v15 rows stay preferred on the v15-set cells.
RETIRED_SUF = {"portoff", "aimoff", "numoff", "padoff", "p2off", "kongoff", "seq", "toff", "hoff", "gcoff", "segoff", "rsoff", "fmoff",
               "pbrA", "pbrC", "relocon", "bundleon", "bundlewid", "untilton"}   # untilton: rejected (pour_balls 100006 lost; rule below)   # relocon: watch_relocate parked (lead, 30 Oct); relocoff stays the make_kong candidate row   # pbrA: lacks the adopted peg_base_refine; pbrC: rejected (lead, 30 Oct)
POST267_SWAP = {}
# lead (2026-10-30 11:25Z machine time): precedence = newest CODE first (git ancestry), then row time, whichever way it moves the number;
# where neither sha is an ancestor of the other, the one whose merge-base with main is nearer main wins (noted in NONLINEAR)
import functools
_MB = {}
def _main_dist(h):                    # commits from merge-base(h, origin/main) to origin/main: smaller = nearer main
    if h not in _MB:
        r = subprocess.run(["git", "-C", str(ARGS.repo), "merge-base", h, ARGS.main_ref], capture_output=True, text=True)
        mb = r.stdout.strip()
        c = subprocess.run(["git", "-C", str(ARGS.repo), "rev-list", "--count", f"{mb}..{ARGS.main_ref}"],
                           capture_output=True, text=True).stdout.strip() if mb else ""
        _MB[h] = int(c) if c.isdigit() else 10 ** 6
    return _MB[h]
NONLINEAR = []
def _code_cmp(h1, h2):
    if h1 == h2:
        return 0
    if post_boundary(h2, h1):          # h1 is an ancestor of h2 -> h2 newer
        return -1
    if post_boundary(h1, h2):
        return 1
    d1, d2 = _main_dist(h1), _main_dist(h2)
    NONLINEAR.append(f"{h1} vs {h2}: merge-base distance to main {d1} vs {d2}")
    return (d1 < d2) - (d1 > d2)
def _fam_code(a):                     # the newest code among a family's rows
    hs = sorted({str(x.get("head"))[:7] for x in a["runs"] if x.get("head")})
    return max(hs, key=functools.cmp_to_key(_code_cmp)) if hs else ""
def _cmp_offer(t, x, y):
    vx = t in V15_CELLS and x[0].replace("c1:", "").startswith("v15:"); vy = t in V15_CELLS and y[0].replace("c1:", "").startswith("v15:")
    if vx != vy:
        return 1 if vx else -1        # v15 rows stay preferred on the v15-set cells
    c = _code_cmp(_fam_code(x[1]), _fam_code(y[1]))
    if c:
        return c
    return (x[1]["mtime"] > y[1]["mtime"]) - (x[1]["mtime"] < y[1]["mtime"])
METHODOLOGY = ["2026-10-30 11:25Z (machine time, lead): counted-row precedence = newest code (git ancestry; non-linear -> merge-base nearer "
               "main), then row time, whichever way it moves the number (was: row time)"]
for t, offers in ALLC.items():
    cur_ = best.get(t)
    good = [o for o in offers if o[2] >= 2 and o[1]["n"] >= MIN_N.get(t, (5,))[0] and _post267(o[1]) and not o[0].startswith("c1:")
            and o[0].rsplit("_", 1)[-1] not in RETIRED_SUF]
    if not good:
        continue
    f, a, rank = max(good, key=functools.cmp_to_key(lambda x, y: _cmp_offer(t, x, y)))
    if cur_ and cur_[0] == f:
        continue
    POST267_SWAP[t] = f"{cur_[0] if cur_ else 'none'} -> {f}"
    best[t] = (f, a, rank, (rank, True, a["mtime"]))
for t, (f, a, rank, _k) in best.items():
    if rank == 0:                       # a thin candidate row (n < 5) never enters the headline: in progress only (THIN)
        continue
    oldgen_note = ", old-gen (layouts gen {})".format(OLDGEN.get(t, "?"))
    P[t] = dict(date=time.strftime("%m-%d", time.gmtime(a["mtime"])), src=f"{f} (5.1{', b4cd6c2+' if post_boundary(a['sha'], 'b4cd6c2') else ', 3ad78f2+' if post_boundary(a['sha'], '3ad78f2') else ', e783de4+' if post_boundary(a['sha'], 'e783de4') else ', 32dc2a9+' if post_boundary(a['sha'], '32dc2a9') else ', cf3ed7f+' if post_boundary(a['sha'], 'cf3ed7f') else ', c65a9ea+' if post_boundary(a['sha'], 'c65a9ea') else ''}{'' if post_boundary(a['sha']) else ', pre-#267'}{', pre-seg' if t in SEG_CELLS and not has_seg(f, a['sha'], list(fam[(t, f)].values())) else ''}{(', pre-glyph' if any(glyph_status(x) == 'pre' for x in fam[(t, f)].values()) else ', glyph-unknown (14e25ed..30db5ff)' if any(glyph_status(x) == 'unknown' for x in fam[(t, f)].values()) else '') if t in GLYPH_CELLS else ''}{', pre-#278' if t in PORTED_ALL and (GRIPFIX_SHA is None or not post_boundary(a['sha'], GRIPFIX_SHA)) else ''}{(', v15 layouts' if f.startswith('v15:') else oldgen_note) if t in V15_CELLS else (f', layouts gen {OLDGEN[t]}' if t in OLDGEN else '')}{', 4090 class (compute1)' if t in BOX4090 and BOX4090[t][0] == f else ''}{(', candidate (v15)' if f.startswith('v15:') else ', candidate (pour routing fixed #319)') if rank == 6 else ((', candidate (pour routing fixed #319)' if f.endswith(('_rt', '_unionoff')) else ', candidate (ported_pour_aim b10, adopted)') if t in AIM_CELLS else ', candidate (imitate o2o + wait origin, adopted)' if t == 'imitate_sorting_sequence' else ', candidate (pad_row_extend, adopted)' if t == 'solve_equation' else ', candidate (kong_draw_read place v2, adopted)' if t == 'make_kong' else ', candidate (ported_classify p2, adopted)' if t in P1_CELLS else ', candidate (series_labels numbered, adopted)' if t in NUMON_CELLS else ', candidate (align_measure parallel, adopted)') if rank == 5 else ', candidate (pickup library, e783de4+)' if rank == 4 else ', candidate (sg, 20:05Z)' if rank == 3 else ', candidate (h95 on)' if rank == 2.5 else ', candidate' if rank == 2 else ', candidate, IN PROGRESS n<5' if rank == 0 else ', NOT candidate: grip_check off'}"
                + (f", MIXED catalogs {'/'.join(a['catalogs'])}" if len(a["catalogs"]) > 1 else "")
                + (", on 100000-04 (dev set; kept until 100005-09 rows replace it)"
                   if any(100000 <= lid <= 100004 for lid in fam[(t, f)]) else "") + ")",
                sha=a["sha"], n=a["n"], succ=a["succ"], score=a["score"], candidate=rank not in (1,), sg=rank == 3, catalogs=a["catalogs"],
                runs=a["runs"])       # the counted episodes (W's failure census, 29 Oct)
    if t in MIN_N and a["n"] < 5:
        P[t]["src"] += f" [{MIN_N[t][1]}]"
    if t in SEED_PROTO_CELLS:           # lead (30 Oct, S's rehearsal finding): no number change, a note only
        P[t]["src"] += " [integ seed protocol != official; D-C19 not comparable]"
    # env boundary (lead, 30 Oct): DOJO_COLOR_CALIB on every row started from 03:08:48Z (pbrA_100005; expected no-op). Rows carry no
    # start time; the re-baseline finished before the boundary, so a row that ENDED after it started after it
    # (a job ends <= ~30 min after it starts: ended before T = before; ended >= T + 30 min = after; in between = before only for the
    # re-baseline families, which Integration confirmed all started before T; else unknown)
    def _ccs(x, f_=f):
        if x["mtime"] < 1793329728:
            return "before"
        if x["mtime"] >= 1793329728 + 1800:
            return "after"
        return "before" if f_.replace("v15:", "").rsplit("_", 1)[0] in REBASE_PRE else "unknown"
    _cc = {_ccs(x) for x in fam[(t, f)].values()} if (t, f) in fam else set()
    P[t]["color_calib_env"] = ("all" if _cc == {"after"} else "none" if _cc <= {"before"} else "unknown" if "unknown" in _cc else "mixed")
    if P[t]["color_calib_env"] in ("mixed", "unknown"):
        P[t]["src"] += f" [{P[t]['color_calib_env']}: rows around the DOJO_COLOR_CALIB boundary]"
    # strict-headline flags (Misha's recomputation, lead 29 Oct): post-#267, candidate, measurement slice, one catalogue, v15 layouts
    P[t].update(post267=_post267(a), cand=rank >= 2, devset=any(100000 <= (x["layout_id"] or 0) <= 100004 for x in a["runs"]),
                mixed=len(a["catalogs"]) > 1, oldgen=(t in V15_CELLS and not f.replace("c1:", "").startswith("v15:")))
    _ex = EXCL_TUNED.get((t, f))
    if _ex:                             # point (a): the un-excluded rate next to the counted (D-C10c-excluded) one
        _un = agg(list(fam[(t, f)].values()) + [x for l_, x in _ex.items() if l_ not in fam[(t, f)]])
        P[t]["unexcluded"] = dict(n=_un["n"], succ=_un["succ"], score=_un["score"])
        P[t]["src"] += f" [un-excluded {_un['succ']}/{_un['n']} / {_un['score']}]"
# guard cells: a route-flag-on row must behave like the candidate row without the flag on the same layout
GUARD_ALERTS, GUARD_OK = [], []
for t in GUARDS:                  # P's guard pairs only: _port (routes on) vs _portoff (routes off), same box, same layout id
    on = {lid: r for (t2, f), v in fam.items() if t2 == t and f.rsplit("_", 1)[-1] == "port" for lid, r in v.items()}
    off = {lid: r for (t2, f), v in fam.items() if t2 == t and f.rsplit("_", 1)[-1] == "portoff" for lid, r in v.items()}
    for lid in sorted(set(on) & set(off)):
        a, b = on[lid], off[lid]
        if a.get("box") != b.get("box"):
            continue
        (GUARD_ALERTS if (bool(a["success"]) != bool(b["success"]) or abs((a["score"] or 0) - (b["score"] or 0)) > 0.1)
         else GUARD_OK).append(f"{t} {lid}: on {a['run']} {a['success']}/{a['score']} vs off {b['run']} {b['success']}/{b['score']}")
# pour_order guard (lead): on pour_liquid_into_cup(+_random) the route must never fire (the ported pour_bottle runs there)
for r in ROWS:
    if r.get("task") in ("pour_liquid_into_cup", "pour_liquid_into_cup_random") and "pour_in_order" in str(r.get("skill")):
        GUARD_ALERTS.append(f"pour_order fired on {r['task']}: {r.get('box')}:{r['run']} skill={r.get('skill')} (ruling flips to a lever)")
# lead: #314 (bb0e14e) changes align_measure "parallel" in place (failure path only, no A/B): flag if align's candidate rows on
# shas >= bb0e14e fall below the adopted 7/10
_al = {lid: r for (t2, f), v in fam.items() if t2 == "align_blocks" and f.rsplit("_", 1)[-1] not in ("yawon",)
       for lid, r in v.items() if (r.get("cfg") or {}).get("align_measure") == "parallel" and post_boundary(r.get("head"), "bb0e14e")}
ALIGN314 = dict(n=len(_al), succ=sum(bool(r["success"]) for r in _al.values()))
if ALIGN314["n"] >= 5 and ALIGN314["succ"] / ALIGN314["n"] < 0.7:
    GUARD_ALERTS.append(f"align_blocks on >= bb0e14e (#314): {ALIGN314['succ']}/{ALIGN314['n']}, below the adopted 7/10")
_pon, _poff = fam.get(("solve_equation", "c4b_solveequationsg_padon"), {}), fam.get(("solve_equation", "c4b_solveequationsg_padoff"), {})
_pc = sorted(set(_pon) & set(_poff) & set(range(100020, 100025)))
PAD_CONFIRM = dict(pairs=len(_pc), on=sum(bool(_pon[i]["success"]) for i in _pc), off=sum(bool(_poff[i]["success"]) for i in _pc),
                   on_score=round(100 * sum(_pon[i]["score"] or 0 for i in _pc) / max(1, len(_pc)), 1),
                   off_score=round(100 * sum(_poff[i]["score"] or 0 for i in _pc) / max(1, len(_pc)), 1))
if _pc and (PAD_CONFIRM["on"] < PAD_CONFIRM["off"] or PAD_CONFIRM["on_score"] < PAD_CONFIRM["off_score"]):
    GUARD_ALERTS.append(f"pad_row_extend confirmation (100020-24): padon {PAD_CONFIRM['on']}/{len(_pc)} {PAD_CONFIRM['on_score']} "
                        f"below padoff {PAD_CONFIRM['off']}/{len(_pc)} {PAD_CONFIRM['off_score']}")
# v3 rejected by the lead: the WORSE status is settled, no longer an alert; flag any non-kongon3 make_kong row in the v3 window
for x in KONG_V3_OTHER:
    GUARD_ALERTS.append(f"make_kong row in the rejected-v3 window (excluded): {x}")
_aon, _aoff = fam.get(("pour_liquid_into_cup", "c4b_pourliquidsg_aimon"), {}), fam.get(("pour_liquid_into_cup", "c4b_pourliquidsg_aimoff"), {})
_ac = sorted(set(_aon) & set(_aoff))
AIM_CONFIRM = dict(pairs=len(_ac), on=sum(bool(_aon[i]["success"]) for i in _ac), off=sum(bool(_aoff[i]["success"]) for i in _ac))
if len(_ac) >= 10 and AIM_CONFIRM["on"] < AIM_CONFIRM["off"]:
    GUARD_ALERTS.append(f"ported_pour_aim confirmation (pour_liquid, {len(_ac)} pairs): aimon {AIM_CONFIRM['on']} < aimoff {AIM_CONFIRM['off']}")
POUR_GUARD = sorted({f"{r['task']}:{r.get('skill')}" for r in ROWS if r.get("task") in ("pour_liquid_into_cup", "pour_liquid_into_cup_random")
                     and "aim" in r["run"]})
# lead (29 Oct): #330 (384a28df) is a runtime fix without an A/B (belief.Divergence in make_skills' resync re-plans instead of
# ending as "error"). Flag a cell whose candidate rows on shas >= 384a28df drop versus its earlier candidate rows (paired by layout).
DIV_SHA = "384a28d"
ADOPTED_SUF = ("o2owait", "padon", "kongon2", "aimon", "numon", "nump1", "p2off", "p2on", "par", "yawoff", "rt", "unionoff", "v15c", "plib")
DIV330 = {}
for (t, f), allr in fam_all.items():
    if (t, f) not in fam or f.startswith("c1:"):
        continue
    if not (candidate(f, list(fam[(t, f)].values())) or (sg_family(f)[0] and f.rsplit("_", 1)[-1] in ADOPTED_SUF)):
        continue
    d = DIV330.setdefault(t, {"pre": {}, "post": {}})
    for r in allr:
        side = d["post" if post_boundary(r.get("head"), DIV_SHA) else "pre"]
        o = side.get(r["layout_id"])
        if o is None or r["mtime"] > o["mtime"]:
            side[r["layout_id"]] = r
DIV330_SUMMARY = {}
for t, d in sorted(DIV330.items()):
    if not d["post"]:
        continue
    pc = sorted(set(d["pre"]) & set(d["post"]))
    s_ = lambda v: dict(n=len(v), succ=sum(bool(x["success"]) for x in v), score=round(100 * sum(x["score"] or 0 for x in v) / max(1, len(v)), 1),  # noqa: E731
                        errors=sum(x.get("category") == "error" for x in v))
    pre, post = s_([d["pre"][i] for i in pc]), s_([d["post"][i] for i in pc])
    DIV330_SUMMARY[t] = dict(pairs=len(pc), pre=pre, post=post, post_all=s_(list(d["post"].values())),
                             post_runs=sorted(f"{x.get('box')}:{x['run']}" for x in d["post"].values()))
    if len(pc) >= 3 and (post["succ"] < pre["succ"] or post["score"] < pre["score"] - 10):
        GUARD_ALERTS.append(f"#330 (384a28d) {t}: candidate rows on >= 384a28d {post['succ']}/{len(pc)} {post['score']} vs earlier "
                            f"{pre['succ']}/{len(pc)} {pre['score']} on the same layouts (errors {pre['errors']} -> {post['errors']})")
# lead (30 Oct): code-drift audit before the freeze. Per cell, the NEWEST-sha candidate rows vs the latest OLDER-sha candidate row on the
# same layout ids, flags matched on the cell's relevant RuntimeConfig keys; a sign test across cells. A arms of current A/Bs count as
# candidate rows (pbrA too: the pre-adoption candidate); B arms, rejected / parked arms and dev rows never do. No action unless negative.
LEVER_SCOPE = {"ported_classify": {"classify_objects", "classify_objects_by_language"},
               "series_labels": {"classify_objects", "classify_objects_by_language", "sweep_blocks", "sweep_blocks_random"},
               "imitate_pairing": {"imitate_sorting_sequence"}, "imitate_wait": {"imitate_sorting_sequence"}, "imitate_look": {"imitate_sorting_sequence"},
               "kong_draw_read": {"make_kong"}, "align_measure": {"align_blocks"},
               "ported_pour_aim": AIM_CELLS | {"pour_by_language"}, "ported_pour": {"pour_balls_into_vase"}, "ported_pour_pace": {"pour_by_language"},
               "ported_pickup": {"general_pickup"}, "ported_drag": {"push_T", "push_T_random"}, "ported_swap": {"swap_T"},
               "belt_wrist_veto": {"match_and_pick_from_conveyor", "pick_from_conveyor_by_image"}, "slot_read": {"deposit_coin"}, "held_thin": {"deposit_coin"},
               "key_handover": {"insert_key"}, "key_lean": {"insert_key"}, "tile_grasp": GLYPH_CELLS, "tile_recover": GLYPH_CELLS}
DRIFT_KEYS = ["preset", "grip_check", "put_into_opts", "refine_recentre_guard", "drop_verdict", "joint_limits", "seg_propose",
              "ported_routes", "select_ties", "vocab_found_needs_score"] + list(LEVER_SCOPE)
_DRIFT_NEVER = {"pbrC", "pbrB", "relocon", "smaskon", "bundleon", "bundlewid", "rsron", "fpliftB", "fpliftC", "rskB", "afixon", "bowls3on",
                "vetonly", "exp", "on", "keyon", "ubgon", "unionon", "yawon", "ovlon", "ovlres", "paceon", "numon", "p2on", "nump1"}
def _drift_cand(t, f):
    suf = f.rsplit("_", 1)[-1]
    if f.startswith("c1:") or suf in _DRIFT_NEVER and not (t in P1_CELLS and suf == "p2on") and not (t in NUMON_CELLS and suf == "numon"):
        return False
    return (candidate(f, list(fam[(t, f)].values())) or (sg_family(f)[0] and suf in ADOPTED_SUF + ("pbrA", "relocoff")))
def _relevant(t, k):
    return k not in LEVER_SCOPE or t in LEVER_SCOPE[k] or (k == "new_routes")
DRIFT = {}
for t in {k[0] for k in fam}:
    # pair only within one layout generator: the same id on v15 and old-gen layouts is a different scene
    rows_t = [dict(r, _lk=(f.replace("c1:", "").startswith("v15:"), r["layout_id"]))
              for (t2, f), v in fam_all.items() if t2 == t and (t2, f) in fam and _drift_cand(t2, f) for r in v]
    if not rows_t:
        continue
    by_sha = collections.defaultdict(list)
    for r in rows_t:
        by_sha[str(r.get("head"))[:7]].append(r)
    newest = max(by_sha, key=lambda h: max(x["mtime"] for x in by_sha[h]))
    new = {}
    for r in by_sha[newest]:
        if r["_lk"] not in new or r["mtime"] > new[r["_lk"]]["mtime"]:
            new[r["_lk"]] = r
    _sig = lambda r: tuple(str((r.get("cfg") or {}).get(k)) for k in DRIFT_KEYS if _relevant(t, k) and k != "new_routes")  # noqa: E731
    old, old_any = {}, {}
    for h, v in by_sha.items():
        if h == newest or post_boundary(h, newest):        # strictly older code only
            continue
        for r in v:
            i = r["_lk"]
            if i not in new:
                continue
            if i not in old_any or r["mtime"] > old_any[i]["mtime"]:
                old_any[i] = r
            if _sig(r) == _sig(new[i]) and (i not in old or r["mtime"] > old[i]["mtime"]):   # same relevant flags preferred
                old[i] = r
    for i, r in old_any.items():
        old.setdefault(i, r)
    pairs = sorted(set(new) & set(old))
    if not pairs:
        continue
    diff = sorted({k for i in pairs for k in DRIFT_KEYS if _relevant(t, k)
                   and (new[i].get("cfg") or {}).get(k) != (old[i].get("cfg") or {}).get(k)})
    if t == "new_routes" or "new_routes" in diff:      # route lists grew over time: a mismatch only matters if the cell's route changed
        diff = [k for k in diff if k != "new_routes"] + (["new_routes"] if t in PLAN_NONE else [])
    sn = sum(new[i]["score"] or 0 for i in pairs) / len(pairs) * 100
    so = sum(old[i]["score"] or 0 for i in pairs) / len(pairs) * 100
    DRIFT[t] = dict(pairs=len(pairs), new_sha=newest, old_shas=sorted({str(old[i].get("head"))[:7] for i in pairs}),
                    new_mean=round(sn, 1), old_mean=round(so, 1), delta=round(sn - so, 1),
                    succ_new=sum(bool(new[i]["success"]) for i in pairs), succ_old=sum(bool(old[i]["success"]) for i in pairs),
                    flag_diff=diff, matched=not diff,
                    new_fams=sorted({new[i]["run"].split("__")[0].rsplit("_", 1)[0] for i in pairs}),
                    old_fams=sorted({old[i]["run"].split("__")[0].rsplit("_", 1)[0] for i in pairs}))
def _sign(cells_):
    import math
    neg = sum(1 for d in cells_ if d["delta"] < 0); pos = sum(1 for d in cells_ if d["delta"] > 0); n_ = neg + pos
    p_ = sum(math.comb(n_, k) for k in range(0, min(neg, pos) + 1)) / 2 ** n_ * 2 if n_ else 1.0
    return dict(cells=len(cells_), neg=neg, pos=pos, zero=len(cells_) - n_, p_two_sided=round(min(1.0, p_), 3),
                mean_delta=round(sum(d["delta"] for d in cells_) / max(1, len(cells_)), 1))
# supplementary same-flag cross-sha pairs that are not candidate rows (e.g. match_and_pick veto on: b4cd6c2 vs 7722b0b)
for _t, _old_f, _new_f in (("match_and_pick_from_conveyor", "v15:c4b_matchvetosg_on", "v15:c4b_matchvetosg_vetonly"),):
    _o, _n = fam.get((_t, _old_f), {}), fam.get((_t, _new_f), {})
    _pp = sorted(set(_o) & set(_n))
    if _pp and _t not in DRIFT:
        _sn = 100 * sum(_n[i]["score"] or 0 for i in _pp) / len(_pp); _so = 100 * sum(_o[i]["score"] or 0 for i in _pp) / len(_pp)
        DRIFT[_t] = dict(pairs=len(_pp), new_sha=sorted({str(_n[i].get("head"))[:7] for i in _pp})[-1],
                         old_shas=sorted({str(_o[i].get("head"))[:7] for i in _pp}), new_mean=round(_sn, 1), old_mean=round(_so, 1),
                         delta=round(_sn - _so, 1), succ_new=sum(bool(_n[i]["success"]) for i in _pp),
                         succ_old=sum(bool(_o[i]["success"]) for i in _pp), flag_diff=[], matched=True,
                         new_fams=[_new_f], old_fams=[_old_f], note="same flags (veto on), not the candidate arm")
DRIFT_SIGN = dict(matched=_sign([d for d in DRIFT.values() if d["matched"]]), all=_sign(list(DRIFT.values())))
print("DRIFT SIGN TEST:", DRIFT_SIGN)
# lead (29 Oct): #334 (cd4036da) fixes two failure paths in place (pour_order continues only after a verified pour; p2's put_back
# opens only when the lowering arrived). Watch classify x2 candidate rows on >= cd4036d vs the earlier p2on rows (same rule as
# #330); pour_by_language is watched the same way vs its earlier candidate rows.
W334_SHA = "cd4036d"
W334 = {t: {"pre": {}, "post": {}} for t in ("classify_objects", "classify_objects_by_language", "pour_by_language")}
for (t, f), allr in fam_all.items():
    if t not in W334 or (t, f) not in fam or f.startswith("c1:"):
        continue
    if not (candidate(f, list(fam[(t, f)].values())) or (sg_family(f)[0] and f.rsplit("_", 1)[-1] in ADOPTED_SUF)):
        continue
    for r in allr:
        post = post_boundary(r.get("head"), W334_SHA)
        if not post and t != "pour_by_language" and not f.endswith("_p2on"):
            continue                    # classify: the earlier side is the adopted p2on rows only
        side = W334[t]["post" if post else "pre"]
        o = side.get(r["layout_id"])
        if o is None or r["mtime"] > o["mtime"]:
            side[r["layout_id"]] = r
W334_SUMMARY = {}
for t, d in W334.items():
    if not d["post"]:
        continue
    pc = sorted(set(d["pre"]) & set(d["post"]))
    s_ = lambda v: dict(n=len(v), succ=sum(bool(x["success"]) for x in v), score=round(100 * sum(x["score"] or 0 for x in v) / max(1, len(v)), 1))  # noqa: E731
    pre, post = s_([d["pre"][i] for i in pc]), s_([d["post"][i] for i in pc])
    W334_SUMMARY[t] = dict(pairs=len(pc), pre=pre, post=post, post_runs=sorted(f"{x.get('box')}:{x['run']}" for x in d["post"].values()))
    if len(pc) >= 3 and (post["succ"] < pre["succ"] or post["score"] < pre["score"] - 10):
        GUARD_ALERTS.append(f"#334 (cd4036d) {t}: candidate rows on >= cd4036d {post['succ']}/{len(pc)} {post['score']} vs earlier "
                            f"{'p2on' if t != 'pour_by_language' else 'candidate'} {pre['succ']}/{len(pc)} {pre['score']} on the same layouts")
# lead (29 Oct): classify p2 A/B -> per-layout wins / losses / ties per cell (adopt below the 10-pt bar if 0 losing layouts on
# success count; same principle as imitate D)
def ab_wlt(t, on, off):
    a, b = fam.get((t, on), {}), fam.get((t, off), {})
    pc = sorted(set(a) & set(b))
    per = [dict(layout=i, on=f"{int(bool(a[i]['success']))}/{a[i]['score']}", off=f"{int(bool(b[i]['success']))}/{b[i]['score']}",
                succ=("W" if a[i]["success"] and not b[i]["success"] else "L" if b[i]["success"] and not a[i]["success"] else "T"),
                score=("W" if (a[i]["score"] or 0) > (b[i]["score"] or 0) else "L" if (a[i]["score"] or 0) < (b[i]["score"] or 0) else "T"))
           for i in pc]
    c = lambda k, v: sum(x[k] == v for x in per)  # noqa: E731
    return dict(pairs=len(pc), on_only=sorted(set(a) - set(b)), off_only=sorted(set(b) - set(a)),
                succ_wlt=f"{c('succ', 'W')}/{c('succ', 'L')}/{c('succ', 'T')}", score_wlt=f"{c('score', 'W')}/{c('score', 'L')}/{c('score', 'T')}",
                on=agg([a[i] for i in pc]) if pc else None, off=agg([b[i] for i in pc]) if pc else None, per_layout=per)
P2_AB = {t: ab_wlt(t, f"{pre}_p2on", f"{pre}_p2off") for t, pre in
         (("classify_objects", "c4b_classifysg"), ("classify_objects_by_language", "c4b_classifylangsg"))}
for t, d in P2_AB.items():
    for k in ("on", "off"):
        if d[k]:
            d[k] = {kk: d[k][kk] for kk in ("n", "succ", "score")}
    print(f"P2 AB {t}: {d['pairs']} pairs, success W/L/T {d['succ_wlt']}, score W/L/T {d['score_wlt']}, unpaired on {d['on_only']} off {d['off_only']}")
    for x in d["per_layout"]:          # p2 adopted (lead, 29 Oct): the remaining pairs are confirmation; any success loss alerts
        if x["succ"] == "L":
            GUARD_ALERTS.append(f"ported_classify p2 confirmation: {t} {x['layout']} loses a success (p2on {x['on']} vs p2off {x['off']})")
# H / Integration (29 Oct): #332 acts only under overlap -> diagnostic ovlres vs ovlon (not a lever; the lever is ovlres vs ovloff)
OVL_DIAG = ab_wlt("pour_by_language", "c4b_pourlangsg_ovlres", "c4b_pourlangsg_ovlon")
# lead (29 Oct): class-(c) levers (failure class identified on measurement episodes, cause / fix / thresholds dev-only) are reported
# SPLIT: (i) all pairs, (ii) the unseen ids = the regression check, (iii) the identified ids = an optimistic gain.
# Adoption bar: 0 success losers on (ii) and a gain on (i).
def ab_split(t, on, off, identified, expected=None):     # expected: the A/B's full pair count (bar pending until then)
    d = ab_wlt(t, on, off)
    def grp(ids):
        per = [x for x in d["per_layout"] if ids is None or x["layout"] in ids]
        a, b = fam.get((t, on), {}), fam.get((t, off), {})
        c = lambda k, v: sum(x[k] == v for x in per)  # noqa: E731
        lay = [x["layout"] for x in per]
        return dict(pairs=len(per), succ_wlt=f"{c('succ', 'W')}/{c('succ', 'L')}/{c('succ', 'T')}",
                    score_wlt=f"{c('score', 'W')}/{c('score', 'L')}/{c('score', 'T')}", succ_losers=c("succ", "L"),
                    on=round(100 * sum(a[i]["score"] or 0 for i in lay) / max(1, len(lay)), 1),
                    off=round(100 * sum(b[i]["score"] or 0 for i in lay) / max(1, len(lay)), 1),
                    on_succ=sum(bool(a[i]["success"]) for i in lay), off_succ=sum(bool(b[i]["success"]) for i in lay))
    allg = grp(None)
    unseen = grp({x["layout"] for x in d["per_layout"]} - set(identified))
    ident = grp(set(identified))
    bar = ("pending" if not d["pairs"] or (expected and d["pairs"] < expected) else
           "NO UNSEEN IDS (no regression check possible)" if not unseen["pairs"] else
           "MET" if unseen["succ_losers"] == 0 and allg["on"] > allg["off"] else "NOT MET")
    return dict(all=allg, unseen=unseen, identified=ident, identified_ids=sorted(identified), bar=bar,
                note="" if unseen["pairs"] or not d["pairs"] else "no unseen ids: every paired id was looked at (optimistic only)")
CLASS_C = {"make_kong watch_relocate (#337)": ab_split("make_kong", "c4b_makekongsg_relocon", "c4b_makekongsg_relocoff",
                                                       (100005, 100006, 100022, 100023, 100026, 100028), expected=15),
           # glyph_upright (afix B): W looked at arrange 100005-09 / 100020-24 (std / random) -> no unseen measurement id
           **{f"{t} afix (glyph_upright + tiles)": ab_split(t, f"{pre}_afixon", f"{pre}_afixoff",
                                                            tuple(range(100005, 100010)) + tuple(range(100020, 100025)))
              for t, pre in (("arrange_largest_number", "c4b_largestnumbersg"),
                             ("arrange_largest_number_random", "c4b_largestnumberrandomsg"))}}
for k, d in CLASS_C.items():
    if d["all"]["pairs"]:
        print(f"CLASS C {k}: (i) all {d['all']['pairs']} pairs {d['all']['on_succ']}-{d['all']['off_succ']} succ, {d['all']['on']} vs "
              f"{d['all']['off']}, succ W/L/T {d['all']['succ_wlt']}; (ii) unseen {d['unseen']['pairs']}: succ W/L/T {d['unseen']['succ_wlt']}, "
              f"{d['unseen']['on']} vs {d['unseen']['off']}; (iii) identified {d['identified']['pairs']}: succ W/L/T {d['identified']['succ_wlt']}, "
              f"{d['identified']['on']} vs {d['identified']['off']}; bar {d['bar']} {d['note']}")
# H / lead (29 Oct): pre-registered bowls lever, both arms on 680ab86 (#339): B = candidate + stack_reach grasp + nest_height measured
# + nest_verify support (developed on 100030-49 only). ADOPT iff the pooled bowls score mean (stack_bowls + _random, 10 pairs each)
# is B > A AND the guards (stack_blocks, _random, _by_language; play_stacking_toy optional; 5 pairs each) have 0 success losers.
BOWLS3_LEVER = "stack_reach grasp + nest_height measured + nest_verify support (H, #339 680ab86; pre-registered rule)"
def _arms(t, base=None):              # Integration's names: c4b_<cell>sg_bowls3off / _bowls3on (+ _v15), paired by layout
    out = {}
    for (t2, f2), v in fam.items():
        if t2 == t and not f2.startswith("c1:") and re.search(r"_bowls3(off|on)$", f2):
            out.setdefault("A" if f2.endswith("off") else "B", {}).update(v)
    return out.get("A", {}), out.get("B", {})
_b = {t: _arms(t) for t in ("stack_bowls", "stack_bowls_random")}
_g = {t: _arms(t) for t in ("stack_blocks", "stack_blocks_random", "stack_blocks_by_language")}
# lead (30 Oct ~09:45Z, pre-data, for the D-C19 freeze): n cut to bowls 5 + 5 pairs on 100005-09 and guards 3 pairs each on 100005-07
_bp = [(t, i) for t, (a, b) in _b.items() for i in sorted(set(a) & set(b)) if 100005 <= i <= 100009]
_gp = [(t, i) for t, (a, b) in _g.items() for i in sorted(set(a) & set(b)) if 100005 <= i <= 100007]
_bA = round(100 * sum(_b[t][0][i]["score"] or 0 for t, i in _bp) / max(1, len(_bp)), 1)
_bB = round(100 * sum(_b[t][1][i]["score"] or 0 for t, i in _bp) / max(1, len(_bp)), 1)
_gl = [f"{t} {i}" for t, i in _gp if _g[t][0][i]["success"] and not _g[t][1][i]["success"]]
_need_g = 9                          # 3 guards x 3 pairs (n cut pre-data 2026-10-30 ~09:45Z for the D-C19 freeze)
BOWLS3_RULE = dict(bowl_pairs=len(_bp), bowls_A=_bA, bowls_B=_bB, guard_pairs=len(_gp), guard_success_losers=_gl,
                   per_cell={t: len(set(a) & set(b)) for t, (a, b) in {**_b, **_g}.items()},
                   n_cut="n cut pre-data 2026-10-30 ~08:45Z (machine time) for the D-C19 freeze (bowls 5 + 5 on 100005-09, guards 3 x 3 on 100005-07)",
                   status=("pending" if len(_bp) < 10 or len(_gp) < _need_g else
                           "ADOPT" if _bB > _bA and not _gl else "REJECT"))
if _bp or _gp:
    print("BOWLS3 RULE:", BOWLS3_RULE)
# S / lead (29 Oct): pre-registered play_stacking_toy A/B, both arms on 57dd7f3: ADOPT iff the pooled score mean B > A over the 10
# pairs (100005-09 + 100020-24) and no pair loses more than 0.3 (score, not successes); no guards (the flag reaches stack_on_pegs only)
PBR_LEVER = "peg_base_refine local + peg_base_tries 8 (S, #349 / #351; arms on 27c7a5b; pre-registered rule)"
_pa, _pb = fam.get(("play_stacking_toy", "c4b_playstackingtoysg_pbrA"), {}), fam.get(("play_stacking_toy", "c4b_playstackingtoysg_pbrB"), {})
_pp = sorted(set(_pa) & set(_pb))
_pA = round(100 * sum(_pa[i]["score"] or 0 for i in _pp) / max(1, len(_pp)), 1)
_pB = round(100 * sum(_pb[i]["score"] or 0 for i in _pp) / max(1, len(_pp)), 1)
_pl = [f"{i}: A {_pa[i]['score']} vs B {_pb[i]['score']}" for i in _pp if (_pa[i]["score"] or 0) - (_pb[i]["score"] or 0) > 0.3 + 1e-9]
PBR_RULE = dict(pairs=len(_pp), A=_pA, B=_pB, big_losses=_pl,
                status="pending" if len(_pp) < 10 else ("ADOPT" if _pB > _pA and not _pl else "REJECT"))
if _pp:
    print("PBR RULE:", PBR_RULE)
# S / lead (29 Oct): arm C = B + peg_stuck_confirm on (#355); C over B iff pooled C > B and no pair B - C > 0.3 (score)
PBRC_LEVER = "peg_stuck_confirm on, over pbrB (S, #355, 27c7a5b; pre-registered rule)"
_pc = fam.get(("play_stacking_toy", "c4b_playstackingtoysg_pbrC"), {})
_pcp = sorted(set(_pb) & set(_pc))
_cB = round(100 * sum(_pb[i]["score"] or 0 for i in _pcp) / max(1, len(_pcp)), 1)
_cC = round(100 * sum(_pc[i]["score"] or 0 for i in _pcp) / max(1, len(_pcp)), 1)
_cl = [f"{i}: B {_pb[i]['score']} vs C {_pc[i]['score']}" for i in _pcp if (_pb[i]["score"] or 0) - (_pc[i]["score"] or 0) > 0.3 + 1e-9]
PBRC_RULE = dict(pairs=len(_pcp), B=_cB, C=_cC, big_losses=_cl,
                 status="pending" if len(_pcp) < 10 else ("ADOPT" if _cC > _cB and not _cl else "REJECT"))
if _pcp:
    print("PBR C RULE:", PBRC_RULE)
# W / lead (29 Oct): pre-registered imitate detection A/B (57dd7f3): ADOPT iff imitate's pooled score B > A (10 pairs: 100006/07/09,
# 100020-26), AND every make_kong guard B row (100005-09) shows both gate-off reasons, AND no make_kong success loss
SMASK_LEVER = "imitate bundle: support_mask + wrist_pool support_masked + imitate_wait home + imitate_keep_sides (W; arms on 01c1a2b; pre-registered rule)"
SMASK_WHY = {"support_mask": "no support robot at its home", "wrist_pool": "no support robot masked in this frame"}
# lead (29 Oct): the smask A/B (never ran) is superseded by the combined "bundle" families on ae4eaca (#357); same rule
_ia, _ib = fam.get(("imitate_sorting_sequence", "c4b_imitatesg_bundleoff"), {}), fam.get(("imitate_sorting_sequence", "c4b_imitatesg_bundleon"), {})
_ka, _kb = fam.get(("make_kong", "c4b_makekongsg_bundleoff"), {}), fam.get(("make_kong", "c4b_makekongsg_bundleon"), {})
_ip, _kp = sorted(set(_ia) & set(_ib)), sorted(set(_ka) & set(_kb))
_iA = round(100 * sum(_ia[i]["score"] or 0 for i in _ip) / max(1, len(_ip)), 1)
_iB = round(100 * sum(_ib[i]["score"] or 0 for i in _ip) / max(1, len(_ip)), 1)
def _gate_off(r):                    # both gate-off reasons present, nothing active (missing fields = FAIL)
    c = r.get("cfg") or {}
    return (c.get("smask_initial_both") is True and c.get("support_mask_why") == [SMASK_WHY["support_mask"]]
            and c.get("wrist_pool_why") == [SMASK_WHY["wrist_pool"]])
_kgate = [f"{i}: support_mask {(_kb[i].get('cfg') or {}).get('support_mask_why')}, wrist_pool {(_kb[i].get('cfg') or {}).get('wrist_pool_why')}"
          for i in sorted(_kb) if not _gate_off(_kb[i])]
_kloss = [str(i) for i in _kp if _ka[i]["success"] and not _kb[i]["success"]]
SMASK_RULE = dict(imitate_pairs=len(_ip), imitate_A=_iA, imitate_B=_iB, kong_pairs=len(_kp), kong_gate_fail=_kgate, kong_success_losers=_kloss,
                  status=("REJECT" if _kgate else "pending" if len(_ip) < 10 or len(_kp) < 5 else
                          "ADOPT" if _iB > _iA and not _kgate and not _kloss else "REJECT"))   # a gate failure is final (block stopped)
if _ip or _kp:
    print("SMASK RULE:", SMASK_RULE)
# W / lead (30 Oct): bundle arm C (bundlewid) over B: pooled imitate C > B (10 ids), no pair C - B < -0.3, every make_kong C row's initial
# record has support_mask.applied false + wrist_pool / wrist_identity why "no support robot masked in this frame" + changed [], and no
# make_kong success loss C vs B. Diagnostics (not decision inputs): wrist_identity checked / changed / time_s, repair drops per row
BWID_LEVER = "wrist_identity support_masked + imitate_repair, over the bundle (W, #369; arms on 01c1a2b; pre-registered rule)"
_NOMASK = "no support robot masked in this frame"
_ic = fam.get(("imitate_sorting_sequence", "c4b_imitatesg_bundlewid"), {})
_kc = fam.get(("make_kong", "c4b_makekongsg_bundlewid"), {})
_icp, _kcp = sorted(set(_ib) & set(_ic)), sorted(set(_kb) & set(_kc))
_cBm = round(100 * sum(_ib[i]["score"] or 0 for i in _icp) / max(1, len(_icp)), 1)
_cCm = round(100 * sum(_ic[i]["score"] or 0 for i in _icp) / max(1, len(_icp)), 1)
_cbl = [f"{i}: B {_ib[i]['score']} vs C {_ic[i]['score']}" for i in _icp if (_ic[i]["score"] or 0) - (_ib[i]["score"] or 0) < -0.3 - 1e-9]
def _gate_c(r):
    g = (r.get("cfg") or {}).get("init_gate") or {}
    return (g.get("support_mask_applied") is False and g.get("wrist_pool_why") == _NOMASK and g.get("wrist_identity_why") == _NOMASK
            and g.get("wrist_identity_changed") == [])
_kcg = [f"{i}: {(_kc[i].get('cfg') or {}).get('init_gate')}" for i in sorted(_kc) if not _gate_c(_kc[i])]
_kcl = [str(i) for i in _kcp if _kb[i]["success"] and not _kc[i]["success"]]
BWID_DIAG = {i: dict(score=_ic[i]["score"], **{k: ((_ic[i].get("cfg") or {}).get("init_gate") or {}).get(k)
                                              for k in ("wrist_identity_checked", "wrist_identity_changed", "wrist_identity_time_s", "wrist_identity_budget_hit")},
                     repair_drops=(_ic[i].get("cfg") or {}).get("imitate_repair_drops")) for i in sorted(_ic)}
BWID_RULE = dict(imitate_pairs=len(_icp), B=_cBm, C=_cCm, big_losses=_cbl, kong_pairs=len(_kcp), kong_gate_fail=_kcg,
                 kong_success_losers_vs_B=_kcl, diagnostics=BWID_DIAG,
                 status=("pending" if len(_icp) < 10 or len(_kcp) < 5 else
                         "ADOPT" if _cCm > _cBm and not _cbl and not _kcg and not _kcl else "REJECT"))
if _icp or _kcp:
    print("BWID RULE:", {k: v for k, v in BWID_RULE.items() if k != "diagnostics"})
# lead (30 Oct): the bundle block was stopped; both rules read REJECT as registered (guard premise error, not amended)
SMASK_RULE["status"] = BWID_RULE["status"] = "REJECT"
SMASK_RULE["decision"] = BWID_RULE["decision"] = ("REJECT (lead, 2026-10-30 ~08:40Z machine time): guard premise error (support robot present on make_kong); not "
                                                   "amended because a kong score had been seen; block stopped; re-test after D-C19")
# N / lead (30 Oct): pre-registered dolls A/B on c42e011 (#358): ADOPT iff the pooled score over both dolls cells (10 pairs each)
# is B > A and no pair loses more than 0.3; kept apart from twin_union_pool's dolls arms (different suffixes, never pooled)
RSR_LEVER = "row_size_read look_union (N, #358; arms on 4654086; dolls x2; pre-registered rule)"
_rp = [(t, i, a[i], b[i]) for t, pre in (("sort_nesting_dolls_by_size", "c4b_sortdollssg"), ("sort_nesting_dolls_by_size_random", "c4b_sortdollsrandomsg"))
       for a, b in [(fam.get((t, f"{pre}_rsroff"), {}), fam.get((t, f"{pre}_rsron"), {}))] for i in sorted(set(a) & set(b))]
_rA = round(100 * sum(x[2]["score"] or 0 for x in _rp) / max(1, len(_rp)), 1)
_rB = round(100 * sum(x[3]["score"] or 0 for x in _rp) / max(1, len(_rp)), 1)
_rl = [f"{t} {i}: A {a['score']} vs B {b['score']}" for t, i, a, b in _rp if (a["score"] or 0) - (b["score"] or 0) > 0.3 + 1e-9]
RSR_RULE = dict(pairs=len(_rp), A=_rA, B=_rB, big_losses=_rl,
                status="pending" if len(_rp) < 20 else ("ADOPT" if _rB > _rA and not _rl else "REJECT"))
if _rp:
    print("RSR RULE:", RSR_RULE)
# H / lead (30 Oct): pre-registered fill_pen_holder A/B on 165e007 (#360), v15 layouts: ADOPT iff pooled B > A over 10 pairs and no
# pair loses more than 0.3
FPL_LEVER = "fill_move_lift clear (H, #360; arms on be7a798; fill_pen_holder v15; pre-registered rule)"
_fa = {**fam.get(("fill_pen_holder", "v15:c4b_fillpensg_fpliftA"), {})}
_fb = {**fam.get(("fill_pen_holder", "v15:c4b_fillpensg_fpliftB"), {})}
_fp = sorted(set(_fa) & set(_fb))
_fA = round(100 * sum(_fa[i]["score"] or 0 for i in _fp) / max(1, len(_fp)), 1)
_fB = round(100 * sum(_fb[i]["score"] or 0 for i in _fp) / max(1, len(_fp)), 1)
_fl = [f"{i}: A {_fa[i]['score']} vs B {_fb[i]['score']}" for i in _fp if (_fa[i]["score"] or 0) - (_fb[i]["score"] or 0) > 0.3 + 1e-9]
FPL_RULE = dict(pairs=len(_fp), A=_fA, B=_fB, big_losses=_fl,
                status="pending" if len(_fp) < 10 else ("ADOPT" if _fB > _fA and not _fl else "REJECT"))
if _fp:
    print("FPL RULE:", FPL_RULE)
# H / lead (30 Oct): arm C = B + fill_joint_limits measured. C over B iff pooled C > B and no pair B - C > 0.3 (needs B adopted);
# if B fails, C is judged vs A by the same rule; pending until all 10 pairs land
FPLC_LEVER, FPLCA_LEVER = "fill_joint_limits measured, over fpliftB (H; arms on be7a798; fill_pen_holder v15; pre-registered rule)", "fill_move_lift clear + fill_joint_limits measured, C vs A (H; be7a798; used only if B fails; pre-registered rule)"
_fc = {**fam.get(("fill_pen_holder", "v15:c4b_fillpensg_fpliftC"), {})}
def _pool(x, y):                     # pooled y vs x over common ids, big losses (x - y > 0.3)
    ids = sorted(set(x) & set(y))
    mx = round(100 * sum(x[i]["score"] or 0 for i in ids) / max(1, len(ids)), 1)
    my = round(100 * sum(y[i]["score"] or 0 for i in ids) / max(1, len(ids)), 1)
    bl = [f"{i}: {x[i]['score']} vs {y[i]['score']}" for i in ids if (x[i]["score"] or 0) - (y[i]["score"] or 0) > 0.3 + 1e-9]
    return dict(pairs=len(ids), base=mx, arm=my, big_losses=bl,
                status="pending" if len(ids) < 10 else ("ADOPT" if my > mx and not bl else "REJECT"))
_cb, _ca = _pool(_fb, _fc), _pool(_fa, _fc)
FPLC_RULE = dict(c_over_b=_cb, c_vs_a=_ca,
                 status=("pending" if FPL_RULE["status"] == "pending" else
                         (_cb["status"] if FPL_RULE["status"] == "ADOPT" else _ca["status"])),
                 path=("C over B (B adopted)" if FPL_RULE["status"] == "ADOPT" else "C vs A (B failed)" if FPL_RULE["status"] == "REJECT"
                       else "waiting for B"))
if _cb["pairs"] or _ca["pairs"]:
    print("FPL C RULE:", FPLC_RULE)
# H / lead (30 Oct): pre-registered make_toast A/B (01c1a2b): ADOPT iff pooled make_toast B > A (10 pairs, raw 100005-09 + 100020-24),
# no make_toast pair loses > 0.3, no guard pair loses > 0.3, and each guard's pooled B >= A - 0.05 (guards: build_tower v15, stack_bowls
# v15, play_stacking_toy; 3 pairs each on 100005-07)
RSK_LEVER = "rehover_siblings keep + sibling_order far + press_repeat cross (H; arms on c10d765; make_toast + guards; pre-registered rule)"
def _arm(t, f):
    return {**fam.get((t, f), {}), **fam.get((t, "v15:" + f), {})}
_ta, _tb = _arm("make_toast", "c4b_maketoastsg_rskA"), _arm("make_toast", "c4b_maketoastsg_rskB")
_tp = sorted(set(_ta) & set(_tb))
_tA = round(100 * sum(_ta[i]["score"] or 0 for i in _tp) / max(1, len(_tp)), 1)
_tB = round(100 * sum(_tb[i]["score"] or 0 for i in _tp) / max(1, len(_tp)), 1)
_tl = [f"make_toast {i}: A {_ta[i]['score']} vs B {_tb[i]['score']}" for i in _tp if (_ta[i]["score"] or 0) - (_tb[i]["score"] or 0) > 0.3 + 1e-9]
_rg = {}
for t, f in (("build_tower", "c4b_buildtowersg"), ("stack_bowls", "c4b_stackbowlssg"), ("play_stacking_toy", "c4b_playstackingtoysg")):
    a, b = _arm(t, f + "_rskA"), _arm(t, f + "_rskB")
    ids = sorted(set(a) & set(b))
    ga = round(100 * sum(a[i]["score"] or 0 for i in ids) / max(1, len(ids)), 1)
    gb = round(100 * sum(b[i]["score"] or 0 for i in ids) / max(1, len(ids)), 1)
    _rg[t] = dict(pairs=len(ids), A=ga, B=gb, big_losses=[f"{i}: A {a[i]['score']} vs B {b[i]['score']}" for i in ids
                                                          if (a[i]["score"] or 0) - (b[i]["score"] or 0) > 0.3 + 1e-9],
                  pooled_ok=(gb >= ga - 5.0))
RSK_RULE = dict(make_toast=dict(pairs=len(_tp), A=_tA, B=_tB, big_losses=_tl), guards=_rg,
                status=("pending" if len(_tp) < 10 or any(g["pairs"] < 3 for g in _rg.values()) else
                        "ADOPT" if _tB > _tA and not _tl and all(not g["big_losses"] and g["pooled_ok"] for g in _rg.values()) else "REJECT"))
if _tp or any(g["pairs"] for g in _rg.values()):
    print("RSK RULE:", RSK_RULE)
# lead (30 Oct, registered at ~08:06Z, AFTER 8 of the 10 pairs had landed and been reported): belt_wrist_veto ON is ADOPTED iff at the
# full 10 pairs (v15 100005-09 + 100020-24, both b4cd6c2) pooled on > off, 0 success losers (off succeeds, on fails) and net
# successes >= +1; otherwise not adopted. Expire (exp vs vetonly) is not a lever.
VETO_LEVER = "belt_wrist_veto on (3651b68) [v15 layouts]"
_von, _voff = fam.get(("match_and_pick_from_conveyor", "v15:c4b_matchvetosg_on"), {}), fam.get(("match_and_pick_from_conveyor", "v15:c4b_matchvetosg_off"), {})
_vp = sorted(set(_von) & set(_voff) & (set(range(100005, 100010)) | set(range(100020, 100025))))
_vOn = round(100 * sum(_von[i]["score"] or 0 for i in _vp) / max(1, len(_vp)), 1)
_vOff = round(100 * sum(_voff[i]["score"] or 0 for i in _vp) / max(1, len(_vp)), 1)
_vl = [str(i) for i in _vp if _voff[i]["success"] and not _von[i]["success"]]
_vnet = sum(bool(_von[i]["success"]) for i in _vp) - sum(bool(_voff[i]["success"]) for i in _vp)
VETO_RULE = dict(registered="2026-10-30 ~08:06Z machine time (lead), after 8 of 10 pairs had landed and been reported",
                 pairs=len(_vp), on=_vOn, off=_vOff, success_losers=_vl, net_successes=_vnet,
                 status="pending" if len(_vp) < 10 else ("ADOPT" if _vOn > _vOff and not _vl and _vnet >= 1 else "REJECT"))
if _vp:
    print("VETO RULE:", VETO_RULE)
# lead (30 Oct): template rules registered SCORE-BLIND at 2026-10-30 09:14:32Z (machine clock), before any of these blocks' scores were
# looked at (M had seen only ids / shas): ADOPT iff over the judged pairs pooled on > off, 0 success losers, net successes >= +1, and
# 0 success losers on the guard cells. Pending until the expected pair count lands (Integration's queue).
REG_TS = "2026-10-30 09:14:32Z (machine clock), score-blind"
def _tmpl(blocks, guards=(), expected=None, expected_guard=None):
    def arm(t, f):
        return {**fam.get((t, f), {}), **fam.get((t, "v15:" + f), {})}
    pairs, gl, gp = [], [], 0
    for t, on, off, ids in blocks:
        a, b = arm(t, on), arm(t, off)
        pairs += [(t, i, a[i], b[i]) for i in sorted(set(a) & set(b)) if ids is None or i in ids]
    for t, on, off in guards:
        a, b = arm(t, on), arm(t, off)
        com = sorted(set(a) & set(b)); gp += len(com)
        gl += [f"{t} {i}" for i in com if b[i]["success"] and not a[i]["success"]]
    m_on = round(100 * sum(x[2]["score"] or 0 for x in pairs) / max(1, len(pairs)), 1)
    m_off = round(100 * sum(x[3]["score"] or 0 for x in pairs) / max(1, len(pairs)), 1)
    los = [f"{t} {i}" for t, i, a, b in pairs if b["success"] and not a["success"]]
    net = sum(bool(a["success"]) for _, _, a, _ in pairs) - sum(bool(b["success"]) for _, _, _, b in pairs)
    done = expected is not None and len(pairs) >= expected and (not guards or (expected_guard is not None and gp >= expected_guard))
    return dict(registered=REG_TS, pairs=len(pairs), expected=expected, guard_pairs=gp, expected_guard=expected_guard, on=m_on, off=m_off,
                success_losers=los, net_successes=net, guard_success_losers=gl,
                status=("pending" if not done else "ADOPT" if m_on > m_off and not los and net >= 1 and not gl else "REJECT"))
REG_RULES = {
    "coin_slot": _tmpl([("deposit_coin", "c4b_coinslotsg_on", "c4b_coinslotsg_off", {100008} | set(range(100020, 100030)))], expected=11),
    "ttt_slot": _tmpl([("play_tic_tac_toe", "c4b_tttslotsg_on", "c4b_tttslotsg_off", None)], expected=5),
    "insert_key_key": _tmpl([("insert_key", "c4b_insertkeysg_keyon", "c4b_insertkeysg_keyoff", None)], expected=10),
    "yaw": _tmpl([("align_blocks", "c4b_alignsg_yawon", "c4b_alignsg_yawoff", None)],
                 guards=[("stack_blocks", "c4b_stackblockssg_yawon", "c4b_stackblockssg_yawoff")], expected=9, expected_guard=5),
    # keyslot / blocksslot share the slot lever with coin / ttt: same template, registered 2026-10-30 09:17:09Z (machine clock) score-blind (no rows existed)
    "key_slot": _tmpl([("insert_key", "c4b_keyslotsg_on", "c4b_keyslotsg_off", None)], expected=5),
    "blocks_slot": _tmpl([("stack_blocks", "c4b_blocksslotsg_on", "c4b_blocksslotsg_off", None)], expected=5)}
# expected counts from Integration's queue (09:17Z, score-blind): ttt 5; key 10 (v15 05-09 + 20-24); yaw 9 (align 05-09 + 20-24 minus
# the D-C10c 100020) + guard 5 (stack_blocks v15 05-09); keyslot / blocksslot 5 each; ubg 9 (05-09 + 20, 21, 23, 24); union 5 per cell
# on 05-09 minus push_T(_random) 100005 (D-C10c, ported_drag turn) -> gain 14, guard 14
# lead (30 Oct): registered 2026-10-30 09:16:13Z (machine clock), score-blind (no unionon / ubgon / ubgoff row existed yet; union A arms
# known, as plain candidate rows). Union: gain = the _random cells, guard = 0 success losers on the std cells (W's std-loss concern)
REG_TS2 = "2026-10-30 09:16:13Z (machine clock), score-blind (no B-arm row existed)"
# lead (30 Oct): yaw guard premise error (the registered stack_blocks guard was inert: its sym-4 labels are yaw-fixed). Guards added
# PRE-DATA at 2026-10-30 10:45:52Z (machine time), no guard row existed: 0 success losers AND pooled B >= A - 0.05 on each of
# stack_blocks_by_language (5), swap_blocks v15 (5), sweep_blocks (3). align's completed result stands as the gain criterion.
def _yaw_guards():
    out = {}
    for t, pre, n in (("stack_blocks_by_language", "c4b_stackblocksbylanguagesg", 5), ("swap_blocks", "c4b_swapblockssg", 5),
                      ("sweep_blocks", "c4b_sweepblockssg", 3)):
        a = {**fam.get((t, pre + "_yawon"), {}), **fam.get((t, "v15:" + pre + "_yawon"), {})}
        b = {**fam.get((t, pre + "_yawoff"), {}), **fam.get((t, "v15:" + pre + "_yawoff"), {})}
        ids = sorted(set(a) & set(b))
        ma = round(100 * sum(a[i]["score"] or 0 for i in ids) / max(1, len(ids)), 1)
        mb = round(100 * sum(b[i]["score"] or 0 for i in ids) / max(1, len(ids)), 1)
        los = [str(i) for i in ids if b[i]["success"] and not a[i]["success"]]
        out[t] = dict(pairs=len(ids), expected=n, on=ma, off=mb, success_losers=los, pooled_ok=(ma >= mb - 5.0))
    return out
_yg = _yaw_guards()
_yw = REG_RULES["yaw"]
_yw["guards_added"] = dict(registered="2026-10-30 10:45:52Z (machine time), guard premise error; guards added pre-data", **_yg)
_yw["status"] = ("pending" if _yw["pairs"] < 9 or any(g["pairs"] < g["expected"] for g in _yg.values()) else
                 "ADOPT" if (_yw["on"] > _yw["off"] and not _yw["success_losers"] and _yw["net_successes"] >= 1 and not _yw["guard_success_losers"]
                             and all(not g["success_losers"] and g["pooled_ok"] for g in _yg.values())) else "REJECT")
REG_RULES["put_bottles_ubg"] = dict(_tmpl([("put_bottles_into_dustbin", "c4b_putbottlessg_ubgon", "c4b_putbottlessg_ubgoff", None)], expected=9),
                                    registered=REG_TS2)
REG_RULES["twin_union_pool"] = dict(_tmpl(
    [("pour_liquid_into_cup_random", "c4b_pourliquidrandomsg_unionon", "c4b_pourliquidrandomsg_unionoff", None),
     ("sort_nesting_dolls_by_size_random", "c4b_sortdollsrandomsg_unionon", "c4b_sortdollsrandomsg_unionoff", None),
     ("push_T_random", "c4b_pushtrandomsg_unionon", "c4b_pushtrandomsg_unionoff", None)],
    guards=[("sort_nesting_dolls_by_size", "c4b_sortdollssg_unionon", "c4b_sortdollssg_unionoff"),
            ("push_T", "c4b_pushtsg_unionon", "c4b_pushtsg_unionoff"),
            ("pour_liquid_into_cup", "c4b_pourliquidsg_unionon", "c4b_pourliquidsg_unionoff")], expected=14, expected_guard=14),
    registered=REG_TS2)
# lead (30 Oct, pre-registered 12:30Z before any data): ported_pour_untilt "capped" on pour_balls_into_vase, 5 pairs on settled 100005-09,
# both arms on >= #378's merge (62f43590). ADOPT iff no pair loses (B score < A score); no guards (it only touches pour_p1).
def _untilt():
    a = {**fam.get(("pour_balls_into_vase", "c4b_pourballssg_untiltoff"), {})}
    b = {**fam.get(("pour_balls_into_vase", "c4b_pourballssg_untilton"), {})}
    ids = sorted(set(a) & set(b) & set(range(100005, 100010)))
    los = [f"{i}: A {a[i]['score']} vs B {b[i]['score']}" for i in ids if (b[i]["score"] or 0) < (a[i]["score"] or 0) - 1e-9]
    return dict(registered="2026-10-30 12:30Z (lead), pre-data", pairs=len(ids), expected=5, guard_pairs=0,
                A=round(100 * sum(a[i]["score"] or 0 for i in ids) / max(1, len(ids)), 1),
                B=round(100 * sum(b[i]["score"] or 0 for i in ids) / max(1, len(ids)), 1),
                A_succ=sum(bool(a[i]["success"]) for i in ids), B_succ=sum(bool(b[i]["success"]) for i in ids),
                shas=sorted({str(x.get("head"))[:8] for x in list(a.values()) + list(b.values())}), losing_pairs=los,
                status="pending" if len(ids) < 5 else ("ADOPT" if not los else "REJECT"))
REG_RULES["pour_untilt"] = _untilt()
# coordinator (30 Oct ~17:40Z): make_toast code-vs-noise repeats (Integration): rep1A / rep2A = the rskA line (c10d7658), rep1rb / rep2rb = the
# old rb line (e783de43), on 100005 / 100006. Diagnostic only: their suffixes are not in SG_CAND, so they are never a counted row.
def _mt_rep():
    arms = {a: {k: fam.get(("make_toast", f"c4b_maketoastsg_{k}"), {}) for k in (f"rep1{a}", f"rep2{a}")} for a in ("A", "rb")}
    out = {}
    for a, ks in arms.items():
        for i in (100005, 100006):
            out[f"{a} {i}"] = [f"{(ks[k][i]['score'] or 0):.2f}{'' if ks[k][i]['success'] else ' ' + str(ks[k][i].get('category'))} ({str(ks[k][i].get('head'))[:8]})"
                               if i in ks[k] else "pending" for k in ks]
    n = sum(1 for v in out.values() for x in v if x != "pending")
    a_fail = all(not x.startswith("1.00") for k in ("A 100005", "A 100006") for x in out[k] if x != "pending")
    rb_pass = all(x.startswith("1.00") for k in ("rb 100005", "rb 100006") for x in out[k] if x != "pending")
    read = "code (A fails, rb passes on every repeat)" if a_fail and rb_pass else "noise (mixed outcomes)"
    return dict(rows=n, expected=8, per_id=out, status="pending" if n < 8 else f"done: {read}")
MT_REP = _mt_rep()
for k, v in REG_RULES.items():
    if v["pairs"] or v["guard_pairs"]:
        print(f"REG RULE {k}:", {kk: vv for kk, vv in v.items() if kk != "registered"})
# lead's match design (29 Oct): veto = on vs off (b4cd6c2); expire = exp vs vetonly (7722b0b, the lever above); interim merges =
# on (b4cd6c2) vs vetonly (7722b0b), same flags, different sha: a diagnostic only
EXP_DIAG = ab_wlt("match_and_pick_from_conveyor", "v15:c4b_matchvetosg_on", "v15:c4b_matchvetosg_vetonly")
if EXP_DIAG["pairs"]:
    print(f"MERGES DIAG match_and_pick v15 on (b4cd6c2) vs vetonly (7722b0b): {EXP_DIAG['pairs']} pairs, success W/L/T {EXP_DIAG['succ_wlt']}, score W/L/T {EXP_DIAG['score_wlt']}")
if OVL_DIAG["pairs"]:
    print(f"OVL DIAG pour_by_language ovlres vs ovlon: {OVL_DIAG['pairs']} pairs, {OVL_DIAG['on']['succ']}/{OVL_DIAG['on']['score']} vs "
          f"{OVL_DIAG['off']['succ']}/{OVL_DIAG['off']['score']}, success W/L/T {OVL_DIAG['succ_wlt']}, score W/L/T {OVL_DIAG['score_wlt']}")
for g in GUARD_ALERTS:
    print("GUARD ALERT:", g)
# pending levers: every rendered A/B pair (treatment vs its *off control, same task, paired on layout id); all flags are
# default off on main, so adopting one applies ALL its measured deltas (one fixed system), good and bad
LEVER_NAMES = {"gc": "grip_check (in the candidate)", "rs": "release search (post_place_read 6dof_search)", "seg": "seg_propose",
               "h": "h95 (adopted into the candidate; shown for reference)", "emb": "belt embedding identity", "stereo": "coin stereo", "fm": "fm (organize)"}
LEV = collections.defaultdict(dict)          # lever -> cell -> dict(delta, n, ctrl, treat, mtime)
for (t, g), tr in fam.items():               # treatment arm g = <prefix>_<suffix>, control = <prefix>_<control suffix>
    pre, suf = g.rsplit("_", 1)
    if pre.endswith("sg") and suf in SG_TREAT:
        csuf, lever = SG_TREAT[suf]
        if pre.startswith("c1:") and lever:
            lever = f"{lever} [compute1]"
        lever = lever or SG_ON_LEVER.get(pre.replace("v15:", ""), f"{pre.split('_', 1)[-1]} on")
        if pre.startswith("v15:"):
            lever += " [v15 layouts]"
    elif pre.endswith("sg") and suf in NOARM and t in PLAN_NONE:
        # no A arm (Integration, 29 Oct): without the route the policy prints Plan none, so the control is 0 by construction
        csuf, lever = None, NOARM[suf] + (" [compute1]" if pre.startswith("c1:") else "") + (" [v15 layouts]" if pre.startswith("v15:") else "")
    elif suf not in PAIRS or pre.endswith("sg"):
        continue
    else:
        csuf, lever = PAIRS[suf]
    ctrl = (fam.get((t, f"{pre}_{csuf}")) if csuf else
            {i: dict(success=False, score=0.0, mtime=0, head="plan-none", run="plan none (0 by construction)") for i in tr})
    if not ctrl:
        continue
    common = sorted(set(ctrl) & set(tr))
    if len(common) < 2:
        continue
    c = agg([ctrl[i] for i in common]); x = agg([tr[i] for i in common])
    prev = LEV[lever].get(t)
    if prev is None or x["mtime"] > prev["mtime"]:
        LEV[lever][t] = dict(delta=round(x["score"] - c["score"], 1), n=len(common), ctrl=c, treat=x,
                             mtime=x["mtime"], arms=f"{g} vs " + (f"{pre}_{csuf}" if csuf else "Plan none (0 by construction)"),
                             dsucc=x["succ"] - c["succ"])
if _ca["pairs"] >= 2:                 # the C-vs-A lever (pending only on the B-failed path)
    _ids = sorted(set(_fa) & set(_fc))
    LEV[FPLCA_LEVER]["fill_pen_holder"] = dict(delta=round(_ca["arm"] - _ca["base"], 1), n=_ca["pairs"], ctrl=agg([_fa[i] for i in _ids]),
                                              treat=agg([_fc[i] for i in _ids]), mtime=0, arms="fpliftC vs fpliftA",
                                              dsucc=sum(bool(_fc[i]["success"]) for i in _ids) - sum(bool(_fa[i]["success"]) for i in _ids))
# proxy rows where nothing newer exists: the newest round per task
ORDER = ["P3r3", "TL", "P3n", "P3", "S79", "B_breadth2", "B_heldpose", "B_breadth", "AG", "early (B_stack / B_build / B_tower ...)"]
prox = [] if J5 else sorted(load_rows(ARGS.state_dir / "proxy_perc.json", optional=True), key=lambda r: ORDER.index(r["family"]) if r["family"] in ORDER else 99)
AGP = {}
for r in prox:
    t = r["task"]
    if r["family"] == "AG":           # the agent policy (an LLM drives skills): another policy, not G3; shown, not counted
        AGP.setdefault(t, dict(date=r["date"], n=r["n"], succ=r["success"], score=round(r["score"])))
        continue
    if t not in P or "IN PROGRESS" in P[t]["src"]:      # a thin candidate row never displaces an older full row
        P[t] = dict(date=r["date"], src=f"{r['family']} (proxy)", sha=r["heads"][0][:7], n=r["n"], succ=r["success"],
                    score=round(r["score"]))
NOTES = {"play_tic_tac_toe": "c4b_tttcur_gc (0/5, 26, 08b0dd7) excluded: the #241 bug, fixed; Integration's 028bb76 re-run replaces the row"}

_kl = [d for lv, cs in LEV.items() if "kong_draw_read place v2" in lv for t, d in cs.items() if t == "make_kong"]
KONG_GAIN = any(d["delta"] >= 10 and d["n"] >= 5 for d in _kl)      # lead: drop the "blocked: draw read" tag once B gains
# ---------------------------------------------------------------- the table
rows = []
for cell, dim, w in cells():
    base = cell[:-7] if cell.endswith("_random") else cell
    cands = C.get(cell, [])
    pend = lambda c: str(c["sha"])[:7] in PENDING_CEIL or cell in PENDING_CEIL_BY_TASK.get(str(c["sha"])[:7], ())  # noqa: E731
    upside = [c for c in cands if pend(c)]
    cands = [c for c in cands if not pend(c)]
    pick = max(cands, key=lambda c: (c["n"] >= 5, c["date"], c["n"])) if cands else None   # a full row beats a thin newer one
    if pick and upside:
        u = max(upside, key=lambda c: (c["n"] >= 5, c["score"], c["date"]))
        why = PENDING_CEIL.get(str(u["sha"])[:7]) or f"{str(u['sha'])[:7]} (unmerged tiles commit)"
        pick = dict(pick, note=(pick["note"] + "; " if pick["note"] else "") + f"pending {why}: {u['succ']}/{u['n']} / {u['score']}")
    if pick and cell in BLOCKED and not KONG_GAIN:
        pick = dict(pick, note=(pick["note"] + "; " if pick["note"] else "") + BLOCKED[cell])
    if pick and cell in DEV_UPSIDE and "Plan none" in str(pick["note"]):
        pick = dict(pick, note=(pick["note"] + "; " if pick["note"] else "") + DEV_UPSIDE[cell])
    if cell == "imitate_sorting_sequence":       # C: 100000-09 has dev demos only on 100002 / 100004; use the demo-layout row
        demo = [c for c in cands if "demo" in str(c["lay"])]
        pick = max(demo, key=lambda c: (c["date"], c["n"])) if demo else pick
    p = P.get(cell)
    gs = pick["score"] if pick else None
    ps = p["score"] if p else None
    lower = None
    if pick is None and p is not None:
        lower = ps                       # a perception row bounds the skill ceiling from below
    rows.append(dict(cell=cell, dim=dim, w=w, gt=pick, perc=p, gt_score=gs, perc_score=ps, lower=lower,
                     other=OTHER.get(cell, []), all=cands))


def pts(w, x):
    return 100 * w * x / 100 if x is not None else None


for r in rows:
    w = r["w"]
    g = r["gt_score"] if r["gt_score"] is not None else r["lower"]
    pc = r["perc_score"] if r["perc_score"] is not None else 0.0
    r["gap_skill"] = None if g is None else 100 - g
    r["gap_perc"] = None if g is None else max(0.0, g - pc)
    r["pts_skill"] = None if g is None else w * (100 - g)
    r["pts_perc"] = None if g is None else w * max(0.0, g - pc)
    r["pts_total"] = w * (100 - pc)
# J5: a cell is a possible distribution shift when its optimistic (main) row sits on old-gen layouts, or its J5 generator differs
J5_SHIFT = {}
if J5:
    try:
        _main = {x["cell"]: x for x in json.load(open(f"{D}/cells.json"))}
    except FileNotFoundError:
        _main = {}
    for r in rows:
        m = _main.get(r["cell"], {}).get("perc") or {}
        why = ("measurement set v7 -> J5 v15" if r["cell"] == "make_kong" else
               "parked v13 rows" if r["cell"] == "pick_from_conveyor_by_image" else
               "optimistic row on old-gen layouts" if "old-gen" in str(m.get("src", "")) else None)
        if why:
            J5_SHIFT[r["cell"]] = why
            if r["perc"]:
                r["perc"] = dict(r["perc"], src=r["perc"]["src"] + f" [shift possible: {why}]")
# lead: the clean headline = the Average over all 54 cells, unmeasured J5 cells at 0 (same as main); plus coverage N/54 and a
# like-for-like line (the main optimistic score on the same N cells)
def _j5_lfl():                         # evaluated at output time (after the N/C pass)
    if not J5:
        return None
    _cov = [r for r in rows if r["perc"] and not r.get("nc")]
    _ms = lambda c: (0.0 if (_main.get(c, {}).get("nc") or not _main.get(c, {}).get("perc")) else (_main[c].get("perc_score") or 0.0))  # noqa: E731
    return dict(coverage=f"{len(_cov)}/{len(rows)}", cells=sorted(r["cell"] for r in _cov),
                j5_pts=round(sum(r["w"] * (r["perc_score"] or 0) for r in _cov), 2),
                main_pts_same_cells=round(sum(r["w"] * _ms(r["cell"]) for r in _cov), 2),
                max_pts_same_cells=round(sum(r["w"] * 100 for r in _cov), 2))
# if pending PRs adopt: the levers whose weighted net over every measured cell is positive (all default off on main)
W = {r["cell"]: r["w"] for r in rows}
NET = {lv: sum(W.get(t, 0) * d["delta"] for t, d in cs.items()) for lv, cs in LEV.items()}
# adopt = net positive AND at least one cell moved >= 10 score points on n >= 5 paired layouts (else it is noise)
IN_CANDIDATE = {"gc", "h"} | ({"align_measure parallel (91eb98f)"} if ALIGN_ADOPT else set())   # adopted: shown, never pending
PARKED = {"select_ties off", "select_ties (D-C17)", "h95 off (re-check on the adopted candidate)",
          "watch_relocate (W, #337 5194d42; make_kong; class c: report split)",   # bar not met (lead, 30 Oct): parked
          "slot_read + held_thin + thin_slot_keep + slot_settle_merged on (dfb8169, #308)",   # coin slot REJECT (lead, 30 Oct); ttt / keyslot / blocksslot parked unrun
          "ported_pour_pace overlap on #332 (4f90f97, pour last untilt; B only, vs ovloff on 53f1043, a documented cross-sha exception)",   # rejected / parked (lead, 30 Oct)
          "track_expire on (#335 7722b0b; vs vetonly = candidate + belt_wrist_veto, same sha) [v15 layouts]",   # not a lever (lead, 30 Oct)
          "imitate bundle: support_mask + wrist_pool support_masked + imitate_wait home + imitate_keep_sides (W; arms on 01c1a2b; pre-registered rule)",
          "wrist_identity support_masked + imitate_repair, over the bundle (W, #369; arms on 01c1a2b; pre-registered rule)",   # REJECTED (lead, 30 Oct)
          "imitate o2o (#305)", "imitate o2o + twin (#305)"}     # o2o / o2otwin parked: superseded by o2owait / o2owaitlook     # parked by Integration (lead)
IN_CANDIDATE |= PARKED
IN_CANDIDATE |= {"imitate o2o + wait origin (#322/#323, c86dc91; vs gcon on d2cd632, cross-sha exception)",
                 "imitate o2o + wait origin + look ambiguous (#322/#323, c86dc91; vs gcon, cross-sha)", "ported_pour_aim b10 (#293, b16707e)", "kong_draw_read v3 (#315 a21eaee1, in place; vs adopted v2 on the same 15 layouts)", "pad_row_extend (#298, ed44940)", "kong_draw_read place v2 (#311 merged, + second wrist view; vs kongoff on 08ea836, a documented cross-sha exception)",
                 "series_labels numbered (#299, ed44940)", "ported_classify p1 (on the candidate, vs numon; ddf5faa)",
                 "ported_classify p2 (7a48ac9)", "belt_wrist_veto on (3651b68) [v15 layouts]",   # adopted (lead, 30 Oct)
                 "peg_base_refine local + peg_base_tries 8 (S, #349 / #351; arms on 27c7a5b; pre-registered rule)",   # adopted (lead, 30 Oct)
                 "kong_draw_read place v1 (S branch 08ea836; first-version reference, not adoptable)"}   # shown, never pending      # adopted into the candidate (lead, 29 Oct)
# class-(c) levers are pending only when the lead's split bar is MET (0 success losers on the unseen ids, a gain on all pairs)
CLASS_C_LEVERS = {"watch_relocate (W, #337 5194d42; make_kong; class c: report split)": ["make_kong watch_relocate (#337)"],
                  "arrange fix: tile_grasp prefer + tile_recover + tile_look_cap + series_link_raise + glyph_upright (W, #331 2203d2c; replaces the held tile A/B)":
                      ["arrange_largest_number afix (glyph_upright + tiles)", "arrange_largest_number_random afix (glyph_upright + tiles)"]}
_cc_ok = lambda lv: (not lv.startswith(BOWLS3_LEVER) or BOWLS3_RULE["status"] == "ADOPT") and (not lv.startswith(PBR_LEVER) or PBR_RULE["status"] == "ADOPT") and (not lv.startswith(RSR_LEVER) or RSR_RULE["status"] == "ADOPT") and (not lv.startswith(FPL_LEVER) or FPL_RULE["status"] == "ADOPT") and (not lv.startswith(FPLC_LEVER) or (FPL_RULE["status"] == "ADOPT" and _cb["status"] == "ADOPT")) and (not lv.startswith(FPLCA_LEVER) or (FPL_RULE["status"] == "REJECT" and _ca["status"] == "ADOPT")) and (not lv.startswith(PBRC_LEVER) or (PBRC_RULE["status"] == "ADOPT" and PBR_RULE["status"] == "ADOPT")) and (not lv.startswith(SMASK_LEVER) or SMASK_RULE["status"] == "ADOPT") and (lv not in CLASS_C_LEVERS or all(CLASS_C[k]["bar"] == "MET" for k in CLASS_C_LEVERS[lv] if CLASS_C[k]["all"]["pairs"]))  # noqa: E731
# pre-registered levers: their own rule decides (ADOPT), the generic 10-pt rule does not apply (C over B also needs B over A)
RULE_LEVERS = {BOWLS3_LEVER: BOWLS3_RULE, PBR_LEVER: PBR_RULE, PBRC_LEVER: PBRC_RULE, SMASK_LEVER: SMASK_RULE, RSR_LEVER: RSR_RULE,
               FPL_LEVER: FPL_RULE, RSK_LEVER: RSK_RULE, VETO_LEVER: VETO_RULE,
               "slot_read + held_thin + thin_slot_keep + slot_settle_merged on (dfb8169, #308)": dict(status=(
                   "ADOPT" if all(REG_RULES[k]["status"] == "ADOPT" for k in ("coin_slot", "ttt_slot", "key_slot", "blocks_slot")) else
                   "pending" if any(REG_RULES[k]["status"] == "pending" for k in ("coin_slot", "ttt_slot", "key_slot", "blocks_slot")) else "REJECT")),
               "key_handover v2 + key_lean (#320, 7ab74f6)": REG_RULES["insert_key_key"],
               "edge_yaw (perception; #318, 8c78457)": REG_RULES["yaw"],
               "ported_pour_untilt capped (lead pre-registered 12:30Z": REG_RULES["pour_untilt"],
               "put_into unblock + gentle (#321, e321d3d)": REG_RULES["put_bottles_ubg"],
               "twin_union_pool (perception; #325, ed1a773)": REG_RULES["twin_union_pool"], BWID_LEVER: dict(status=BWID_RULE["status"] if SMASK_RULE["status"] == "ADOPT" else "REJECT"), FPLC_LEVER: dict(status=_cb["status"] if FPL_RULE["status"] == "ADOPT" else "REJECT"),
               FPLCA_LEVER: dict(status=_ca["status"] if FPL_RULE["status"] == "REJECT" else "REJECT")}
_ruled = lambda lv: next((r for k, r in RULE_LEVERS.items() if lv.startswith(k)), None)  # noqa: E731
ADOPT = sorted(lv for lv, v in NET.items() if lv not in IN_CANDIDATE and "4090" not in lv and _cc_ok(lv)
               and ((_ruled(lv) is not None and _ruled(lv)["status"] == "ADOPT") or
                    (_ruled(lv) is None and v > 0 and any(d["delta"] >= 10 and d["n"] >= 5 for d in LEV[lv].values()))))
for r in rows:
    pc = r["perc_score"] if r["perc_score"] is not None else 0.0
    add = sum(LEV[lv][r["cell"]]["delta"] for lv in ADOPT if r["cell"] in LEV[lv])
    r["adopt_score"] = max(0.0, min(100.0, pc + add))
    r["adopt_levers"] = [lv for lv in ADOPT if r["cell"] in LEV[lv]]
LEGAL_FIX = {"#227 make_kong identities from image evidence": ("59b2fb9", {"make_kong"}),
             "#233 imitate quota / same_as_label off": ("04952bc", {"imitate_sorting_sequence"}),
             "#232 observed card digit; unread cards stop": ("08b0dd7", {"press_by_number"}),
             "#230 bundle-verify twin record without an evaluated template": ("249aa1d", {
                 "put_bottles_into_dustbin", "make_toast", "make_toast_random", "play_Xylophone", "deposit_coin",
                 "press_by_number", "make_kong"})}
CALIB_FIX = "687e3a4"            # #225: no layout id / task / crc32 fold reaches the policy (tag only: effect within noise)
SHA_ALIAS = {"main~10-27": None, "#169": None}


def row_sha(p):
    sh = str(p.get("sha") or "").split("/")[0].split(" ")[0].split("+")[-1][:7]
    return sh if sh and sh != "None" and sh not in SHA_ALIAS else None


for r in rows:
    nc = NOT_COMPARABLE.get(r["cell"])
    r["nc"] = nc[1] if nc and r["perc"] and r["perc"]["src"].startswith(nc[0]) else None
    r["legal_tags"] = []
    if r["perc"]:
        sh = row_sha(r["perc"])
        for why, (fix, tasks) in LEGAL_FIX.items():
            if r["cell"] in tasks and (sh is None or not post_boundary(sh, fix)):
                r["nc"] = (r["nc"] + "; " if r["nc"] else "") + f"not comparable: predates legality fix {why} ({fix})"
        if sh is None or not post_boundary(sh, CALIB_FIX):
            r["legal_tags"].append("pre-legality (calibration fold, #225)")
rows.sort(key=lambda r: (-round(r["pts_total"], 2), r["gt"] is None, -(r["pts_perc"] or 0)))


def fmt(x, d=1):
    return "-" if x is None else f"{x:.{d}f}"


out = ["| # | cell | dim | weight (% of Avg) | skill ceiling: success / score (source, sha, layouts, n) | measured now: success / score "
       "(source, sha, n) | if pending adopt | gap_skill | gap_perc | Avg pts: fix skills / fix perception / total |",
       "|---|---|---|---|---|---|---|---|---|---|"]
for i, r in enumerate(rows, 1):
    g, p = r["gt"], r["perc"]
    gtxt = (f"{g['succ']}/{g['n']} / {g['score']} ({g['src']}, {g['sha']}, {g['lay']}{', ' + g['note'] if g['note'] else ''})"
            if g else (f"none; >= {r['lower']} from perception" if r["lower"] is not None else "**none**"))
    if not g and r["other"]:
        o = max(r["other"], key=lambda o: (not o["seen"], o["date"]))
        gtxt += f"; other system (robot-harness A_gt): {o['succ']}/{o['n']} / {o['score']:.0f} ({o['lay']}{', SEEN' if o['seen'] else ''})"
    ptxt = f"{p['succ']}/{p['n']} / {p['score']} ({p['src']}, {p['sha']}, {p['date']})" if p else "**none** (counts 0)"
    if r["nc"]:
        ptxt = "**N/C** " + ptxt + f"; {r['nc']}"
    if r.get("legal_tags"):
        ptxt += " [" + "; ".join(r["legal_tags"]) + "]"
    if not p and r["cell"] in AGP:
        a = AGP[r["cell"]]
        ptxt += f"; agent policy (proxy, {a['date']}, not G3): {a['succ']}/{a['n']} / {a['score']}"
    atxt = (f"{r['adopt_score']:.0f} (+{'+'.join(r['adopt_levers'])})" if r["adopt_levers"] else "=")
    out.append(f"| {i} | {r['cell']} | {r['dim'][:4]} | {100 * r['w']:.2f} | {gtxt} | {ptxt} | {atxt} | {fmt(r['gap_skill'], 0)} | "
               f"{fmt(r['gap_perc'], 0)} | {fmt(r['pts_skill'], 2)} / {fmt(r['pts_perc'], 2)} / {r['pts_total']:.2f} |")
open(f"{D}/cells{SFX}.md", "w").write("\n".join(out))
json.dump(rows, open(f"{D}/cells{SFX}.json", "w"), indent=1, default=str)
tot = collections.Counter()
for r in rows:
    if r["pts_skill"] is not None:
        tot["skill"] += r["pts_skill"]
        tot["perc"] += r["pts_perc"]
    else:
        tot["unknown"] += r["pts_total"]
    tot["total"] += r["pts_total"]
print({k: round(v, 2) for k, v in tot.items()})
def avg(key):     # the board Average with not-comparable cells kept out (their dimension's mean over the other cells)
    tot = 0.0
    for dim in {r["dim"] for r in rows}:
        rs = [r for r in rows if r["dim"] == dim]
        ok = [r for r in rs if not r["nc"]]
        tot += sum(r["w"] * (r[key] or 0) for r in ok) * sum(r["w"] for r in rs) / sum(r["w"] for r in ok)
    return tot


import os as _os
_vf = f"{D}/v15_state.json"
V15_STATE = json.load(open(_vf)) if _os.path.exists(_vf) else {}
V15_NEW = []
for r in rows:
    c, p = r["cell"], r["perc"]
    if c in V15_CELLS and p and "v15 layouts" in p["src"] and not V15_STATE.get(c, {}).get("switched"):
        prev = V15_STATE.get(c, {}).get("old_score")
        d = r["w"] * (p["score"] - (prev or 0))
        V15_STATE[c] = dict(switched=True, old_score=prev, new_score=p["score"], delta_pts=round(d, 2),
                            at=time.strftime("%m-%d %H:%MZ", time.gmtime()), big=abs(d) > 1.0)
        V15_NEW.append(f"{c}: {prev} -> {p['score']} ({d:+.2f} Avg pts){' FIDELITY SIGNAL (> 1 pt)' if abs(d) > 1.0 else ''}")
    elif c in V15_CELLS and not V15_STATE.get(c, {}).get("switched"):
        V15_STATE.setdefault(c, {})["old_score"] = (p["score"] if p and not r["nc"] else 0.0)
if not J5:
    json.dump(V15_STATE, open(_vf, "w"), indent=1)
if J5:
    V15_NEW = []                      # J5 never logs v15 switches (main mode owns v15_state.json)
for x in V15_NEW:
    print("V15 SWITCH:", x)
cur0 = sum(r["w"] * (r["perc_score"] or 0) for r in rows)
thin_delta = sum(r["w"] * (THIN[r["cell"]][1]["score"] - (0.0 if r["nc"] else (r["perc_score"] or 0)))
                 for r in rows if r["cell"] in THIN and not (r["perc"] and r["perc"]["src"].startswith(THIN[r["cell"]][0])))
# HEADLINE (conservative, lead 28 Oct): an unmeasured or not-comparable cell counts 0; levers only as a delta
cur = sum(r["w"] * (0.0 if r["nc"] else (r["perc_score"] or 0)) for r in rows)
lever_delta = sum(r["w"] * (r["adopt_score"] - (r["perc_score"] or 0)) for r in rows if not r["nc"])
# Misha's stricter headlines (lead 29 Oct; quoted side by side with the as-published one until D-C19)
def _strict_ok(r, level):
    p = r["perc"]
    if not p or r["nc"]:
        return False
    ok = p.get("post267") and p.get("cand") and not p.get("devset")
    if level == 2:
        ok = ok and p["n"] >= 5 and not p.get("mixed") and not p.get("oldgen")
    return bool(ok)
H_POST267 = sum(r["w"] * (r["perc_score"] or 0) for r in rows if _strict_ok(r, 1))
H_STRICT = sum(r["w"] * (r["perc_score"] or 0) for r in rows if _strict_ok(r, 2))
_cnt = lambda pred: (sum(1 for r in rows if r["perc"] and not r["nc"] and pred(r["perc"])),  # noqa: E731
                     round(sum(r["w"] * (r["perc_score"] or 0) for r in rows if r["perc"] and not r["nc"] and pred(r["perc"])), 2))
STRICT_BREAKDOWN = {"pre-#267": _cnt(lambda p: not p.get("post267")), "dev set (100000-04) or breadth3": _cnt(lambda p: p.get("devset", True)),
                    "not candidate": _cnt(lambda p: not p.get("cand")), "n < 5": _cnt(lambda p: p["n"] < 5),
                    "MIXED catalogs": _cnt(lambda p: p.get("mixed")), "old-gen": _cnt(lambda p: p.get("oldgen"))}
STRICT_INVALID = sorted(r["cell"] for r in rows if not _strict_ok(r, 2))
# lead (29 Oct): STRICT is the north star. Per cell: strict score (0 unless strict-valid), gap = w x (ceiling - strict), ceiling = the
# G3 skill ceiling (100 where no ceiling row exists); lever projections are computed against strict
for r in rows:
    r["strict_ok"] = _strict_ok(r, 2)
    r["strict_score"] = (r["perc_score"] or 0.0) if r["strict_ok"] else 0.0
    _ceil = r["gt_score"] if r["gt_score"] is not None else 100.0
    r["strict_gap_pts"] = round(r["w"] * max(0.0, _ceil - r["strict_score"]), 2)
    r["strict_gap_ceiling"] = "G3" if r["gt_score"] is not None else "100 (no G3 row)"
LEVER_DELTA_STRICT = sum(r["w"] * (min(100.0, r["strict_score"] + sum(LEV[lv][r["cell"]]["delta"] for lv in ADOPT if r["cell"] in LEV[lv]))
                                   - r["strict_score"]) for r in rows if r["strict_ok"])
STRICT_GAPS = sorted(((r["cell"], r["strict_gap_pts"], r["strict_score"], r["gt_score"], r["strict_gap_ceiling"]) for r in rows),
                     key=lambda x: -x[1])
json.dump(rows, open(f"{D}/cells{SFX}.json", "w"), indent=1, default=str)     # again, now with the strict fields
adopt = cur + lever_delta
cur_dimmean = avg("perc_score")
print("N/C cells:", {r["cell"]: r["nc"][:40] for r in rows if r["nc"]}, "| measured now with N/C cells at their rows:", round(cur0, 2))
print(f"NORTH STAR (strict): {H_STRICT:.2f} | post-#267 candidate-only {H_POST267:.2f} | as published {cur:.2f} | pending levers "
      f"(vs strict) {LEVER_DELTA_STRICT:+.2f}")
print("AS PUBLISHED measured now:", round(cur, 2), "| pending levers delta:", round(lever_delta, 2), ADOPT,
      "| side: N/C cells at their dimension mean", round(cur_dimmean, 2), "| N/C at their rows", round(cur0, 2))
for lv in sorted(LEV, key=lambda k: -NET[k]):
    print(f"  lever {lv:7s} net {NET[lv]:+.2f} pts:", ", ".join(f"{t} {d['delta']:+.0f} (n{d['n']}, {d['arms']})" for t, d in LEV[lv].items()))
CAND_SHAS = sorted({r["perc"]["sha"] for r in rows if r["perc"] and r["perc"].get("candidate")})
NONCAND = sorted(r["cell"] for r in rows if r["perc"] and not r["perc"].get("candidate"))
print("candidate rows:", sum(1 for r in rows if r["perc"] and r["perc"].get("candidate")), "cells, shas", CAND_SHAS,
      "| other measured cells (not the candidate):", len(NONCAND))
json.dump({"aim_confirm": AIM_CONFIRM, "kong_v3": KONG3, "pad_confirm": PAD_CONFIRM, "align_314": ALIGN314, "divergence": {t: dict(v) for t, v in DIVERGE.items()}, "v15_state": V15_STATE, "v15_new_switches": V15_NEW, "older_generator_cells": OLDGEN, "align_rule": ALIGN_RULE, "pour_guard_skills": POUR_GUARD, "remeasure": sorted(r["cell"] for r in rows if r["nc"]),
           "pre_calibration_fold": sorted(r["cell"] for r in rows if r.get("legal_tags")),
           "perception_boundary": "69890e2 (#267 uniform color_map)",
           "cells_post_boundary": sorted(t for t, p in P.items() if p.get("candidate") and post_boundary(p.get("sha"))),
           "dc19_tranche": DC19, "dc19_source": DC19_SOURCE, "dc19_shift_possible": J5_SHIFT, "dc19_like_for_like": _j5_lfl(),
           "dc19_cells_measured": sum(1 for r in rows if r["perc"] and not r["nc"]) if J5 else None,
           "div330": DIV330_SUMMARY, "w334": W334_SUMMARY, "drift_audit": DRIFT, "drift_sign": DRIFT_SIGN, "p2_ab": P2_AB, "ovl_diag_ovlres_vs_ovlon": OVL_DIAG, "merges_diag_on_vs_vetonly": EXP_DIAG, "class_c_split": CLASS_C, "bowls3_rule": BOWLS3_RULE, "pbr_rule": PBR_RULE, "pbr_c_rule": PBRC_RULE, "rsr_rule": RSR_RULE, "fpl_rule": FPL_RULE, "rsk_rule": RSK_RULE, "veto_rule": VETO_RULE, "registered_rules": REG_RULES, "fpl_c_rule": FPLC_RULE, "smask_rule": SMASK_RULE, "bundle_c_rule": BWID_RULE, "box_4090_precision": {t: f"{f} {a['succ']}/{a['n']} {a['score']}" for t, (f, a) in BOX4090.items()}, "in_progress": {t: f"{f} {a['succ']}/{a['n']} {a['score']}" for t, (f, a) in THIN.items()},
           "in_progress_delta": thin_delta, "catalog_mix": sorted(t for t, p in P.items() if len(p.get("catalogs") or []) > 1),
           "catalogs_in_candidate_rows": sorted({c for p in P.values() for c in (p.get("catalogs") or [])}),
           "unclassified_families": sorted(UNCLASSIFIED), "candidate_cfg": CANDIDATE_CFG, "guard_alerts": GUARD_ALERTS, "guard_ok": len(GUARD_OK),
           "measured_now": cur, "headline_post267_candidate": H_POST267, "headline_strict": H_STRICT, "seed_footnote": SEED_FOOTNOTE, "methodology_changes": METHODOLOGY, "code_order_nonlinear": sorted(set(NONLINEAR)), "color_calib_note": "DOJO_COLOR_CALIB ruled out (W, 30 Oct): the initial estimate is bit-identical unset / old catalogue / frozen file (colour map sha 6f826688, record sha 12f09dc6); the tag is kept for the record", "color_calib_env": {r["cell"]: (r["perc"] or {}).get("color_calib_env") for r in rows if r["perc"] and (r["perc"] or {}).get("color_calib_env") not in ("none", None)},
           "lever_delta_strict": LEVER_DELTA_STRICT, "strict_cells": {r["cell"]: r["strict_score"] for r in rows},
           "strict_gaps": STRICT_GAPS,
           "strict_breakdown": STRICT_BREAKDOWN, "strict_invalid_cells": STRICT_INVALID, "post267_swaps": POST267_SWAP,
           "headline_caveat": HEADLINE_CAVEAT, "make_toast_repeats": MT_REP,
           "tuned_on_excluded": TUNED_GROUPED, "lever_delta": lever_delta, "measured_now_nc_at_row": cur0, "nc_at_dim_mean": cur_dimmean, "not_comparable": {r["cell"]: r["nc"] for r in rows if r["nc"]},
           "artefacts": {k: sorted(set(v)) for k, v in ARTEFACTS.items()}, "if_adopt": adopt, "adopt": ADOPT, "net": NET, "candidate": CANDIDATE,
           "candidate_shas": CAND_SHAS, "noncandidate_cells": NONCAND,
           "generated": time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime()),
           "levers": {lv: {t: {k: d[k] for k in ("delta", "n", "arms", "dsucc")} for t, d in cs.items()} for lv, cs in LEV.items()}},
          open(f"{D}/northstar{SFX}.json", "w"), indent=1)
ceil = sum(r["w"] * (r["gt_score"] if r["gt_score"] is not None else (r["lower"] or 0)) for r in rows)
print("projected Average now (latest perception, 0 where missing):", round(cur, 1), "| with perfect perception (skill ceilings, 0 where unknown):", round(ceil, 1))
print("cells:", len(rows), "with a skill-ceiling row:", sum(1 for r in rows if r["gt"]), "with a perception row:", sum(1 for r in rows if r["perc"]))
print("no skill ceiling:", [r["cell"] for r in rows if not r["gt"]])
for t in NOTES:
    print("NOTE", t, NOTES[t])

# ---------------------------------------------------------------- cells with no skill ceiling: the cheapest way to get one
# routing read off dojo/skills/planner.plan (origin/main) and supervisor.VOCABULARIES, not run: verify on episode 1
VARIANT = {"stack_bowls_random", "fold_clothes_random", "sweep_blocks_random", "stack_blocks_random",
           "store_laptop_and_headphones_random"}
GENERIC = {"pour_balls_into_vase": "fill_container ('all X into' a vase)?", "classify_objects": "put_into / arrange?",
           "classify_objects_by_language": "put_into / arrange?", "pack_objects_into_box": "put_into?",
           "pack_objects_into_box_random": "put_into?", "sort_nesting_dolls_by_size": "stack (nest)?",
           "sort_nesting_dolls_by_size_random": "stack (nest)?", "general_pickup": "no lift-only branch; arrange?",
           "align_blocks": "arrange ('push the X into')?", "arrange_largest_number": "no card-row branch?",
           "arrange_largest_number_random": "no card-row branch?"}
NONE = {"push_T": "push", "push_T_random": "push", "swap_T": "T swap (the mat-swap branch needs mats)",
        "pour_liquid_into_cup": "pour", "pour_liquid_into_cup_random": "pour", "pour_by_language": "pour",
        "cover_blocks": "cover / uncover", "solve_equation": "equation tiles"}
H = {"variant": 0.25, "generic": 0.25, "none": 0.10}
nc = ["| cell | weight | Avg pts at stake | G3 route (read off planner, unverified) | other system (robot-harness A_gt) | "
      "est. slot-h (5 ep) |", "|---|---|---|---|---|---|"]
hsum = collections.Counter()
for r in rows:
    if r["gt"]:
        continue
    c = r["cell"]
    k = "variant" if c in VARIANT else "generic" if c in GENERIC else "none"
    route = {"variant": "same skill as the std cell", "generic": "generic branch: " + GENERIC.get(c, ""),
             "none": "no branch (" + NONE.get(c, "?") + "): Plan none, certifies 0"}[k]
    o = max(r["other"], key=lambda o: (not o["seen"], o["date"])) if r["other"] else None
    otxt = f"{o['succ']}/{o['n']} / {o['score']:.0f} ({o['lay']}{', SEEN' if o['seen'] else ''})" if o else "-"
    hsum[k] += H[k]
    nc.append(f"| {c} | {100 * r['w']:.2f} | {r['pts_total']:.2f} | {route} | {otxt} | {H[k]:.2f} |")
open(f"{D}/cells_noceil{SFX}.md", "w").write("\n".join(nc))
print("no-ceiling slot-hours:", dict(hsum), round(sum(hsum.values()), 2))
