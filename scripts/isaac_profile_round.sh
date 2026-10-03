#!/usr/bin/env bash
# Speed round on compute3 (no LLM): one Isaac server per physics setting, scripted scenarios
# under per-execute variants (scripts/isaac_profile.py). Results: runs/profile/<tag>.jsonl.
#   bash scripts/isaac_profile_round.sh TAG STAGE...      (stages below)
set -uo pipefail
cd "$HOME/controlr"
set -a; [ -f .env ] && . ./.env; set +a
tag=$1; shift
out="runs/profile/$tag.jsonl"
mkdir -p runs/profile
P=".venv/bin/python scripts/isaac_profile.py --port ${CONTROLR_ISAAC_TEST_PORT:-7821} --out $out"
for stage in "$@"; do
  echo "=== $stage $(date -u +%H:%M:%S)"
  case $stage in
    s1) $P --label dt1_it64 --plan "baseline:expert,direct:expert+box_push+mat_push,sub5p:expert+box_push+mat_push" ;;
    s2) $P --label nousd --physics '{"usd_writeback": false}' \
            --plan "direct:expert,sub5p:expert+box_push+mat_push,sub10p:expert+box_push+mat_push" ;;
    s3) $P --label nousd_it32 --physics '{"usd_writeback": false, "solver_position_iterations": 32}' \
            --plan "sub5p:expert+box_push+mat_push" ;;
    s4) $P --label nousd_it16 --physics '{"usd_writeback": false, "solver_position_iterations": 16}' \
            --plan "sub5p:expert+box_push+mat_push" ;;
    s5) $P --label nousd_dt2_it32 --physics '{"usd_writeback": false, "dt": 0.002, "solver_position_iterations": 32}' \
            --plan "sub2:expert+box_push+mat_push,sub5p:expert+box_push+mat_push" ;;
    s6) $P --label it32_v4 --physics '{"usd_writeback": false, "solver_position_iterations": 32, "solver_velocity_iterations": 4}' \
            --plan "sub5p:expert+box_push+mat_push" ;;
    s7) $P --label it32_hull --physics '{"usd_writeback": false, "solver_position_iterations": 32, "forearm_collision_approximation": "convexHull"}' \
            --plan "sub5p:expert+box_push+mat_push" ;;
    s8) $P --label it48 --physics '{"usd_writeback": false, "solver_position_iterations": 48}' \
            --plan "sub5p:expert+box_push+mat_push" ;;
    s9) $P --label it32_t4 --physics '{"usd_writeback": false, "solver_position_iterations": 32, "num_threads": 4}' \
            --plan "sub5p:expert+box_push+mat_push" ;;
    s10) $P --label it64_hull --physics '{"usd_writeback": false, "forearm_collision_approximation": "convexHull"}' \
            --plan "sub5p:expert+box_push+mat_push" ;;
    s11) $P --label it32_nolegacy --physics '{"usd_writeback": false, "solver_position_iterations": 32, "legacy_contact_views": false}' \
            --plan "sub5p:expert+box_push+mat_push,sub1:expert+mat_push+box_push_static" ;;
    s12) $P --label it64_nolegacy --physics '{"usd_writeback": false, "legacy_contact_views": false}' \
            --plan "sub5p:expert+box_push+mat_push,sub1:mat_push+box_push_static" ;;
    s13) $P --label it24 --physics '{"usd_writeback": false, "solver_position_iterations": 24, "legacy_contact_views": false}' \
            --plan "sub5p:expert+mat_push" --expert-scenes 4 ;;
    s14) $P --label it32_noself --physics '{"usd_writeback": false, "solver_position_iterations": 32, "legacy_contact_views": false, "self_collisions": false}' \
            --plan "sub5p:expert+mat_push" --expert-scenes 2 ;;
    f_on) $P --label frames_usd --physics '{"usd_writeback": true}' --plan "sub5p:frames" ;;
    f_off) $P --label frames_nousd --physics '{"usd_writeback": false}' --plan "sub5p:frames" ;;
    *) echo "unknown stage $stage" ;;
  esac
done
echo "=== done $(date -u +%H:%M:%S)"
