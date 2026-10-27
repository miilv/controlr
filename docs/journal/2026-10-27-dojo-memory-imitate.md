# 2026-10-27: imitate_sorting_sequence, the departure read (Memory stream)

## Why the breadth-2 episode ran into the 25-min cap
- breadth-2's imitate episode: `watch_read` mean 23.7 s × 52 reads = 1230 of the 1488 s worker time.
- Every container read was a full-scene inverse-layout estimate (12 labels, the demonstrator's arm in view).
- It also never saw an object inside the basket. An object the demonstrator moved is outside every generator region,
  so B1 never proposes it there.
- On percrnd's 6.0-proxy watch sequences (100002): 0 of 5 reads with objects inside right, 31-254 s per read on a
  loaded compute3.
- An ROI-only variant (the estimate on the basket's region) also read nothing inside.
- breadth-2 ran imitate on 100000, which has no demonstration (the no-demo stub never moves). Only 13 dev layouts have
  one: 100002, 04, 10, 14-23.

## The departure read (`watch.read_sources`, region `sources`)
- Every mover's episode-start pose (asset, label, pose, confidence) goes with the container region.
- A read is the share of each source's own head-view silhouette (front-most among the sources, our robot excluded)
  that is still foreground.
- A spot goes "gone" below min(0.35, half its baseline). The baseline is the first readable read or the source's
  start-fit confidence (the demonstrator may take one before our first read).
- A gone spot stays gone. The demonstrator's arm in front of an empty spot reads as present: no false departures.
- 0.05-0.07 s per read.

`copy_demonstration` with departure reads:
- the order is the demonstrator's objects (the farther half of the movers from our arms) in the order they left;
- each twin is matched by asset, else by category. The config puts 5 categories × 1 per side, and aim_i is the same
  asset as t_i.
- Done when all left, or 260 steps after the last departure, then `wait_actors_still`.
- G3 (GT reads) keeps the arrival path.

## Measurements (offline, GT for scoring only)
- percrnd's 6.0-proxy watch sequences (5 demo layouts, a frame every 15 steps), GT start poses: demonstrated order
  exact 4/5. The miss was an empty spot in the arm's shadow read as unknown; since fixed, shadow pixels count as
  background.
- Our 5.1-rendered hold sequences (`compute5:~/dojo-data/memory/watch51`, 13 demo layouts), sources from B1's frame-0
  estimate: 0/6 exact.
  - B1 misses or misplaces some of the 10 small movers, so their departures go unseen.
  - It confuses identities (two action-camera models; a phone read as an action_camera).
- Genuine-5.1 B episode, 100002, scan / look / search off (the step-0 scan fails the episode: #102):
  - 736 steps, 400 s wall;
  - 141 departure reads, 8.9 s in total;
  - 3 of 5 departures seen;
  - the first twin's pick had no IK at the believed pose. 0/1.

Next: frame-0 foreground blobs as the sources (obs0 into the first `_watch_read`), and twins paired by appearance.
