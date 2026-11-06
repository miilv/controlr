# Compute3 camera lane preparation

INTEG3 was explicitly released to use compute3 after the shared-session driver
incident. No driver, package-manager, service or VPN changes are authorized or
made. The colleague's desktop session remains present. Initial GPU check:
RTX 5090, driver 595.99.02, 3,233 MiB used, 6.13 W; disk 110 GB free.

Main `86ad1987` is staged as a new `~/dojo-integ-c3-86ad1987` deployment.
The candidate package reference is compute4's 416a2a03 snapshot retained on
compute6; compute4 itself is never contacted. Compute6 is read-only. Offline
Kit extscache, backgrounds and side-model assets are copied into the private
`~/dojo-data/integ3-1106` root. No secret is copied into the deployment.

The peg/waist registration and salts were committed as `ce58682a` before any
fresh layout generation. PEN's conditional #397 arm and the lead's common
Claude-route override were committed as `a1e1e24e` before collection. No camera
or implementation-agent model calls have been made at this checkpoint.

The c3 lane profile supports a private environment file, only heavy1 on the
existing `~/dojo-gpu.lock`, no coord access, admission at <=14 GiB used and a
`who`/GPU receipt before each camera job. Other profiles retain their defaults.
Preparation uses the same exclusive heavy lock, <=14 GiB admission and checked
raw/settled records. Five-minute power logging is required before GPU work.

The three blocks are conditional PEN (10 pairs), peg re-test (20 gain + 10 + 10
guard pairs), waist re-test (20 gain + 10 guard pairs). Historical timings
predict over 12 single-lane camera hours for the re-tests alone; the eight-hour
work period cannot imply all blocks finish. Dispatch remains stopped until
package/data parity, route audit and input manifests are recorded.
