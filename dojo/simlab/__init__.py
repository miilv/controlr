"""simlab: headless RoboDojo runner, dev-layout generator, GT dumps and the Twin API.

Dev only. Ground truth read here (object poses, rewards, contacts) never reaches an eval-time
policy; everything in this package runs on layouts we generate ourselves (seed >= 100000).
"""
