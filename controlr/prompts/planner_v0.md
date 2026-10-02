# PLANNING CALL

This is not a control turn: do NOT write MOVE, GRIP, HOLD or STATUS lines. The operating
manual above describes the robot, its base frame, units and the action grammar the
controller will use. The user message contains the TASK and the first observation (camera
image(s) and the STATE line) taken right after the robot was reset.

Write a concise plan (at most {max_words} words) that the controller will read before its
first action. Plain text, exactly these three headings:

SCENE
- Each task-relevant object: what it is, its estimated position in the base frame (x y z
  of its centre or top, in {pos_unit}), its approximate size, and how sure you are.
  Use the camera axis directions, the overlays (if any) and the current TCP position
  from the STATE line as references. Note anything that could be in the way.

STEPS
- Numbered steps, each with an approximate TCP waypoint (x y z in {pos_unit}), the gripper
  state, and what to check in the image before moving on (e.g. "fingers straddle the
  packet", "object rose with the gripper").

RISKS
- The 2-4 most likely failure modes (misjudged depth, collision with a container wall,
  slipping grasp, workspace or joint limits) and how to detect and recover from each.

Be concrete and numeric; do not repeat the manual.
