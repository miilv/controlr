# HARNESS_ENV for robot-harness's launchers on isr-lab (bin/robot-start-j2 and harness/robodojo_{start,launch}_j2.sh
# source it): our symlink farms over Mikhail's BENCH / RD_HOME (scripts/dojo/isr_farm.sh), temp files in our tree.
# Set by scripts/dojo/isr_env.sh.
export BENCH=/home/anywherevla/data2/ilia/bench
export RD_HOME=/home/anywherevla/data2/ilia/rdhome
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR=/home/anywherevla/data2/ilia/tmp
export OMNI_KIT_ACCEPT_EULA=YES MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
