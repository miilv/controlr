# Gap 6 derivation (see gap-6.md §2). MuJoCo 3.14. Inputs: i2rt@120c3c8 i2rt/robot_models/arm/yam/v1/yam.xml (meshes stripped);
# linear_4310 gripper lumped as 0.553 kg body + 0.142 kg tips; payload point mass at grasp_site (0.14465 m). Kp from i2rt/robots/config/yam_v1.yml.
# Fetch: gh api repos/i2rt-robotics/i2rt/contents/i2rt/robot_models/arm/yam/v1/yam.xml -H "Accept: application/vnd.github.raw" > yam.xml
import re, numpy as np, mujoco
s=open('yam.xml').read()
s=re.sub(r'<asset>.*?</asset>','',s,flags=re.S)
s=re.sub(r'<geom[^>]*mesh="[^"]*"[^>]*/>','',s)
# replace gripper placeholder inertial with linear_4310 gripper + tips (lumped) and add sites
grip_inertial='<inertial pos="0 0 -0.0293" mass="0.553" diaginertia="0.00041 0.00037 0.00032"/>'
tips='<body name="tips" pos="0 0 -0.097"><inertial pos="0 0 0" mass="0.142" diaginertia="1e-4 1e-4 1e-4"/></body>'
payload='<body name="payload" pos="0 0 -0.14465"><inertial pos="0 0 0" mass="PAYLOAD" diaginertia="1e-5 1e-5 1e-5"/></body><site name="grasp" pos="0 0 -0.14465"/>'
s=s.replace('<inertial pos="0 0 0" mass="1e-6" diaginertia="1e-9 1e-9 1e-9"/>', grip_inertial+tips+payload)
def model(pl):
    m=mujoco.MjModel.from_xml_string(s.replace('PAYLOAD',str(max(pl,1e-6))))
    return m, mujoco.MjData(m)
m0,d0=model(0.0); m5,d5=model(0.5); m1,d1=model(1.0)
print('total mass arm+gripper', sum(m0.body_mass))
rng=np.random.default_rng(0)
lo=m0.jnt_range[:,0]; hi=m0.jnt_range[:,1]
kp=np.array([80,80,80,10,10,10.])
rows=[]
sid=mujoco.mj_name2id(m0,mujoco.mjtObj.mjOBJ_SITE,'grasp')
for i in range(400000):
    q=lo+(hi-lo)*rng.random(6)
    d0.qpos[:]=q; mujoco.mj_kinematics(m0,d0)
    p=d0.site_xpos[sid].copy(); R=d0.site_xmat[sid].reshape(3,3)
    # tool axis: site z axis; we want it pointing roughly down (within 20 deg) 
    tool=R[:,2]
    if tool[2] > -0.94: continue
    r=np.hypot(p[0],p[1])
    if not (0.0<=p[2]<=0.20 and 0.15<=r<=0.62): continue
    out=[]
    for m,d in ((m0,d0),(m5,d5),(m1,d1)):
        d.qpos[:]=q; d.qvel[:]=0; mujoco.mj_forward(m,d); out.append(d.qfrc_bias.copy())
    tau0,tau5,tau1=out
    J=np.zeros((3,6)); mujoco.mj_jacSite(m0,d0,J,None,sid)
    dq5=(tau5-tau0)/kp; dq1=(tau1-tau0)/kp
    dq10=0.1*tau0/kp  # 10% gravity-model error
    rows.append((r,p[2],tau0[1],tau0[2],tau0[3],tau0[4],np.linalg.norm(J@dq5),np.linalg.norm(J@dq1),np.linalg.norm(J@dq10),tau5[1],tau5[2]))
    if len(rows)>=4000: break
A=np.array(rows); print('n',len(A))
np.set_printoptions(precision=2,suppress=True)
for a,b in [(0.15,0.30),(0.30,0.40),(0.40,0.50),(0.50,0.62)]:
    sel=(A[:,0]>=a)&(A[:,0]<b)
    if sel.sum()==0: continue
    B=A[sel]
    print(f"r {a:.2f}-{b:.2f} n={sel.sum():4d} |tau_g| J2 med {np.median(abs(B[:,2])):.1f} max {abs(B[:,2]).max():.1f}  J3 med {np.median(abs(B[:,3])):.1f} max {abs(B[:,3]).max():.1f}  J4 med {np.median(abs(B[:,4])):.2f} J5 med {np.median(abs(B[:,5])):.2f} | sag mm: 0.5kg unmodeled med {1000*np.median(B[:,6]):.1f} max {1000*B[:,6].max():.1f}; 1kg med {1000*np.median(B[:,7]):.1f}; 10% gcomp err med {1000*np.median(B[:,8]):.1f} max {1000*B[:,8].max():.1f} | J2 w/0.5kg med {np.median(abs(B[:,9])):.1f}")

# --- Cartesian stiffness (translational compliance C = J diag(1/Kp) J^T) ---
rng=np.random.default_rng(1)
lo=m0.jnt_range[:,0]; hi=m0.jnt_range[:,1]
sid=mujoco.mj_name2id(m0,mujoco.mjtObj.mjOBJ_SITE,'grasp')
res=[]
for kp in (np.array([80,80,80,10,10,10.]), np.array([80,80,80,40,10,10.]), np.array([200,200,200,40,40,40.])):
  vals=[];Ms=[]
  rng=np.random.default_rng(1); n=0
  while n<1500:
    q=lo+(hi-lo)*rng.random(6)
    d0.qpos[:]=q; mujoco.mj_forward(m0,d0)
    p=d0.site_xpos[sid]; R=d0.site_xmat[sid].reshape(3,3)
    if R[2,2] > -0.94: continue
    r=np.hypot(p[0],p[1])
    if not (0.0<=p[2]<=0.20 and 0.25<=r<=0.50): continue
    n+=1
    J=np.zeros((3,6)); mujoco.mj_jacSite(m0,d0,J,None,sid)
    C=J@np.diag(1/kp)@J.T
    w=np.linalg.eigvalsh(C)   # m/N
    vals.append((1/w.max()/1000, 1/w.min()/1000))  # N/mm softest, stiffest
    pass
  V=np.array(vals); Mv=np.array(Ms)
  print('kp',kp,'Cartesian stiffness N/mm: softest-dir median %.2f [p10 %.2f p90 %.2f], stiffest-dir median %.1f'%(np.median(V[:,0]),np.percentile(V[:,0],10),np.percentile(V[:,0],90),np.median(V[:,1])))

