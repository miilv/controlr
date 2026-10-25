"""Eval-client side of "dojo_replay": the stock XPolicyLab loop (observation over ws, action chunk back,
RoboDojo's own take_action / step limit / success latch). When the trajectory is over before RoboDojo
ended the episode, the episode counts as failed (what the controlr shim does when it stops early)."""


def eval_one_episode(TASK_ENV, model_client):
    model_client.call(func_name="reset")
    while not TASK_ENV.is_episode_end():
        obs = TASK_ENV.get_obs()
        model_client.call(func_name="update_obs", obs=obs)
        actions = model_client.call(func_name="get_action")
        if not actions:
            break
        for action in actions:
            TASK_ENV.take_action(action)
            if TASK_ENV.is_episode_end():
                break
    if not TASK_ENV.is_episode_end():
        for i in range(int(getattr(TASK_ENV, "num_envs", 1) or 1)):
            TASK_ENV.success[i] = False
        TASK_ENV.is_episode_end()
    print(f"[dojo_replay] episode over: steps {list(getattr(TASK_ENV, 'take_action_cnt', []))}, "
          f"success {list(getattr(TASK_ENV, 'success', []))}", flush=True)


def eval_one_episode_batch(TASK_ENV, model_client):
    if int(getattr(TASK_ENV, "num_envs", 1) or 1) > 1:
        raise RuntimeError("dojo_replay runs one env (eval_batch: false)")
    eval_one_episode(TASK_ENV, model_client)
