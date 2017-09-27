import time,threading
import gym
import gym_pull

def create_env(env_id, client_id, remotes, **kwargs):
    if 'doom' in env_id.lower() or 'labyrinth' in env_id.lower():
        return create_doom(env_id, client_id, **kwargs)

    #spec = gym.spec(env_id)


def create_doom(env_id, client_id, env_wrap = True, record = False, outdir = None, no_life_reward = False, ac_repeat = 0, **_):
    from ppaquette_gym_doom import wrappers

    env_id_lower = env_id.lower()
    if 'labyrinth' in env_id_lower:
        if 'single' in env_id_lower:
            env_id = 'ppaquette/LabyrinthSingle-v0'
        elif 'fix' in env_id_lower:
            env_id = 'ppaquette/LabyrinthManyFixed-v0'
        else:
            env_id = 'ppaquette/LabyrinthMany-v0'
    elif 'very' in env_id.lower():
        env_id = 'ppaquette/DoomMyWayHomeFixed15-v0'
    elif 'sparse' in env_id.lower():
        env_id = 'ppaquette/DoomMyWayHomeFixed-v0'
    elif 'fix' in env_id.lower():
        if '1' in env_id or '2' in env_id:
            env_id = 'ppaquette/DoomMyWayHomeFixed' + str(env_id[-2:]) + '-v0'
        elif 'new' in env_id.lower():
            env_id = 'ppaquette/DoomMyWayHomeFixedNew-v0'
        else:
            env_id = 'ppaquette/DoomMyWayHomeFixed-v0'
    else:
        env_id = 'ppaquette/DoomMyWayHome-v0'

    # VizDoom workaround: Simultaneously launching multiple vizdoom processes
    # makes program stuck, so use the global lock in multi-threading/processing

    client_id = int(client_id)
    time.sleep(client_id * 10)

    env = gym.make(env_id)

    modewrapper = wrappers.SetPlayingMode('algo')
    obwrapper = wrappers.SetResolution('160x120')
    acwrapper = wrappers.ToDiscrete('minimal')

    env = acwrapper(obwrapper(modewrapper(env)))

    env = Vectorize(env)
    env = DiagnosticsInfo(env)
    env = Unvectorize(env)
    return env
