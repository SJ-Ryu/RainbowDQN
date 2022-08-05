from gym.envs.registration import register
register(
    id='VTT-v0',
    entry_point='VTT_RL.envs:VTTEnv'
)