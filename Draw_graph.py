import os
import matplotlib.pyplot as plt
import torch
from collections import namedtuple, deque

cur_path = os.path.dirname(__file__)
aim_agent_version = "agent_xyIncrease_cnn2d_v1.02.pth"

final_episode = 250
reward_hisroty = []
loss_hisroty = []

# Transition = namedtuple('Transition',
#                         ('state', 'action', 'next_state', 'reward', 'done'))
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

agent_path = cur_path + "/_agent data/" + aim_agent_version

def load_model(path):
    checkpoint = torch.load(path)
    reward_hisroty = checkpoint['total_reward']
    loss_hisroty = checkpoint['loss_hisroty']

    return reward_hisroty, loss_hisroty

reward_hisroty, loss_hisroty = load_model(agent_path)

fig = plt.figure(100)
plt.clf()

ax_one = plt.gca()
ax_loss = ax_one.twinx()

plt.title('Training...')
ax_one.set_xlabel("Episode")
ax_one.set_ylabel("Reward_avg/life")
ax_loss.set_ylabel("Loss_avg/life")

p_reward, =  ax_one.plot([], color='deepskyblue', label="Reward_avg/life"   )
p_mean,   =  ax_one.plot([], color='navy'       , label="Reward_filter_100" )
p_loss,   = ax_loss.plot([], color='tomato'     , label="Loss_avg/life"     )

objective_val = torch.tensor(reward_hisroty[:final_episode], dtype=torch.float)
    
ax_one.plot(objective_val.numpy(), color='deepskyblue')

# tkw = dict(size=3, width=1)
# ax_one.tick_params(axis='y', colors=p_reward.get_color(), **tkw)
# ax_loss.tick_params(axis='y', colors=p_loss.get_color(), **tkw)
# ax_one.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
# ax_one.yaxis.label.set_color(p_reward.get_color())
# ax_loss.yaxis.label.set_color(p_loss.get_color())

# Take 100 episode averages and plot them too
if len(objective_val) >= 100:
    means = objective_val.unfold(0, 100, 1).mean(1).view(-1)
    loss_val = torch.tensor(loss_hisroty[:final_episode], dtype=torch.float)
    means = torch.cat((torch.zeros(99), means))
    ax_one.plot(means.numpy(), color='navy')

if len(loss_hisroty) > 0:
    loss_val = torch.tensor(loss_hisroty[:final_episode], dtype=torch.float)
    ax_loss.plot(loss_val.numpy(), color='tomato')

ax_one.grid(True)
ax_loss.legend(handles=[p_reward, p_mean, p_loss], loc='upper left')

plt.pause(0.001)  # pause a bit so that plots are updated
plt.show()