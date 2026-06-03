import argparse, os
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt

TIMESTEP,EPOCHS,EPSILON,GAMMA=2048,10,0.2,0.99
LR_ACTOR,LR_CRITIC=3e-4,1e-3
EPISODES,EP_MAX_STEPS=2000,1000
EP_REWARD_PENALTY,EP_TIMEOUT_PENALTY=-0.01,-25.0
MODEL_PATH="ppo_lunar_lander.pt"

class Actor(nn.Module):
    def __init__(self,s,a):
        super().__init__(); self.net=nn.Sequential(nn.Linear(s,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,a))
    def forward(self,x): return self.net(x)

class Critic(nn.Module):
    def __init__(self,s):
        super().__init__(); self.net=nn.Sequential(nn.Linear(s,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1))
    def forward(self,x): return self.net(x)

class AgentPPO(nn.Module):
    def __init__(self,s,a):
        super().__init__(); self.actor=Actor(s,a); self.critic=Critic(s)
    def act(self,s):
        d=Categorical(logits=self.actor(s)); a=d.sample(); return a,d.log_prob(a),self.critic(s)
    def evaluate(self,s,a):
        d=Categorical(logits=self.actor(s)); return d.log_prob(a),self.critic(s),d.entropy()
    def save_model(self,p): torch.save({"Actor_state_dict":self.actor.state_dict(),"Critic_state_dict":self.critic.state_dict()},p)
    def load_model(self,p,device):
        c=torch.load(p,map_location=device); self.actor.load_state_dict(c["Actor_state_dict"]); self.critic.load_state_dict(c["Critic_state_dict"])

class PPOAgent:
    def __init__(self,state_size,action_size,device):
        self.update_timestep,self.epochs,self.epsilon,self.gamma=TIMESTEP,EPOCHS,EPSILON,GAMMA
        self.device,self.state_dim,self.action_dim=device,state_size,action_size
        self.actions,self.states,self.logprobs,self.rewards,self.state_values,self.dones=[],[],[],[],[],[]
        self.policy=AgentPPO(state_size,action_size).to(device)
        self.policy_old=AgentPPO(state_size,action_size).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer=torch.optim.Adam([{"params":self.policy.actor.parameters(),"lr":LR_ACTOR},{"params":self.policy.critic.parameters(),"lr":LR_CRITIC}])
    def clear(self):
        self.actions.clear(); self.states.clear(); self.logprobs.clear(); self.rewards.clear(); self.state_values.clear(); self.dones.clear()
    def act(self,state):
        with torch.no_grad():
            s=torch.as_tensor(state,device=self.device,dtype=torch.float32); a,lp,v=self.policy_old.act(s)
        self.states.append(s); self.actions.append(a); self.logprobs.append(lp); self.state_values.append(v); return a.item()
    def update(self,returns,writer=None,step=None):
        s=torch.squeeze(torch.stack(self.states)).detach().to(self.device)
        a=torch.squeeze(torch.stack(self.actions)).detach().to(self.device)
        old_lp=torch.squeeze(torch.stack(self.logprobs)).detach().to(self.device)
        old_v=torch.squeeze(torch.stack(self.state_values)).detach().to(self.device)
        adv=returns.detach()-old_v.detach(); adv=(adv-adv.mean())/(adv.std()+1e-7)
        for _ in range(self.epochs):
            lp,v,ent=self.policy.evaluate(s,a); r=torch.exp(lp-old_lp.detach()); v=torch.squeeze(v)
            actor_loss=-torch.min(r*adv,torch.clamp(r,1-self.epsilon,1+self.epsilon)*adv)
            critic_loss=F.smooth_l1_loss(v,returns); loss=actor_loss+0.5*critic_loss-0.01*ent
            self.optimizer.zero_grad(); loss.mean().backward(); self.optimizer.step()
            if writer and step:
                writer.add_scalar("Loss/Total",loss.mean().item(),step); writer.add_scalar("Loss/Actor",actor_loss.mean().item(),step); writer.add_scalar("Loss/Critic",critic_loss.item(),step); writer.add_scalar("Policy/Entropy",ent.mean().item(),step)
        self.policy_old.load_state_dict(self.policy.state_dict())
    def learn(self,writer=None,step=None):
        returns=[]; g=0
        for r,d in zip(reversed(self.rewards),reversed(self.dones)):
            if d: g=0
            g=r+self.gamma*g; returns.insert(0,g)
        returns=torch.tensor(returns,dtype=torch.float32,device=self.device); returns=(returns-returns.mean())/(returns.std()+1e-7)
        self.update(returns,writer,step); self.clear()
    def train_agent(self,env,episodes=EPISODES,max_steps=EP_MAX_STEPS,writer=None):
        time_step,scores=0,[]
        for ep in range(1,episodes+1):
            state,_=env.reset(); total=0; episode_length=max_steps
            for t in range(max_steps):
                action=self.act(state); state,reward,terminated,truncated,_=env.step(action)
                reward+=EP_REWARD_PENALTY; done=terminated or truncated
                if t==max_steps-1 and not done: reward+=EP_TIMEOUT_PENALTY; done=True
                self.rewards.append(reward); self.dones.append(done); time_step+=1; total+=reward
                if time_step%self.update_timestep==0: self.learn(writer,time_step)
                if done: episode_length=t+1; break
            scores.append(total); avg=np.mean(scores[-100:])
            if writer:
                writer.add_scalar("Train/Episode_Return",total,ep); writer.add_scalar("Train/Average_Return_100",avg,ep); writer.add_scalar("Train/Episode_Length",episode_length,ep); writer.add_scalar("Train/Total_Timesteps",time_step,ep)
            print(f"\rEpisode {ep}\tAverage Score: {avg:.2f}",end="")
            if ep%100==0: print(f"\rEpisode {ep}\tAverage Score: {avg:.2f}")
            if avg>=200: print(f"\nEnvironment solved in {ep:d} episodes!\tAverage Score: {avg:.2f}"); break
        return scores
    def save(self,p): self.policy.save_model(p)
    def load(self,p): self.policy.load_model(p,self.device); self.policy_old.load_state_dict(self.policy.state_dict())

def make_agent(env,device): return PPOAgent(env.observation_space.shape[0],env.action_space.n,device)

def train():
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env=gym.make("LunarLander-v2"); writer=SummaryWriter("runs/ppo_lunar_lander")
    agent=make_agent(env,device); scores=agent.train_agent(env,writer=writer); agent.save(MODEL_PATH); writer.close(); env.close()
    plt.plot(scores); plt.xlabel("Episode"); plt.ylabel("Reward"); plt.title("PPO LunarLander Training Scores"); plt.show()

def render(episodes=5,model_path=MODEL_PATH):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env=gym.make("LunarLander-v2",render_mode="human"); agent=make_agent(env,device); agent.load(model_path); agent.policy.eval()
    for ep in range(episodes):
        state,_=env.reset(); done=False; total=0
        while not done:
            s=torch.as_tensor(state,dtype=torch.float32,device=device)
            with torch.no_grad(): action=torch.argmax(agent.policy.actor(s)).item()
            state,reward,terminated,truncated,_=env.step(action); done=terminated or truncated; total+=reward
        print(f"Episode {ep+1}: reward = {total:.2f}")
    env.close()

def play():
    from gymnasium.utils.play import play
    env=gym.make("LunarLander-v2",render_mode="rgb_array")
    play(env,keys_to_action={"a":1,"w":2,"d":3,"s":0,"":0},noop=0,fps=30); env.close()

def export_pngs(log_dir="runs/ppo_lunar_lander",out_dir="tensorboard_pngs"):
    from pathlib import Path
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    Path(out_dir).mkdir(exist_ok=True); ea=EventAccumulator(log_dir); ea.Reload()
    for tag in ea.Tags().get("scalars",[]):
        e=ea.Scalars(tag); x=[v.step for v in e]; y=[v.value for v in e]
        plt.figure(); plt.plot(x,y); plt.xlabel("Step"); plt.ylabel(tag); plt.title(tag); plt.tight_layout()
        plt.savefig(os.path.join(out_dir,tag.replace("/","_")+".png"),dpi=200); plt.close()

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("mode",choices=["train","render","play","export"],nargs="?",default="train"); p.add_argument("--episodes",type=int,default=5); p.add_argument("--model",default=MODEL_PATH); a=p.parse_args()
    {"train":train,"render":lambda:render(a.episodes,a.model),"play":play,"export":export_pngs}[a.mode]()