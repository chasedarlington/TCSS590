import argparse, os
from concurrent.futures import ThreadPoolExecutor
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt

TIMESTEP,EPOCHS,EPISODES,EP_MAX_STEPS=500,1000,2000,500 ## TIMESTEP: env.step(a), EPOCHS: num of training iterations, EPISODES: num of complete env runs
LR_ACTOR,LR_CRITIC=0.0003,0.001
PPO_CLIP,OBS_CLIP,GRAD_CLIP,VF_COEF,ENT_COEF,DISC_COEF=00.20,10.00,00.50,00.50,00.01,00.99
EP_REWARD_PENALTY,EP_TIMEOUT_PENALTY=-0.00001,-10.0
PARALLEL_ENVS,ROLLOUT_WORKERS=1,3
MODEL_PATH="ppo_lunar_lander.pt"
SEED=43

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class Actor(nn.Module):
    def __init__(self,s,a):
        super().__init__(); self.net=nn.Sequential(nn.Linear(s,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,a))
    def forward(self,x): return self.net(x)

class Critic(nn.Module):
    def __init__(self,s):
        super().__init__(); self.net=nn.Sequential(nn.Linear(s,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1))
    def forward(self,x): return self.net(x)

class ActorCriticAgent(nn.Module):
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
    def __init__(self,state_size,action_size,device,timestep=TIMESTEP,epochs=EPOCHS,ppo_clip=PPO_CLIP,obs_clip=OBS_CLIP,disc_coef=DISC_COEF,lr_actor=LR_ACTOR,lr_critic=LR_CRITIC):
        self.update_timestep,self.epochs,self.ppo_clip,self.disc_coef=timestep,epochs,ppo_clip,disc_coef
        self.device,self.state_dim,self.action_dim=device,state_size,action_size
        self.actions,self.states,self.logprobs,self.rewards,self.state_values,self.dones,self.env_ids=[],[],[],[],[],[],[]
        self.states_mean,self.states_var,self.states_count,self.obs_clip=np.zeros(state_size,dtype=np.float32),np.ones(state_size,dtype=np.float32),1e-4,obs_clip
        self.policy=ActorCriticAgent(state_size, action_size).to(device)
        self.policy_old=ActorCriticAgent(state_size, action_size).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer=torch.optim.Adam([{"params":self.policy.actor.parameters(),"lr":lr_actor},{"params":self.policy.critic.parameters(),"lr":lr_critic}])
    def clear(self):
        self.actions.clear(); self.states.clear(); self.logprobs.clear(); self.rewards.clear(); self.state_values.clear(); self.dones.clear(); self.env_ids.clear()
    def act(self,state,env_id=0):
        with torch.no_grad():
            s=torch.as_tensor(state,device=self.device,dtype=torch.float32); a,lp,v=self.policy_old.act(s)
        self.states.append(s); self.actions.append(a); self.logprobs.append(lp); self.state_values.append(v); self.env_ids.append(env_id); return a.item()
    def act_batch(self,states):
        with torch.no_grad():
            s=torch.as_tensor(np.asarray(states),device=self.device,dtype=torch.float32); a,lp,v=self.policy_old.act(s)
        for i in range(len(states)):
            self.states.append(s[i]); self.actions.append(a[i]); self.logprobs.append(lp[i]); self.state_values.append(v[i]); self.env_ids.append(i)
        return a.detach().cpu().numpy().astype(int)
    def update(self, returns, writer=None, time_step=None):
        s=torch.stack(self.states).detach().to(self.device)
        a=torch.stack(self.actions).detach().to(self.device).view(-1)
        old_lp=torch.stack(self.logprobs).detach().to(self.device).view(-1)
        old_v=torch.stack(self.state_values).detach().to(self.device).view(-1)
        returns=returns.detach().to(self.device).view(-1); assert returns.shape == old_v.shape, f"returns {returns.shape}, old_v {old_v.shape}"
        adv=returns-old_v; adv=(adv-adv.mean())/(adv.std()+1e-7)
        for _ in range(self.epochs):
            lp,v,ent=self.policy.evaluate(s,a); lp=lp.view(-1);v=v.view(-1);ent=ent.view(-1)
            r=torch.exp(lp-old_lp)
            actor_loss=-torch.min(r * adv, torch.clamp(r, 1 - self.ppo_clip, 1 + self.ppo_clip) * adv)
            critic_loss=F.smooth_l1_loss(v,returns)
            loss=actor_loss+VF_COEF*critic_loss-ENT_COEF*ent
            self.optimizer.zero_grad(); loss.mean().backward(); self.optimizer.step()
            if writer and time_step: # add actor loss, critic loss, entropy tracking?
                writer.add_scalar("Loss/Total",loss.mean().item(),time_step)
                writer.add_scalar("Loss/Actor",actor_loss.mean().item(),time_step)
                writer.add_scalar("Loss/Critic",critic_loss.item(),time_step)
                writer.add_scalar("Policy/Entropy",ent.mean().item(),time_step)
        self.policy_old.load_state_dict(self.policy.state_dict())
    def learn(self,writer=None,time_step=None):
        r=[0.0]*len(self.rewards); r_gamma={}
        for idx in range(len(self.rewards)-1,-1,-1):
            eid=self.env_ids[idx]
            if self.dones[idx]: r_gamma[eid]=0.0
            r_gamma[eid]=self.rewards[idx]+(self.disc_coef * r_gamma.get(eid, 0.0))
            r[idx]=r_gamma[eid]
        r=torch.tensor(r,dtype=torch.float32,device=self.device)
        self.update(r,writer,time_step); self.clear()
    def train_agent(self, env, episodes=EPISODES, ep_max_steps=EP_MAX_STEPS, ep_reward_penalty=EP_REWARD_PENALTY, ep_timeout_penalty=EP_TIMEOUT_PENALTY, writer=None,seed=43):
        time_step,scores=0,[]
        for ep in range(1,episodes+1):
            if seed is not None: state, _ = env.reset(seed=seed)
            total=0; episode_length=ep_max_steps
            for t in range(1,ep_max_steps+1):
                action=self.act(state); state,reward,terminated,truncated,_=env.step(action)
                reward+=ep_reward_penalty; done=terminated or truncated
                if t==ep_max_steps and not done: reward+=ep_timeout_penalty; done=True
                self.rewards.append(reward); self.dones.append(done); time_step+=1; total+=reward
                if len(self.rewards)>=self.update_timestep: self.learn(writer,time_step)
                if done: episode_length=t; break
            scores.append(total); avg=np.mean(scores[-100:])
            if writer:
                writer.add_scalar("Train/Episode_Return",total,ep); writer.add_scalar("Train/Average_Return_100",avg,ep); writer.add_scalar("Train/Episode_Length",episode_length,ep); writer.add_scalar("Train/Total_Timesteps",time_step,ep)
            print(f"\rEpisode {ep}\tAverage Score: {avg:.2f}",end="",flush=True)
            if ep%100==0: print(f"\rEpisode {ep}\tAverage Score: {avg:.2f}",flush=True)
            if avg>=200: print(f"\nEnvironment solved in {ep:d} episodes!\tAverage Score: {avg:.2f}",flush=True); break
        if self.rewards: self.learn(writer,time_step)
        return scores
    def train_agent_parallel(self, make_env, episodes=EPISODES, ep_max_steps=EP_MAX_STEPS, ep_reward_penalty=EP_REWARD_PENALTY, ep_timeout_penalty=EP_TIMEOUT_PENALTY, parallel_envs=PARALLEL_ENVS, rollout_workers=ROLLOUT_WORKERS, writer=None, seed=43):
        envs=[make_env() for _ in range(parallel_envs)]
        executor=ThreadPoolExecutor(max_workers=max(1,rollout_workers)) if rollout_workers>1 else None
        states=[env.reset(seed=seed)[0] for env in envs]
        totals=[0.0]*parallel_envs; lengths=[0]*parallel_envs; scores=[]; time_step=0; completed=0
        print(f"Collecting rollouts with parallel_envs={parallel_envs}, rollout_workers={rollout_workers}",flush=True)
        try:
            while completed<episodes:
                actions=self.act_batch(states)
                if executor: results=list(executor.map(lambda pair: pair[0].step(int(pair[1])), zip(envs,actions)))
                else: results=[envs[i].step(int(actions[i])) for i in range(parallel_envs)]
                for i,(next_state,reward,terminated,truncated,_) in enumerate(results):
                    lengths[i]+=1; reward+=ep_reward_penalty; done=terminated or truncated
                    if lengths[i]>=ep_max_steps and not done: reward+=ep_timeout_penalty; done=True
                    self.rewards.append(reward); self.dones.append(done); totals[i]+=reward; time_step+=1
                    if done:
                        completed+=1; scores.append(totals[i]); avg=np.mean(scores[-100:])
                        if writer:
                            writer.add_scalar("Train/Episode_Return",totals[i],completed); writer.add_scalar("Train/Average_Return_100",avg,completed); writer.add_scalar("Train/Episode_Length",lengths[i],completed); writer.add_scalar("Train/Total_Timesteps",time_step,completed)
                        print(f"\rEpisode {completed}\tAverage Score: {avg:.2f}",end="",flush=True)
                        if completed%100==0: print(f"\rEpisode {completed}\tAverage Score: {avg:.2f}",flush=True)
                        totals[i]=0.0; lengths[i]=0; states[i]=envs[i].reset()[0]
                        if avg>=200 and completed>=100:
                            print(f"\nEnvironment solved in {completed:d} episodes!\tAverage Score: {avg:.2f}",flush=True); completed=episodes; break
                    else:
                        states[i]=next_state
                if len(self.rewards)>=self.update_timestep: self.learn(writer,time_step)
        finally:
            if executor: executor.shutdown(wait=True)
            for env in envs: env.close()
        if self.rewards: self.learn(writer,time_step)
        return scores
    def save(self,p): self.policy.save_model(p)
    def load(self,p): self.policy.load_model(p,self.device); self.policy_old.load_state_dict(self.policy.state_dict())

def train(args):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu");
    set_seed(args.seed)
    writer=SummaryWriter(args.log_dir)
    probe_env=gym.make("LunarLander-v2")
    probe_env.reset(seed=args.seed)
    probe_env.action_space.seed(args.seed)
    agent=PPOAgent(
        state_size=probe_env.observation_space.shape[0],
        action_size=probe_env.action_space.n,
        device=device,
        timestep=args.timestep,
        epochs=args.epochs,
        ppo_clip=args.ppo_clip,
        disc_coef=args.disc_coef,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
    )
    probe_env.close()
    env = gym.make("LunarLander-v2")
    env.reset(seed=args.seed)
    env.action_space.seed(args.seed)
    if args.parallel_envs>0: ## CHANGING TO 0 as a TEST
        scores=agent.train_agent_parallel(
            make_env=lambda: gym.make("LunarLander-v2"),
            episodes=args.episodes,
            ep_max_steps=args.ep_max_steps,
            ep_reward_penalty=args.ep_reward_penalty,
            ep_timeout_penalty=args.ep_timeout_penalty,
            parallel_envs=args.parallel_envs,
            rollout_workers=args.rollout_workers,
            writer=writer,
            seed=args.seed
        )
    else:
        env=gym.make("LunarLander-v2")
        scores=agent.train_agent(
            env=env,
            episodes=args.episodes,
            ep_max_steps=args.ep_max_steps,
            ep_reward_penalty=args.ep_reward_penalty,
            ep_timeout_penalty=args.ep_timeout_penalty,
            writer=writer,
            seed=args.seed
        )
        env.close()
    agent.save(args.model); writer.close()
    plt.plot(scores); plt.xlabel("Episode"); plt.ylabel("Reward"); plt.title("PPO LunarLander Training Scores")
    os.makedirs(args.log_dir, exist_ok=True)
    plt.savefig(os.path.join(args.log_dir, "training_scores.png"), dpi=200)
    plt.close()

def render(episodes=5,model_path=MODEL_PATH):
    print("render episodes =", episodes)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env=gym.make("LunarLander-v2",render_mode="human"); agent=PPOAgent(env.observation_space.shape[0], env.action_space.n, device); agent.load(model_path); agent.policy.eval()
    for ep in range(episodes):
        state,_=env.reset(seed=SEED); done=False; total=0
        while not done:
            s=torch.as_tensor(state,dtype=torch.float32,device=device)
            with torch.no_grad(): action=torch.argmax(agent.policy.actor(s)).item()
            state,reward,terminated,truncated,_=env.step(action); done=terminated or truncated; total+=reward
        print(f"Episode {ep+1}: reward = {total:.2f}")
    env.close()

def play():
    from gymnasium.utils.play import play
    env=gym.make("LunarLander-v2",render_mode="rgb_array")
    keys_to_action = {
        (): 0,  # no key
        ("s",): 0,  # noop
        ("a",): 1,  # left engine
        ("w",): 2,  # main engine
        ("d",): 3,  # right engine

        ("w", "a"): 2,  # W+A still maps to main engine
        ("w", "d"): 2,  # W+D still maps to main engine
    }
    play(env,keys_to_action=keys_to_action,noop=2,fps=60); env.close()

def export_pngs(log_dir="runs/ppo_lunar_lander",out_dir="tensorboard_pngs"):
    from pathlib import Path
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    Path(out_dir).mkdir(exist_ok=True); ea=EventAccumulator(log_dir); ea.Reload()
    for tag in ea.Tags().get("scalars",[]):
        e=ea.Scalars(tag); x=[v.step for v in e]; y=[v.value for v in e]
        plt.figure(); plt.plot(x,y); plt.xlabel("Step"); plt.ylabel(tag); plt.title(tag); plt.tight_layout()
        plt.savefig(os.path.join(out_dir,tag.replace("/","_")+".png"),dpi=200); plt.close()

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=["train","render","play","export"],nargs="?",default="train")
    parser.add_argument("--timestep",type=int,default=TIMESTEP)
    parser.add_argument("--epochs",type=int,default=EPOCHS)
    parser.add_argument("--ppo_clip", type=float, default=PPO_CLIP)
    parser.add_argument("--disc_coef", type=float, default=DISC_COEF)
    parser.add_argument("--lr-actor",type=float,default=LR_ACTOR)
    parser.add_argument("--lr-critic",type=float,default=LR_CRITIC)
    parser.add_argument("--episodes",type=int,default=EPISODES)
    parser.add_argument("--ep-max-steps",type=int,default=EP_MAX_STEPS)
    parser.add_argument("--ep-reward-penalty",type=float,default=EP_REWARD_PENALTY)
    parser.add_argument("--ep-timeout-penalty",type=float,default=EP_TIMEOUT_PENALTY)
    parser.add_argument("--parallel-envs",type=int,default=PARALLEL_ENVS)
    parser.add_argument("--rollout-workers",type=int,default=ROLLOUT_WORKERS)
    parser.add_argument("--log-dir",default="runs/ppo_lunar_lander")
    parser.add_argument("--model",default=MODEL_PATH)
    parser.add_argument("--seed",type=int,default=SEED)
    args=parser.parse_args()
    {
    "train": lambda: train(args=args),
    "render": lambda: render(model_path=args.model),
    "play": play,
    "export": lambda: export_pngs(args.log_dir),
    }[args.mode]()
