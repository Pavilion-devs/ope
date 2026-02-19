import numpy as np
from typing import Dict, List, Tuple, Optional
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import torch
import os


class GridGuardAgent:
    """PPO agent for grid stabilization recommendations"""
    
    def __init__(self, env, model_dir: str = "./models"):
        self.env = env
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        self.model = None
        self._init_model()
        
    def _init_model(self):
        """Initialize PPO model"""
        
        policy_kwargs = {
            'net_arch': [256, 128, 64],
            'activation_fn': torch.nn.ReLU
        }
        
        self.model = PPO(
            "MlpPolicy",
            self.env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=self.model_dir + "/tensorboard"
        )
        
    def train(self, total_timesteps: int = 100000, eval_freq: int = 10000):
        """Train the agent"""
        
        eval_env = self.env
        
        callbacks = [
            EvalCallback(
                eval_env,
                best_model_save_path=self.model_dir + "/best_model",
                log_path=self.model_dir + "/eval_logs",
                eval_freq=eval_freq,
                deterministic=True
            ),
            CheckpointCallback(
                save_freq=eval_freq,
                save_path=self.model_dir + "/checkpoints",
                name_prefix="gridguard_ppo"
            )
        ]
        
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
        
        self.model.save(self.model_dir + "/final_model")
        
    def predict(self, obs: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, Optional[float]]:
        """Get action recommendation"""
        return self.model.predict(obs, deterministic=deterministic)
    
    def get_recommendation(self, obs: np.ndarray) -> Dict:
        """Get detailed recommendation"""
        action, _ = self.predict(obs)
        
        recommendations = self._action_to_recommendations(action)
        
        return {
            'action': action.tolist(),
            'recommendations': recommendations,
            'confidence': 0.85
        }
    
    def _action_to_recommendations(self, action: np.ndarray) -> List[Dict]:
        """Convert action vector to human-readable recommendations"""
        
        recommendations = []
        
        gen_adjustments = action[:5]
        for i, adj in enumerate(gen_adjustments):
            if abs(adj) > 0.1:
                recommendations.append({
                    'type': 'generation_redispatch',
                    'generator': f'Generator {i+1}',
                    'action': 'increase' if adj > 0 else 'decrease',
                    'magnitude': float(abs(adj))
                })
        
        tap_adjustments = action[5:]
        for i, adj in enumerate(tap_adjustments):
            if abs(adj) > 0.2:
                recommendations.append({
                    'type': 'tap_adjustment',
                    'transformer': f'Tap {i+1}',
                    'action': 'raise' if adj > 0 else 'lower',
                    'magnitude': float(abs(adj))
                })
                
        return recommendations
    
    def load(self, path: str):
        """Load trained model"""
        self.model = PPO.load(path, env=self.env)
        
    def save(self, path: str):
        """Save model"""
        self.model.save(path)


class RecommendationEngine:
    """Engine for generating ranked action recommendations"""
    
    def __init__(self):
        self.tier_actions = {
            'tier1': [
                {'action': 'reduce_industrial_load', 'impact': 'medium', 'disruption': 'low'},
                {'action': 'curtail_renewables', 'impact': 'medium', 'disruption': 'low'},
            ],
            'tier2': [
                {'action': 'activate_spinning_reserves', 'impact': 'high', 'disruption': 'medium'},
                {'action': 'start_peaker_plants', 'impact': 'high', 'disruption': 'medium'},
            ],
            'tier3': [
                {'action': 'manual_substation_decoupling', 'impact': 'very_high', 'disruption': 'high'},
                {'action': 'emergency_load_shedding', 'impact': 'very_high', 'disruption': 'high'},
            ]
        }
        
    def rank_actions(self, scenario: Dict, agent_rec: List[Dict]) -> List[Dict]:
        """Rank actions by impact and disruption"""
        
        ranked = []
        
        for rec in agent_rec:
            ranked.append({
                **rec,
                'priority': self._calculate_priority(rec)
            })
            
        ranked.sort(key=lambda x: x['priority'], reverse=True)
        
        return ranked
    
    def _calculate_priority(self, rec: Dict) -> float:
        """Calculate priority score"""
        impact_scores = {'low': 0.3, 'medium': 0.6, 'high': 0.8, 'very_high': 1.0}
        disruption_scores = {'low': 1.0, 'medium': 0.6, 'high': 0.3}
        
        impact = impact_scores.get(rec.get('impact', 'medium'), 0.5)
        disruption = disruption_scores.get(rec.get('disruption', 'low'), 0.7)
        
        return impact * disruption
    
    def get_tier_recommendation(self, severity: float) -> List[Dict]:
        """Get recommended tier based on severity"""
        
        if severity > 0.8:
            return self.tier_actions['tier3']
        elif severity > 0.5:
            return self.tier_actions['tier2']
        else:
            return self.tier_actions['tier1']
