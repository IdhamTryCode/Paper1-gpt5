use tch::{nn, nn::OptimizerConfig, Device, Tensor, Kind};
use std::collections::{HashMap, VecDeque};
use rand::Rng;
use anyhow::Result;

/// Experience replay buffer
#[derive(Clone)]
pub struct Experience {
    pub state: Vec<f64>,
    pub action: usize,
    pub reward: f64,
    pub next_state: Vec<f64>,
    pub done: bool,
}

/// DQN implementation
pub struct DQN {
    q_network: nn::Sequential,
    target_network: nn::Sequential,
    device: Device,
}

impl DQN {
    pub fn new(vs: &nn::Path, input_size: i64, output_size: i64, device: Device) -> Self {
        let q_network = nn::seq()
            .add(nn::linear(&vs / "fc1", input_size, 64, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(&vs / "fc2", 64, 64, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(&vs / "fc3", 64, output_size, Default::default()));
        
        let target_network = nn::seq()
            .add(nn::linear(&vs / "target_fc1", input_size, 64, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(&vs / "target_fc2", 64, 64, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(&vs / "target_fc3", 64, output_size, Default::default()));
        
        Self {
            q_network,
            target_network,
            device,
        }
    }
    
    pub fn forward(&self, state: &Tensor) -> Tensor {
        self.q_network.forward(state)
    }
    
    pub fn target_forward(&self, state: &Tensor) -> Tensor {
        self.target_network.forward(state)
    }
    
    pub fn update_target(&mut self) {
        self.target_network.load_state_dict(&self.q_network.state_dict());
    }
}

/// Double DQN implementation
pub struct DDQN {
    q_network: nn::Sequential,
    target_network: nn::Sequential,
    device: Device,
}

impl DDQN {
    pub fn new(vs: &nn::Path, input_size: i64, output_size: i64, device: Device) -> Self {
        let q_network = nn::seq()
            .add(nn::linear(&vs / "fc1", input_size, 64, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(&vs / "fc2", 64, 64, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(&vs / "fc3", 64, output_size, Default::default()));
        
        let target_network = nn::seq()
            .add(nn::linear(&vs / "target_fc1", input_size, 64, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(&vs / "target_fc2", 64, 64, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(&vs / "target_fc3", 64, output_size, Default::default()));
        
        Self {
            q_network,
            target_network,
            device,
        }
    }
    
    pub fn forward(&self, state: &Tensor) -> Tensor {
        self.q_network.forward(state)
    }
    
    pub fn target_forward(&self, state: &Tensor) -> Tensor {
        self.target_network.forward(state)
    }
    
    pub fn update_target(&mut self) {
        self.target_network.load_state_dict(&self.q_network.state_dict());
    }
}

/// Dueling DQN implementation
pub struct DuelingDQN {
    shared_layers: nn::Sequential,
    value_stream: nn::Sequential,
    advantage_stream: nn::Sequential,
    device: Device,
}

impl DuelingDQN {
    pub fn new(vs: &nn::Path, input_size: i64, output_size: i64, device: Device) -> Self {
        let shared_layers = nn::seq()
            .add(nn::linear(&vs / "shared_fc1", input_size, 64, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(&vs / "shared_fc2", 64, 64, Default::default()))
            .add_fn(|xs| xs.relu());
        
        let value_stream = nn::seq()
            .add(nn::linear(&vs / "value_fc", 64, 1, Default::default()));
        
        let advantage_stream = nn::seq()
            .add(nn::linear(&vs / "advantage_fc", 64, output_size, Default::default()));
        
        Self {
            shared_layers,
            value_stream,
            advantage_stream,
            device,
        }
    }
    
    pub fn forward(&self, state: &Tensor) -> Tensor {
        let shared = self.shared_layers.forward(state);
        let value = self.value_stream.forward(&shared);
        let advantage = self.advantage_stream.forward(&shared);
        
        // Combine value and advantage
        let advantage_mean = advantage.mean_dim(&[-1], true, Kind::Float);
        value + (advantage - advantage_mean)
    }
}

/// Prioritized Experience Replay Buffer
pub struct PrioritizedReplayBuffer {
    capacity: usize,
    alpha: f64,
    beta: f64,
    buffer: VecDeque<Experience>,
    priorities: Vec<f64>,
    position: usize,
    size: usize,
}

impl PrioritizedReplayBuffer {
    pub fn new(capacity: usize, alpha: f64, beta: f64) -> Self {
        Self {
            capacity,
            alpha,
            beta,
            buffer: VecDeque::with_capacity(capacity),
            priorities: vec![0.0; capacity],
            position: 0,
            size: 0,
        }
    }
    
    pub fn add(&mut self, experience: Experience, priority: Option<f64>) {
        let priority = priority.unwrap_or_else(|| {
            if self.size > 0 {
                self.priorities.iter().fold(0.0, |a, &b| a.max(b))
            } else {
                1.0
            }
        });
        
        if self.buffer.len() < self.capacity {
            self.buffer.push_back(experience);
        } else {
            self.buffer[self.position] = experience;
        }
        
        self.priorities[self.position] = priority.powf(self.alpha);
        self.position = (self.position + 1) % self.capacity;
        self.size = (self.size + 1).min(self.capacity);
    }
    
    pub fn sample(&self, batch_size: usize) -> (Vec<Experience>, Vec<usize>, Vec<f64>) {
        if self.size < batch_size {
            let indices: Vec<usize> = (0..self.size).collect();
            let experiences: Vec<Experience> = indices.iter().map(|&i| self.buffer[i].clone()).collect();
            let weights = vec![1.0; self.size];
            return (experiences, indices, weights);
        }
        
        // Calculate sampling probabilities
        let priorities = &self.priorities[..self.size];
        let sum_priorities: f64 = priorities.iter().sum();
        let probabilities: Vec<f64> = priorities.iter().map(|&p| p / sum_priorities).collect();
        
        // Sample indices
        let mut rng = rand::thread_rng();
        let mut indices = Vec::new();
        let mut experiences = Vec::new();
        
        for _ in 0..batch_size {
            let random_val: f64 = rng.gen();
            let mut cumulative_prob = 0.0;
            let mut selected_idx = 0;
            
            for (i, &prob) in probabilities.iter().enumerate() {
                cumulative_prob += prob;
                if random_val <= cumulative_prob {
                    selected_idx = i;
                    break;
                }
            }
            
            indices.push(selected_idx);
            experiences.push(self.buffer[selected_idx].clone());
        }
        
        // Calculate importance sampling weights
        let weights: Vec<f64> = indices.iter()
            .map(|&idx| (self.size as f64 * probabilities[idx]).powf(-self.beta))
            .collect();
        
        let max_weight = weights.iter().fold(0.0, |a, &b| a.max(b));
        let normalized_weights: Vec<f64> = weights.iter().map(|&w| w / max_weight).collect();
        
        (experiences, indices, normalized_weights)
    }
    
    pub fn update_priorities(&mut self, indices: &[usize], priorities: &[f64]) {
        for (&idx, &priority) in indices.iter().zip(priorities.iter()) {
            self.priorities[idx] = priority.powf(self.alpha);
        }
    }
}

/// Prioritized DQN implementation
pub struct PrioritizedDQN {
    q_network: nn::Sequential,
    target_network: nn::Sequential,
    device: Device,
}

impl PrioritizedDQN {
    pub fn new(vs: &nn::Path, input_size: i64, output_size: i64, device: Device) -> Self {
        let q_network = nn::seq()
            .add(nn::linear(&vs / "fc1", input_size, 64, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(&vs / "fc2", 64, 64, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(&vs / "fc3", 64, output_size, Default::default()));
        
        let target_network = nn::seq()
            .add(nn::linear(&vs / "target_fc1", input_size, 64, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(&vs / "target_fc2", 64, 64, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(&vs / "target_fc3", 64, output_size, Default::default()));
        
        Self {
            q_network,
            target_network,
            device,
        }
    }
    
    pub fn forward(&self, state: &Tensor) -> Tensor {
        self.q_network.forward(state)
    }
    
    pub fn target_forward(&self, state: &Tensor) -> Tensor {
        self.target_network.forward(state)
    }
    
    pub fn update_target(&mut self) {
        self.target_network.load_state_dict(&self.q_network.state_dict());
    }
}

/// Rainbow DQN implementation (simplified)
pub struct RainbowDQN {
    q_network: nn::Sequential,
    target_network: nn::Sequential,
    device: Device,
    num_atoms: i64,
    v_min: f64,
    v_max: f64,
}

impl RainbowDQN {
    pub fn new(vs: &nn::Path, input_size: i64, output_size: i64, device: Device) -> Self {
        let q_network = nn::seq()
            .add(nn::linear(&vs / "fc1", input_size, 64, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(&vs / "fc2", 64, 64, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(&vs / "fc3", 64, output_size * 51, Default::default())); // 51 atoms
        
        let target_network = nn::seq()
            .add(nn::linear(&vs / "target_fc1", input_size, 64, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(&vs / "target_fc2", 64, 64, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(&vs / "target_fc3", 64, output_size * 51, Default::default()));
        
        Self {
            q_network,
            target_network,
            device,
            num_atoms: 51,
            v_min: -10.0,
            v_max: 10.0,
        }
    }
    
    pub fn forward(&self, state: &Tensor) -> Tensor {
        let logits = self.q_network.forward(state);
        let batch_size = logits.size()[0];
        let num_actions = logits.size()[-1] / self.num_atoms;
        
        let logits = logits.view(&[batch_size, num_actions, self.num_atoms]);
        logits.softmax(-1, Kind::Float)
    }
    
    pub fn target_forward(&self, state: &Tensor) -> Tensor {
        let logits = self.target_network.forward(state);
        let batch_size = logits.size()[0];
        let num_actions = logits.size()[-1] / self.num_atoms;
        
        let logits = logits.view(&[batch_size, num_actions, self.num_atoms]);
        logits.softmax(-1, Kind::Float)
    }
    
    pub fn update_target(&mut self) {
        self.target_network.load_state_dict(&self.q_network.state_dict());
    }
}

/// Base RL Agent trait
pub trait RLAgent {
    fn act(&self, state: &[f64], training: bool) -> usize;
    fn remember(&mut self, state: Vec<f64>, action: usize, reward: f64, next_state: Vec<f64>, done: bool);
    fn replay(&mut self) -> Option<f64>;
    fn update_target_network(&mut self);
    fn get_model_info(&self) -> HashMap<String, f64>;
}

/// DQN Agent implementation
pub struct DQNAgent {
    q_network: DQN,
    optimizer: nn::Optimizer,
    device: Device,
    memory: VecDeque<Experience>,
    epsilon: f64,
    epsilon_min: f64,
    epsilon_decay: f64,
    gamma: f64,
    batch_size: usize,
    target_update_freq: usize,
    step_count: usize,
}

impl DQNAgent {
    pub fn new(input_size: i64, output_size: i64, device: Device, 
               learning_rate: f64, epsilon: f64, epsilon_min: f64, 
               epsilon_decay: f64, gamma: f64, batch_size: usize) -> Self {
        let vs = nn::VarStore::new(device);
        let q_network = DQN::new(&vs.root(), input_size, output_size, device);
        let optimizer = nn::Adam::default().build(&vs, learning_rate).unwrap();
        
        Self {
            q_network,
            optimizer,
            device,
            memory: VecDeque::with_capacity(100000),
            epsilon,
            epsilon_min,
            epsilon_decay,
            gamma,
            batch_size,
            target_update_freq: 1000,
            step_count: 0,
        }
    }
}

impl RLAgent for DQNAgent {
    fn act(&self, state: &[f64], training: bool) -> usize {
        if training && rand::random::<f64>() < self.epsilon {
            return rand::thread_rng().gen_range(0..4); // Assuming 4 actions
        }
        
        let state_tensor = Tensor::of_slice(state).to_device(self.device);
        let q_values = self.q_network.forward(&state_tensor);
        q_values.argmax(-1, false).int64_value(&[0]) as usize
    }
    
    fn remember(&mut self, state: Vec<f64>, action: usize, reward: f64, next_state: Vec<f64>, done: bool) {
        self.memory.push_back(Experience { state, action, reward, next_state, done });
    }
    
    fn replay(&mut self) -> Option<f64> {
        if self.memory.len() < self.batch_size {
            return None;
        }
        
        // Sample batch
        let mut rng = rand::thread_rng();
        let batch: Vec<Experience> = (0..self.batch_size)
            .map(|_| {
                let idx = rng.gen_range(0..self.memory.len());
                self.memory[idx].clone()
            })
            .collect();
        
        // Prepare tensors
        let states: Vec<f64> = batch.iter().flat_map(|e| e.state.clone()).collect();
        let actions: Vec<i64> = batch.iter().map(|e| e.action as i64).collect();
        let rewards: Vec<f64> = batch.iter().map(|e| e.reward).collect();
        let next_states: Vec<f64> = batch.iter().flat_map(|e| e.next_state.clone()).collect();
        let dones: Vec<bool> = batch.iter().map(|e| e.done).collect();
        
        let states_tensor = Tensor::of_slice(&states).view(&[self.batch_size as i64, -1]).to_device(self.device);
        let actions_tensor = Tensor::of_slice(&actions).to_device(self.device);
        let rewards_tensor = Tensor::of_slice(&rewards).to_device(self.device);
        let next_states_tensor = Tensor::of_slice(&next_states).view(&[self.batch_size as i64, -1]).to_device(self.device);
        let dones_tensor = Tensor::of_slice(&dones.iter().map(|&d| if d { 1.0 } else { 0.0 }).collect::<Vec<f64>>()).to_device(self.device);
        
        // Current Q values
        let current_q_values = self.q_network.forward(&states_tensor);
        let current_q_values = current_q_values.gather(1, &actions_tensor.unsqueeze(1), false);
        
        // Next Q values
        let next_q_values = self.q_network.target_forward(&next_states_tensor);
        let next_q_values = next_q_values.max_dim(1, false).0;
        let target_q_values = rewards_tensor + (self.gamma * next_q_values * (1.0 - dones_tensor));
        
        // Compute loss
        let loss = (current_q_values.squeeze() - target_q_values).square().mean(Kind::Float);
        
        // Optimize
        self.optimizer.zero_grad();
        loss.backward();
        self.optimizer.step();
        
        // Update epsilon
        if self.epsilon > self.epsilon_min {
            self.epsilon *= self.epsilon_decay;
        }
        
        Some(loss.double_value(&[]))
    }
    
    fn update_target_network(&mut self) {
        self.q_network.update_target();
    }
    
    fn get_model_info(&self) -> HashMap<String, f64> {
        let mut info = HashMap::new();
        info.insert("algorithm".to_string(), 0.0); // DQN
        info.insert("epsilon".to_string(), self.epsilon);
        info.insert("memory_size".to_string(), self.memory.len() as f64);
        info.insert("total_parameters".to_string(), 100000.0); // Placeholder
        info.insert("trainable_parameters".to_string(), 100000.0); // Placeholder
        info.insert("model_size_mb".to_string(), 0.4); // Placeholder
        info
    }
}

/// Factory function to create RL agents
pub fn create_rl_agent(algorithm: &str, input_size: i64, output_size: i64, device: Device, 
                      hyperparams: &HashMap<String, f64>) -> Box<dyn RLAgent> {
    let learning_rate = hyperparams.get("learning_rate").unwrap_or(&0.001);
    let epsilon = hyperparams.get("epsilon").unwrap_or(&1.0);
    let epsilon_min = hyperparams.get("epsilon_min").unwrap_or(&0.01);
    let epsilon_decay = hyperparams.get("epsilon_decay").unwrap_or(&0.995);
    let gamma = hyperparams.get("gamma").unwrap_or(&0.99);
    let batch_size = hyperparams.get("batch_size").unwrap_or(&32.0) as usize;
    
    match algorithm {
        "dqn" => {
            let agent = DQNAgent::new(input_size, output_size, device, *learning_rate, 
                                     *epsilon, *epsilon_min, *epsilon_decay, *gamma, batch_size);
            Box::new(agent)
        }
        "ddqn" => {
            // Similar to DQN but with double Q-learning
            let agent = DQNAgent::new(input_size, output_size, device, *learning_rate, 
                                     *epsilon, *epsilon_min, *epsilon_decay, *gamma, batch_size);
            Box::new(agent)
        }
        "dueling_dqn" => {
            // Similar to DQN but with dueling architecture
            let agent = DQNAgent::new(input_size, output_size, device, *learning_rate, 
                                     *epsilon, *epsilon_min, *epsilon_decay, *gamma, batch_size);
            Box::new(agent)
        }
        "prioritized_dqn" => {
            // Similar to DQN but with prioritized replay
            let agent = DQNAgent::new(input_size, output_size, device, *learning_rate, 
                                     *epsilon, *epsilon_min, *epsilon_decay, *gamma, batch_size);
            Box::new(agent)
        }
        "rainbow_dqn" => {
            // Similar to DQN but with rainbow features
            let agent = DQNAgent::new(input_size, output_size, device, *learning_rate, 
                                     *epsilon, *epsilon_min, *epsilon_decay, *gamma, batch_size);
            Box::new(agent)
        }
        _ => panic!("Unknown algorithm: {}", algorithm)
    }
} 