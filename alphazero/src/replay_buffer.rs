use tch::{ Tensor, Kind, Device };//, autocast };


pub struct ReplayBuffer {
    states: Tensor,
    probs: Tensor,
    rewards: Tensor,
    episode_states: Tensor,
    episode_probs: Tensor,
    capacity: i64,
    episode_cntr: i64,
    buffer_ptr: i64,
    n_episodes: i64,
    full: bool,
}



impl ReplayBuffer {
    pub fn new(capacity: i64, input_dim: i64, n_actions: i64) -> Self {
        ReplayBuffer {
            states: Tensor::zeros(&[capacity, input_dim], (Kind::Int, Device::Cpu)),
            probs: Tensor::zeros(&[capacity, n_actions], (Kind::Float, Device::Cpu)),
            rewards: Tensor::zeros(&[capacity, 1], (Kind::Float, Device::Cpu)),
            episode_states: Tensor::zeros(&[200, input_dim], (Kind::Float, Device::Cpu)),
            episode_probs: Tensor::zeros(&[200, n_actions], (Kind::Float, Device::Cpu)),
            capacity,
            episode_cntr: 0,
            buffer_ptr: 0,
            n_episodes: 0,
            full: false,
        }
    }

    pub fn push(&mut self, state: Tensor, probs: Tensor) {
        self.episode_states.slice(0, self.episode_cntr, self.episode_cntr + 1, 1).copy_(&state);
        self.episode_probs.slice(0, self.episode_cntr, self.episode_cntr + 1, 1).copy_(&probs);
        self.episode_cntr += 1;
    }

    pub fn store_episode(&mut self, reward: i32) -> bool {
        // Reward is -1, 0, 1
        // To create reward tensor for episode, backtrack through episode 
        // and add reward for each state and prob pair multiplying by -1
        // each time.
        self.n_episodes += 1;

        let prev_buffer_ptr = match self.capacity < self.episode_cntr + self.buffer_ptr {
            true => {
                self.full = true;
                0
            },
            false => self.buffer_ptr,
        };

        self.buffer_ptr += self.episode_cntr;

        if reward == 0 {
            self.states.slice(0, prev_buffer_ptr, self.buffer_ptr, 1)
                       .copy_(&self.episode_states.slice(0, 0, self.episode_cntr, 1));

            self.probs.slice(0, prev_buffer_ptr, self.buffer_ptr, 1)
                      .copy_(&self.episode_probs.slice(0, 0, self.episode_cntr, 1));

            self.rewards.slice(0, prev_buffer_ptr, self.buffer_ptr, 1)
                        .copy_(&Tensor::zeros(&[self.episode_cntr, 1], (Kind::Float, Device::Cpu)));

        }
        else {
            let mut rewards = vec![reward as f32; self.episode_cntr as usize];
            for idx in (0..self.episode_cntr - 1).rev() {
                rewards[idx as usize] *= -1.0;
            }
            let reward_tensor = Tensor::of_slice(&rewards).to_kind(Kind::Float).to_device(Device::Cpu).unsqueeze(1);

            self.states.slice(0, prev_buffer_ptr, self.buffer_ptr, 1)
                       .copy_(&self.episode_states.slice(0, 0, self.episode_cntr, 1));

            self.probs.slice(0, prev_buffer_ptr, self.buffer_ptr, 1)
                      .copy_(&self.episode_probs.slice(0, 0, self.episode_cntr, 1));

            self.rewards.slice(0, prev_buffer_ptr, self.buffer_ptr, 1)
                        .copy_(&reward_tensor);
        }


        self.episode_cntr = 0;

        self.episode_states = self.episode_states.zero_();
        self.episode_probs  = self.episode_probs.zero_();

        // Easy way to avoid training on basically no data in beggining.
        self.n_episodes % 5 == 4
    }

    pub fn sample(&self, batch_size: i64) -> (Tensor, Tensor, Tensor) {
        let max_idx = match self.full {
            true => self.capacity,
            false => self.buffer_ptr,
        };

        let idxs    = Tensor::randint(max_idx, &[batch_size], (Kind::Int, Device::Cpu));
        let states  = self.states.index_select(0, &idxs).to_device(Device::cuda_if_available());
        let probs   = self.probs.index_select(0, &idxs).to_device(Device::cuda_if_available());
        let rewards = self.rewards.index_select(0, &idxs).to_device(Device::cuda_if_available());

        (states, probs, rewards)
    }
}



pub struct ReplayBufferMT {
    states: Tensor,
    probs: Tensor,
    rewards: Tensor,
    episode_states: Tensor,
    episode_probs: Tensor,
    capacity: i64,
    episode_cntr: Vec<i64>,
    buffer_ptr: i64,
    n_episodes: i64,
    full: bool,
}



impl ReplayBufferMT {
    pub fn new(capacity: i64, input_dim: i64, n_actions: i64, num_threads: i64) -> Self {
        ReplayBufferMT {
            states: Tensor::zeros(&[capacity, input_dim], (Kind::Int, Device::Cpu)),
            probs: Tensor::zeros(&[capacity, n_actions], (Kind::Float, Device::Cpu)),
            rewards: Tensor::zeros(&[capacity, 1], (Kind::Float, Device::Cpu)),
            episode_states: Tensor::zeros(&[num_threads, 200, input_dim], (Kind::Float, Device::Cpu)),
            episode_probs: Tensor::zeros(&[num_threads, 200, n_actions], (Kind::Float, Device::Cpu)),
            capacity,
            episode_cntr: vec![0; num_threads as usize],
            buffer_ptr: 0,
            n_episodes: 0,
            full: false,
        }
    }

    pub fn push(&mut self, state: Tensor, probs: Tensor, thread_idx: usize) {
        self.episode_states.slice(0, thread_idx as i64, thread_idx as i64 + 1, 1)
                           .slice(1, self.episode_cntr[thread_idx], self.episode_cntr[thread_idx] + 1, 1)
                           .copy_(&state);
        self.episode_probs.slice(0, thread_idx as i64, thread_idx as i64 + 1, 1)
                          .slice(1, self.episode_cntr[thread_idx], self.episode_cntr[thread_idx] + 1, 1)
                          .copy_(&probs);
        self.episode_cntr[thread_idx] += 1;
    }

    pub fn store_episode(&mut self, reward: i32, thread_idx: usize) -> bool {
        // Reward is -1, 0, 1
        // To create reward tensor for episode, backtrack through episode 
        // and add reward for each state and prob pair multiplying by -1
        // each time.
        self.n_episodes += 1;

        let prev_buffer_ptr = match self.capacity < self.episode_cntr[thread_idx] + self.buffer_ptr {
            true => {
                self.full = true;
                0
            },
            false => self.buffer_ptr,
        };

        self.buffer_ptr += self.episode_cntr[thread_idx];

        if reward == 0 {
            self.states.slice(0, prev_buffer_ptr, self.buffer_ptr, 1)
                       .copy_(&self.episode_states.get(thread_idx as i64)
                                                  .slice(0, 0, self.episode_cntr[thread_idx], 1)
                           );

            self.probs.slice(0, prev_buffer_ptr, self.buffer_ptr, 1)
                      .copy_(&self.episode_probs.get(thread_idx as i64)
                                                .slice(0, 0, self.episode_cntr[thread_idx], 1)
                           );
            self.rewards.slice(0, prev_buffer_ptr, self.buffer_ptr, 1)
                        .copy_(&Tensor::zeros(&[self.episode_cntr[thread_idx], 1], (Kind::Float, Device::Cpu)));
        }
        else {
            let mut rewards = vec![reward as f32; self.episode_cntr[thread_idx] as usize];
            for idx in (0..self.episode_cntr[thread_idx] - 1).rev() {
                rewards[idx as usize] *= -1.0;
            }
            let reward_tensor = Tensor::of_slice(&rewards).to_kind(Kind::Float).to_device(Device::Cpu).unsqueeze(1);

            self.states.slice(0, prev_buffer_ptr, self.buffer_ptr, 1)
                       .copy_(&self.episode_states.get(thread_idx as i64)
                                                  .slice(0, 0, self.episode_cntr[thread_idx], 1)
                           );

            self.probs.slice(0, prev_buffer_ptr, self.buffer_ptr, 1)
                      .copy_(&self.episode_probs.get(thread_idx as i64)
                                                .slice(0, 0, self.episode_cntr[thread_idx], 1)
                           );

            self.rewards.slice(0, prev_buffer_ptr, self.buffer_ptr, 1)
                        .copy_(&reward_tensor);
        }


        self.episode_cntr[thread_idx] = 0;

        self.episode_states = self.episode_states.zero_();
        self.episode_probs  = self.episode_probs.zero_();

        // Easy way to avoid training on basically no data in beggining.
        self.n_episodes % 100 == 99
    }

    pub fn sample(&self, batch_size: i64) -> (Tensor, Tensor, Tensor) {
        let max_idx = match self.full {
            true => self.capacity,
            false => self.buffer_ptr,
        };

        let idxs    = Tensor::randint(max_idx, &[batch_size], (Kind::Int, Device::Cpu));
        let states  = self.states.index_select(0, &idxs).to_device(Device::cuda_if_available());
        let probs   = self.probs.index_select(0, &idxs).to_device(Device::cuda_if_available());
        let rewards = self.rewards.index_select(0, &idxs).to_device(Device::cuda_if_available());

        (states, probs, rewards)
    }
}
