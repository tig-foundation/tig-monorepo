use anyhow::{anyhow, Result};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use cudarc::{
    cublas::CudaBlas,
    cudnn::Cudnn,
    driver::{CudaModule, CudaSlice, CudaStream, CudaView, LaunchConfig, PushKernelArg},
    runtime::sys::cudaDeviceProp,
};
use flate2::{read::GzDecoder, write::GzEncoder, Compression};
use rand::{prelude::*, rngs::StdRng};
use serde::{
    de::{self, Visitor},
    Deserialize, Deserializer, Serialize, Serializer,
};
use serde_json::{from_value, Map, Value};
use std::{
    any::Any,
    fmt,
    io::{Read, Write},
    sync::Arc,
};

use crate::neuralnet::MLP;

const THREADS_PER_BLOCK: u32 = 1024;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Difficulty {
    pub num_hidden_layers: usize,
    #[cfg(not(feature = "hide_verification"))]
    pub accuracy_factor: u32,
    #[cfg(feature = "hide_verification")]
    accuracy_factor: u32,
}

impl From<Vec<i32>> for Difficulty {
    fn from(arr: Vec<i32>) -> Self {
        Self {
            num_hidden_layers: arr[0] as usize,
            accuracy_factor: arr[1] as u32,
        }
    }
}
impl Into<Vec<i32>> for Difficulty {
    fn into(self) -> Vec<i32> {
        vec![self.num_hidden_layers as i32, self.accuracy_factor as i32]
    }
}

impl_base64_serde! {
    Solution {
        weights: Vec<Vec<Vec<f32>>>,
        biases: Vec<Vec<f32>>,
        epochs_used: usize,
        bn_weights: Vec<Vec<f32>>,
        bn_biases: Vec<Vec<f32>>,
        bn_running_means: Vec<Vec<f32>>,
        bn_running_vars: Vec<Vec<f32>>,
    }
}

impl Solution {
    pub fn new() -> Self {
        Self {
            weights: Vec::new(),
            biases: Vec::new(),
            epochs_used: 0,
            bn_weights: Vec::new(),
            bn_biases: Vec::new(),
            bn_running_means: Vec::new(),
            bn_running_vars: Vec::new(),
        }
    }
}

pub struct Dataset {
    pub inputs: CudaSlice<f32>,
    pub targets_noisy: CudaSlice<f32>,
    pub targets_true_f: CudaSlice<f32>,

    pub train_size: usize,
    pub validation_size: usize,
    pub test_size: usize,

    pub input_dims: usize,
    pub output_dims: usize,
}

impl Dataset {
    pub fn train_inputs(&self) -> CudaView<f32> {
        self.inputs.slice(0..self.train_size * self.input_dims)
    }
    pub fn train_targets_noisy(&self) -> CudaView<f32> {
        self.targets_noisy
            .slice(0..self.train_size * self.output_dims)
    }
    pub fn train_targets_true_f(&self) -> CudaView<f32> {
        self.targets_true_f
            .slice(0..self.train_size * self.output_dims)
    }
    pub fn validation_inputs(&self) -> CudaView<f32> {
        self.inputs.slice(
            self.train_size * self.input_dims
                ..(self.train_size + self.validation_size) * self.input_dims,
        )
    }
    pub fn validation_targets_noisy(&self) -> CudaView<f32> {
        self.targets_noisy.slice(
            self.train_size * self.output_dims
                ..(self.train_size + self.validation_size) * self.output_dims,
        )
    }
    pub fn validation_targets_true_f(&self) -> CudaView<f32> {
        self.targets_true_f.slice(
            self.train_size * self.output_dims
                ..(self.train_size + self.validation_size) * self.output_dims,
        )
    }
    pub fn test_inputs(&self) -> CudaView<f32> {
        self.inputs.slice(
            (self.train_size + self.validation_size) * self.input_dims
                ..(self.train_size + self.validation_size + self.test_size) * self.input_dims,
        )
    }
    pub fn test_targets_noisy(&self) -> CudaView<f32> {
        self.targets_noisy.slice(
            (self.train_size + self.validation_size) * self.output_dims
                ..(self.train_size + self.validation_size + self.test_size) * self.output_dims,
        )
    }
    pub fn test_targets_true_f(&self) -> CudaView<f32> {
        self.targets_true_f.slice(
            (self.train_size + self.validation_size) * self.output_dims
                ..(self.train_size + self.validation_size + self.test_size) * self.output_dims,
        )
    }
}

pub struct Challenge {
    pub seed: [u8; 32],
    pub difficulty: Difficulty,
    pub hidden_layers_dims: usize,
    pub batch_size: usize,
    pub max_epochs: usize,
    pub patience: usize,
    pub min_loss_delta: f32,
    pub num_frozen_layers: usize,
    pub dataset: Dataset,
}

impl Challenge {
    pub fn generate_instance(
        seed: &[u8; 32],
        difficulty: &Difficulty,
        module: Arc<CudaModule>,
        stream: Arc<CudaStream>,
        _prop: &cudaDeviceProp,
    ) -> Result<Self> {
        const K_RFF: usize = 128;
        const RFF_AMPLITUDE_PER_FUNC: f32 = 1.0;
        const RFF_LENGTHSCALE_PER_INPUT_DIM: f32 = 0.3;
        const NOISE_STD: f32 = 0.2;
        const INPUT_DIMS: usize = 1;
        const OUTPUT_DIMS: usize = 2;
        const TRAIN_SIZE: usize = 1000;
        const VALIDATION_SIZE: usize = 200;
        const TEST_SIZE: usize = 250;
        let scaling_factor = RFF_AMPLITUDE_PER_FUNC * (2.0 / K_RFF as f32).sqrt();

        let d_seed = stream.memcpy_stod(seed)?;

        // Allocate memory for RFF params
        let mut a_params = stream.alloc_zeros::<f32>(OUTPUT_DIMS * K_RFF)?;
        let mut b_params = stream.alloc_zeros::<f32>(OUTPUT_DIMS * K_RFF)?;
        let mut w_params = stream.alloc_zeros::<f32>(OUTPUT_DIMS * K_RFF * INPUT_DIMS)?;

        // Generate RFF params
        let generate_rff_params_kernel = module.load_function("generate_rff_params")?;
        let n = (OUTPUT_DIMS * K_RFF) as u32;
        let grid_dim = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        let cfg = LaunchConfig {
            grid_dim: (grid_dim, 1, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&generate_rff_params_kernel)
                .arg(&d_seed)
                .arg(&(OUTPUT_DIMS as i32))
                .arg(&(INPUT_DIMS as i32))
                .arg(&(K_RFF as i32))
                .arg(&RFF_LENGTHSCALE_PER_INPUT_DIM)
                .arg(&mut a_params)
                .arg(&mut b_params)
                .arg(&mut w_params)
                .launch(cfg)?;
        }

        // Generate datasets
        let generate_dataset_kernel = module.load_function("generate_dataset")?;

        // Training data
        let total_samples = TRAIN_SIZE + VALIDATION_SIZE + TEST_SIZE;
        let mut inputs = stream.alloc_zeros::<f32>(total_samples * INPUT_DIMS)?;
        let mut targets_noisy = stream.alloc_zeros::<f32>(total_samples * OUTPUT_DIMS)?;
        let mut targets_true_f = stream.alloc_zeros::<f32>(total_samples * OUTPUT_DIMS)?;

        let grid_dim = (total_samples as u32 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        let cfg_train = LaunchConfig {
            grid_dim: (grid_dim, 1, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&generate_dataset_kernel)
                .arg(&d_seed)
                .arg(&(total_samples as i32))
                .arg(&(INPUT_DIMS as i32))
                .arg(&(OUTPUT_DIMS as i32))
                .arg(&K_RFF)
                .arg(&scaling_factor)
                .arg(&NOISE_STD)
                .arg(&a_params)
                .arg(&b_params)
                .arg(&w_params)
                .arg(&mut inputs)
                .arg(&mut targets_noisy)
                .arg(&mut targets_true_f)
                .launch(cfg_train)?;
        }

        stream.synchronize()?;

        Ok(Self {
            seed: *seed,
            difficulty: difficulty.clone(),
            hidden_layers_dims: 256,
            batch_size: 128,
            max_epochs: 1000,
            patience: 50,
            min_loss_delta: 1e-7,
            num_frozen_layers: 2,
            dataset: Dataset {
                inputs,
                targets_noisy,
                targets_true_f,
                train_size: TRAIN_SIZE,
                validation_size: VALIDATION_SIZE,
                test_size: TEST_SIZE,
                input_dims: INPUT_DIMS,
                output_dims: OUTPUT_DIMS,
            },
        })
    }

    conditional_pub!(
        fn verify_solution(
            &self,
            solution: &Solution,
            module: Arc<CudaModule>,
            stream: Arc<CudaStream>,
            _prop: &cudaDeviceProp,
        ) -> Result<()> {
            let cublas = CudaBlas::new(stream.clone())?;
            let cudnn = Cudnn::new(stream.clone())?;

            let mut model = MLP::new(&self.layer_dims(), self.num_frozen_layers, stream.clone())?;

            load_solution(&mut model, solution, stream.clone())?;

            let (output, _) = model.forward(
                &self.dataset.test_inputs(),
                false,
                stream.clone(),
                module.clone(),
                &cublas,
                &cudnn,
            )?;
            let (loss, _) = model.loss_and_grad(
                &output,
                &&self.dataset.test_targets_noisy(),
                stream.clone(),
                module.clone(),
            )?;

            let avg_model_loss_on_test = stream.memcpy_dtov(&loss)?[0];

            // Calculate baseline error epsilon_star_squared
            let alpha = 4.0 - self.difficulty.accuracy_factor as f32 / 1000.0;

            let y_h = stream.memcpy_dtov(&self.dataset.test_targets_noisy())?;
            let f_h = stream.memcpy_dtov(&self.dataset.test_targets_true_f())?;
            stream.synchronize()?;

            let sum_sq_diff_true_vs_noisy: f32 = y_h
                .iter()
                .zip(f_h.iter())
                .map(|(y, f)| (*y - *f).powi(2))
                .sum();

            let epsilon_star_squared =
                (alpha / self.dataset.test_size as f32) * sum_sq_diff_true_vs_noisy;

            if avg_model_loss_on_test <= epsilon_star_squared {
                Ok(())
            } else {
                Err(anyhow!(
                "Model test loss ({:.4e}) exceeds target baseline epsilon_star_squared ({:.4e})",
                avg_model_loss_on_test,
                epsilon_star_squared
            ))
            }
        }
    );

    pub fn layer_dims(&self) -> Vec<usize> {
        let mut layer_dims = vec![self.hidden_layers_dims; self.difficulty.num_hidden_layers];
        layer_dims.insert(0, self.dataset.input_dims);
        layer_dims.push(self.dataset.output_dims);
        layer_dims
    }
}

pub trait OptimizerStateTrait: Any + Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn box_clone(&self) -> Box<dyn OptimizerStateTrait>;
}

impl Clone for Box<dyn OptimizerStateTrait> {
    fn clone(&self) -> Self {
        self.box_clone()
    }
}

/// Function type for initializing optimizer state
pub type OptimizerInitStateFn = fn(
    seed: [u8; 32],
    param_sizes: &[usize], // Sizes of all parameter tensors
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    prop: &cudaDeviceProp,
) -> Result<Box<dyn OptimizerStateTrait>>;

/// Function type for querying optimizer at specific parameters (like parameter prediction)
pub type OptimizerQueryAtParamsFn = fn(
    optimizer_state: &dyn OptimizerStateTrait,
    model_params: &[CudaSlice<f32>], // FIXME pass in model map instead
    epoch: usize,
    train_loss: Option<f32>,
    val_loss: Option<f32>,
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    prop: &cudaDeviceProp,
) -> Result<Option<Vec<CudaSlice<f32>>>>;

/// Function type for optimizer step (computes parameter updates)
pub type OptimizerStepFn = fn(
    optimizer_state: &mut dyn OptimizerStateTrait,
    gradients: &[CudaSlice<f32>], // FIXME pass in model map instead
    epoch: usize,
    train_loss: Option<f32>,
    val_loss: Option<f32>,
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    prop: &cudaDeviceProp,
) -> Result<Vec<CudaSlice<f32>>>;

pub fn training_loop(
    challenge: &Challenge,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    prop: &cudaDeviceProp,
    optimizer_init_state: OptimizerInitStateFn,
    optimizer_query_at_params: OptimizerQueryAtParamsFn,
    optimizer_step: OptimizerStepFn,
) -> Result<(Solution, Vec<f32>, Vec<f32>)> {
    let Challenge {
        batch_size,
        max_epochs,
        min_loss_delta,
        patience,
        dataset:
            Dataset {
                train_size,
                validation_size,
                input_dims,
                output_dims,
                ..
            },
        ..
    } = *challenge;
    let cublas = CudaBlas::new(stream.clone())?;
    let cudnn = Cudnn::new(stream.clone())?;

    let mut model = MLP::new(
        &challenge.layer_dims(),
        challenge.num_frozen_layers,
        stream.clone(),
    )?;
    model.init_weights(challenge.seed.clone(), stream.clone(), module.clone())?;

    // Initialize optimizer
    let param_sizes = model.get_parameter_sizes();
    let mut optimizer_state = optimizer_init_state(
        challenge.seed.clone(),
        &param_sizes, // FIXME pass model instead?
        stream.clone(),
        module.clone(),
        prop,
    )?;

    let mut lowest_loss = f32::INFINITY;
    let mut _best_epoch = 0;
    let mut epochs_no_improvement = 0;
    let mut best_model_solution: Option<Solution> = None;
    let mut prev_train_loss = None;
    let mut prev_validation_loss = None;
    let mut train_losses = Vec::with_capacity(max_epochs);
    let mut validation_losses = Vec::with_capacity(max_epochs);

    let num_train_batches = (train_size + batch_size - 1) / batch_size;
    let num_val_batches = (validation_size + batch_size - 1) / batch_size;

    // Initialize RNG for shuffling
    let mut rng = StdRng::from_seed(challenge.seed);

    // Copy training data to host for shuffled batch creation
    let train_inputs = stream.memcpy_dtov(&challenge.dataset.train_inputs())?;
    let train_targets = stream.memcpy_dtov(&challenge.dataset.train_targets_noisy())?;
    let validation_inputs_d = challenge.dataset.validation_inputs();
    let validation_targets_d = challenge.dataset.validation_targets_noisy();
    stream.synchronize()?;

    for epoch in 0..max_epochs {
        // --- Shuffle training data indices each epoch ---
        let mut train_indices: Vec<usize> = (0..train_size).collect();
        train_indices.shuffle(&mut rng);

        // --- Training Phase ---
        let mut epoch_train_loss_sum = 0.0;
        for i in 0..num_train_batches {
            let batch_start_idx = i * batch_size;
            let current_batch_size = (train_size - batch_start_idx).min(batch_size);
            if current_batch_size == 0 {
                continue;
            }

            model.zero_grad(stream.clone(), module.clone())?;

            // Create shuffled batch data
            let mut input_batch_data = vec![0.0f32; current_batch_size * input_dims];
            let mut target_batch_data = vec![0.0f32; current_batch_size * output_dims];

            // Gather shuffled batch data
            for batch_offset in 0..current_batch_size {
                let shuffled_sample_idx = train_indices[batch_start_idx + batch_offset];

                // Copy input data for this sample
                let input_start = shuffled_sample_idx * input_dims;
                let batch_input_start = batch_offset * input_dims;
                for d in 0..input_dims {
                    input_batch_data[batch_input_start + d] = train_inputs[input_start + d];
                }

                // Copy target data for this sample
                let target_start = shuffled_sample_idx * output_dims;
                let batch_target_start = batch_offset * output_dims;
                for d in 0..output_dims {
                    target_batch_data[batch_target_start + d] = train_targets[target_start + d];
                }
            }

            // Upload shuffled batch to GPU
            let mut d_input_batch = stream.alloc_zeros::<f32>(current_batch_size * input_dims)?;
            let mut d_target_batch = stream.alloc_zeros::<f32>(current_batch_size * output_dims)?;
            stream.memcpy_htod(&input_batch_data, &mut d_input_batch)?;
            stream.memcpy_htod(&target_batch_data, &mut d_target_batch)?;

            // Query optimizer for parameter modifications before forward pass
            let model_params = model.extract_parameters(stream.clone())?;
            let original_params = if let Some(modified_params) = optimizer_query_at_params(
                optimizer_state.as_ref(),
                &model_params,
                epoch,
                prev_train_loss,
                prev_validation_loss,
                stream.clone(),
                module.clone(),
                prop,
            )? {
                let backup = model_params.clone();
                model.set_parameters(&modified_params, stream.clone(), module.clone())?;
                Some(backup)
            } else {
                None
            };

            let (output, caches) = model.forward(
                &d_input_batch,
                true,
                stream.clone(),
                module.clone(),
                &cublas,
                &cudnn,
            )?;
            let (loss, grad) = model.loss_and_grad(
                &output,
                &d_target_batch.as_view(),
                stream.clone(),
                module.clone(),
            )?;
            model.backward(
                &grad,
                &caches,
                false,
                stream.clone(),
                module.clone(),
                &cublas,
                &cudnn,
            )?;

            // Restore original parameters if they were modified
            if let Some(params_to_restore) = original_params {
                model.set_parameters(&params_to_restore, stream.clone(), module.clone())?;
            }

            // Get gradients and apply optimizer step
            let gradients = model.extract_gradients(stream.clone())?;
            let param_updates = optimizer_step(
                optimizer_state.as_mut(),
                &gradients,
                epoch,
                prev_train_loss,
                prev_validation_loss,
                stream.clone(),
                module.clone(),
                prop,
            )?;

            model.apply_optimizer_updates(&param_updates, stream.clone(), module.clone())?;

            let batch_loss = stream.memcpy_dtov(&loss)?[0];
            epoch_train_loss_sum += batch_loss * current_batch_size as f32;
        }
        stream.synchronize()?;

        let avg_train_loss = epoch_train_loss_sum / train_size as f32;
        prev_train_loss = Some(avg_train_loss);
        train_losses.push(avg_train_loss);

        // --- Validation Phase ---
        let mut epoch_val_loss_sum = 0.0;
        if validation_size > 0 {
            for i in 0..num_val_batches {
                let batch_start = i * batch_size;
                let current_batch_size = (validation_size - batch_start).min(batch_size);
                if current_batch_size == 0 {
                    continue;
                }

                let d_input_batch = validation_inputs_d.slice(
                    batch_start * input_dims..(batch_start + current_batch_size) * input_dims,
                );
                let d_target_batch = validation_targets_d.slice(
                    batch_start * output_dims..(batch_start + current_batch_size) * output_dims,
                );

                let (output, _) = model.forward(
                    &d_input_batch,
                    false,
                    stream.clone(),
                    module.clone(),
                    &cublas,
                    &cudnn,
                )?;
                let (loss, _) = model.loss_and_grad(
                    &output,
                    &d_target_batch,
                    stream.clone(),
                    module.clone(),
                )?;

                let batch_loss = stream.memcpy_dtov(&loss)?[0];
                epoch_val_loss_sum += batch_loss * current_batch_size as f32;
            }
        }
        stream.synchronize()?;

        let avg_val_loss = if validation_size > 0 {
            epoch_val_loss_sum / validation_size as f32
        } else {
            avg_train_loss
        };
        prev_validation_loss = Some(avg_val_loss);
        validation_losses.push(avg_val_loss);

        // --- Early Stopping ---
        if avg_val_loss < lowest_loss - min_loss_delta {
            lowest_loss = avg_val_loss;
            _best_epoch = epoch;
            best_model_solution = Some(to_solution(&model, epoch + 1, stream.clone())?);
            epochs_no_improvement = 0;
        } else {
            epochs_no_improvement += 1;
            if epochs_no_improvement >= patience {
                break;
            }
        }
    }

    stream.synchronize()?;

    let solution = best_model_solution.ok_or_else(|| anyhow!("No valid solution found during training. Validation loss may have been NaN or never improved."))?;
    Ok((solution, train_losses, validation_losses))
}

pub fn load_solution(mlp: &mut MLP, solution: &Solution, stream: Arc<CudaStream>) -> Result<()> {
    for (i, layer) in mlp.lin.iter_mut().enumerate() {
        let w_flat: Vec<f32> = solution.weights[i].iter().flatten().cloned().collect();
        stream.memcpy_htod(&w_flat, &mut layer.weight)?;

        stream.memcpy_htod(&solution.biases[i], &mut layer.bias)?;
    }
    for (i, bn) in mlp.bns.iter_mut().enumerate() {
        stream.memcpy_htod(&solution.bn_weights[i], &mut bn.weight)?;
        stream.memcpy_htod(&solution.bn_biases[i], &mut bn.bias)?;
        stream.memcpy_htod(&solution.bn_running_means[i], &mut bn.running_mean)?;
        stream.memcpy_htod(&solution.bn_running_vars[i], &mut bn.running_var)?;
    }
    stream.synchronize()?;
    Ok(())
}

pub fn to_solution(mlp: &MLP, epochs_used: usize, stream: Arc<CudaStream>) -> Result<Solution> {
    stream.synchronize()?;
    let mut weights = Vec::new();
    let mut biases = Vec::new();
    for layer in &mlp.lin {
        let w = stream.memcpy_dtov(&layer.weight)?;
        let b = stream.memcpy_dtov(&layer.bias)?;

        weights.push(w.chunks(layer.in_features).map(|c| c.to_vec()).collect());
        biases.push(b);
    }

    let mut bn_weights = Vec::new();
    let mut bn_biases = Vec::new();
    let mut bn_running_means = Vec::new();
    let mut bn_running_vars = Vec::new();

    for bn in &mlp.bns {
        bn_weights.push(stream.memcpy_dtov(&bn.weight)?);
        bn_biases.push(stream.memcpy_dtov(&bn.bias)?);
        bn_running_means.push(stream.memcpy_dtov(&bn.running_mean)?);
        bn_running_vars.push(stream.memcpy_dtov(&bn.running_var)?);
    }

    Ok(Solution {
        weights,
        biases,
        epochs_used,
        bn_weights,
        bn_biases,
        bn_running_means,
        bn_running_vars,
    })
}
