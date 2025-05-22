import torch
from PIL import Image
import sys
import os
import copy
import gc

# Get the current working directory
cwd = os.getcwd()
# Add the current working directory to the system path for module loading
sys.path.append(cwd)

from tqdm import tqdm  # Progress bar for loops
import random
from collections import defaultdict  # Dictionary subclass for creating dictionaries with default values
import prompts as prompts_file  # Custom module for handling prompts
import torch.distributed as dist  # PyTorch distributed computing
import numpy as np
import torch.utils.checkpoint as checkpoint  # Checkpointing to save memory
import wandb  # Weights and Biases for experiment tracking

wandb.require("core")  # Require core functionalities of wandb
import contextlib  # Utilities for context management
import torchvision  # PyTorch vision library
from transformers import AutoProcessor, AutoModel  # Transformers for processing and models
import sys
from diffusers.models.attention_processor import LoRAAttnProcessor  # Low-Rank Adaptation attention processor
from diffusers.loaders import AttnProcsLayers  # Attention processor layers loader
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel  # Stable Diffusion components
import datetime  # For handling date and time

from accelerate.logging import get_logger  # Logging utility from Accelerate
from accelerate import Accelerator  # Multi-GPU/TPU/CPU training utility
from absl import app, flags  # Abseil for command-line flags and application management
from ml_collections import config_flags  # For handling configuration files
import time  # Time utilities

from diffusers_patch.ddim_with_kl import ddim_step_KL  # Customized DDIM step with KL divergence
from online.model_utils import (generate_embeds_fn, evaluate_loss_fn, evaluate,
                                prepare_pipeline, generate_new_x_mid_observation,
                                online_aesthetic_loss_fn)
from online.dataset_pure import D_explored  # Online dataset for exploration

# Define flags for the script
FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/online.py:aesthetic", "Training configuration.")
flags.DEFINE_integer("seed", 42, "Random seed for reproducibility.")  # Added seed flag
flags.DEFINE_integer("num_outer_loop", 4, "Number of outer loops")
from accelerate.utils import set_seed, ProjectConfiguration  # Utility functions from Accelerate

# Get a logger for the script
logger = get_logger(__name__)


def main(_):
    # Start of the main function
    print('=======================')
    print(torch.cuda.is_available())  # Check if CUDA is available
    config = FLAGS.config  # Load configuration from flags
    unique_id = datetime.datetime.now().strftime(
        "%Y.%m.%d_%H.%M.%S")  # Generate a unique ID based on the current date and time
    config.seed = FLAGS.seed
    config.train.num_outer_loop = FLAGS.num_outer_loop
    if config.train.num_outer_loop==2:
        config.num_samples = [6400, 12800] #19200
    elif config.train.num_outer_loop==4:
        config.num_samples = [1280, 2560, 5120, 10240] #19200
    elif config.train.num_outer_loop==8:
        config.num_samples = [640, 864, 1166, 1575, 2126, 2870, 3874, 6085] #19200, \eta_base = 1.35

    # number of observations of PURE_SEIKO is no more than number of observation from SEIKO
    config.num_samples = [int(i//config.num_mid_observation//16*16) for i in config.num_samples]

    if not config.run_name:
        config.run_name = unique_id  # Use unique ID as run name if none is provided
    else:
        config.run_name += "_" + unique_id + f"_{config.train.optimism}_lamda={config.lamda} seed={config.seed} num_mid_observation={config.num_mid_observation} num_outer_loop={config.train.num_outer_loop}"  # Append unique ID to provided run name

    # Set up project configuration for logging and checkpoints
    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    # Initialize Accelerator for multi-device training
    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
    )

    # Explicitly initialize CUDA within the main function if available
    if torch.cuda.is_available():
        torch.cuda.init()
    print(torch.cuda.device_count())  # Print the number of CUDA devices
    print('***********************')

    # Initialize Weights & Biases (wandb) if on the main process
    if accelerator.is_main_process:
        wandb_args = {}
        wandb_args["name"] = config.run_name
        if config.debug:
            wandb_args.update({'mode': "disabled"})  # Disable wandb in debug mode
        accelerator.init_trackers(
            project_name="pure seiko", config=config.to_dict(), init_kwargs={"wandb": wandb_args}
        )

        # Set project and logging directories for wandb
        accelerator.project_configuration.project_dir = os.path.join(config.logdir, config.run_name)
        accelerator.project_configuration.logging_dir = os.path.join(config.logdir, wandb.run.name)

    logger.info(f"\n{config}")  # Log the configuration

    # Set random seed for reproducibility, with device-specific variations
    set_seed(config.seed, device_specific=True)
    np.random.seed(config.seed)
    random.seed(config.seed)
    
    # Load the Stable Diffusion pipeline, either from a file or from pretrained weights
    if config.pretrained.model.endswith(".safetensors") or config.pretrained.model.endswith(".ckpt"):
        pipeline = StableDiffusionPipeline.from_single_file(config.pretrained.model)
    else:
        pipeline = StableDiffusionPipeline.from_pretrained(config.pretrained.model, revision=config.pretrained.revision)

    # Freeze parameters of the models to save memory
    inference_dtype = torch.float32  # Set the data type for inference

    # Prepare the pipeline for training, including unet models
    unet_list, Unet2d_models = prepare_pipeline(pipeline, accelerator, config, inference_dtype, num_outer_loop=config.train.num_outer_loop)

    # Generate embedding function based on the device and data type
    embedding_fn = generate_embeds_fn(device=accelerator.device, torch_dtype=inference_dtype)

    # Define the online loss function for training
    online_loss_fn = online_aesthetic_loss_fn(grad_scale=config.grad_scale,
                                              aesthetic_target=config.aesthetic_target,
                                              config=config,
                                              accelerator=accelerator,
                                              torch_dtype=inference_dtype,
                                              device=accelerator.device)

    # Define the evaluation loss function
    eval_loss_fn = evaluate_loss_fn(grad_scale=config.grad_scale,
                                    aesthetic_target=config.aesthetic_target,
                                    accelerator=accelerator,
                                    torch_dtype=inference_dtype,
                                    device=accelerator.device)

    # Enable TF32 for faster computation if allowed
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Load prompt functions based on the configuration
    prompt_fn = getattr(prompts_file, config.prompt_fn)
    samping_prompt_fn = getattr(prompts_file, config.samping_prompt_fn)

    # Use the evaluation prompt function if specified, otherwise use the regular prompt function
    if config.eval_prompt_fn == '':
        eval_prompt_fn = prompt_fn
    else:
        eval_prompt_fn = getattr(prompts_file, config.eval_prompt_fn)

    # Generate negative prompt embeddings for training and sampling
    neg_prompt_embed = pipeline.text_encoder(
        pipeline.tokenizer(
            [""],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(accelerator.device)
    )[0]

    # Repeat the negative prompt embeddings to match the batch size
    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size_per_gpu_available, 1, 1)
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.batch_size_per_gpu_available, 1, 1)

    autocast = contextlib.nullcontext  # No automatic casting by default

    #################### TRAINING ####################

    # Set the number of fresh samples to generate
    num_fresh_samples = config.num_samples  # 64 samples take 4 minutes to generate
    assert len(
        num_fresh_samples) == config.train.num_outer_loop, "Number of outer loops must match the number of data counts"

    # Initialize the exploration dataset
    exp_dataset = D_explored(config, accelerator.device).to(accelerator.device, dtype=inference_dtype)
    exp_dataset.model = accelerator.prepare(exp_dataset.model)

    global_step = 0  # Initialize the global step counter
    for outer_loop in range(config.train.num_outer_loop):
        # Generate new samples and update the model
        current_unet = unet_list[outer_loop]  # Select the current UNet model
        training_unet = unet_list[outer_loop + 1]  # Select the next UNet model for training
        num_new_x = num_fresh_samples[outer_loop]  # Number of new samples to generate
        print(num_new_x)

        current_unet.eval()  # Set the current UNet model to evaluation mode

        # Freeze parameters of the current UNet model
        if outer_loop == 0:
            for param in current_unet.parameters():
                param.requires_grad = False
        else:
            for name, attn_processor in current_unet.named_children():
                for param in attn_processor.parameters():
                    param.requires_grad = False
        logger.info(f"Freezing current model: {outer_loop}")
        logger.info(f"Start training model: {outer_loop + 1}")

        # Load the weights from the current UNet model into the training UNet model if not the first loop
        if outer_loop > 0:
            logger.info(f"Load previous model: {outer_loop} weight to training model: {outer_loop + 1}")
            training_unet.load_state_dict(current_unet.state_dict())

        # Ensure all LoRA parameters are trainable
        for name, attn_processor in training_unet.named_children():
            for param in attn_processor.parameters():
                assert param.requires_grad == True, "All LoRA parameters should be trainable"

        all_new_x = generate_new_x_mid_observation(
            current_unet,
            num_new_x // config.train.num_gpus,
            pipeline,
            accelerator,
            config,
            inference_dtype,
            samping_prompt_fn,
            sample_neg_prompt_embeds,
            embedding_fn,
            config.num_mid_observation)

        # Before sample generation
        print(f"Expected number of samples: {num_new_x}")

        # After sample generation
        print(f"Actual number of samples generated: {all_new_x.shape[1]}")

        # Update the exploration dataset with the new samples
        exp_dataset.update(all_new_x,lamda=config.lamda)
        del all_new_x  # Delete the new samples to free memory

        # Train the model using a pessimistic reward model or bootstrap approach
        if config.train.optimism in ['none', 'UCB']:
            exp_dataset.train_MLP(accelerator, config)
        elif config.train.optimism == 'bootstrap':
            exp_dataset.train_bootstrap(accelerator, config)
        else:
            raise ValueError(f"Unknown optimism {config.train.optimism}")

        # Sanity check model weight synchronization in distributed training
        if accelerator.num_processes > 1:
            if config.train.optimism == 'bootstrap':
                print(
                    f"Process {accelerator.process_index} model 0 layer 0 bias: {exp_dataset.model.module.models[0].layers[0].bias.data}")
            else:
                print(
                    f"Process {accelerator.process_index} layer 0 bias: {exp_dataset.model.module.layers[0].bias.data}")
            print(f"Process {accelerator.process_index} x: {exp_dataset.x.shape}")

        # Clear CUDA cache and collect garbage to free up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # Update the diffusion model by fine-tuning
        optimizer = torch.optim.AdamW(
            training_unet.parameters(),
            lr=config.train.learning_rate,
            betas=(config.train.adam_beta1, config.train.adam_beta2),
            weight_decay=config.train.adam_weight_decay,
            eps=config.train.adam_epsilon,
        )

        # Prepare the model and optimizer with the accelerator
        training_unet, optimizer = accelerator.prepare(training_unet, optimizer)

        # Get the timesteps from the scheduler
        timesteps = pipeline.scheduler.timesteps  # [981, 961, 941, 921,]

        # Prepare prompts for evaluation
        eval_prompts, eval_prompt_metadata = zip(
            *[eval_prompt_fn() for _ in range(config.train.batch_size_per_gpu_available * config.max_vis_images)]
        )

        # Training loop for each epoch
        for epoch in list(range(0, config.num_epochs)):
            training_unet.train()  # Set the training UNet model to training mode
            info = defaultdict(list)  # Dictionary to store training information
            info_vis = defaultdict(list)  # Dictionary to store visual information
            image_vis_list = []

            # Inner loop for each data loader iteration
            for inner_iters in tqdm(
                    list(range(config.train.data_loader_iterations)),
                    position=0,
                    disable=not accelerator.is_local_main_process
            ):
                # Initialize random latent variables
                latent = torch.randn((config.train.batch_size_per_gpu_available, 4, 64, 64),
                                     device=accelerator.device, dtype=inference_dtype)

                # Log the training progress
                if accelerator.is_main_process:
                    logger.info(
                        f"{config.run_name.rsplit('/', 1)[0]} Loop={outer_loop}/Epoch={epoch}/Iter={inner_iters}: training")

                # Generate prompts for the current iteration
                prompts, prompt_metadata = zip(
                    *[prompt_fn() for _ in range(config.train.batch_size_per_gpu_available)]
                )

                # Tokenize the prompts and move them to the device
                prompt_ids = pipeline.tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=pipeline.tokenizer.model_max_length,
                ).input_ids.to(accelerator.device)

                # Move scheduler alpha values to the device
                pipeline.scheduler.alphas_cumprod = pipeline.scheduler.alphas_cumprod.to(accelerator.device)
                prompt_embeds = pipeline.text_encoder(prompt_ids)[0]  # Encode the prompts into embeddings

                with accelerator.accumulate(training_unet):  # Accumulate gradients across iterations
                    with autocast():  # No automatic mixed precision by default
                        with torch.enable_grad():  # Ensure gradients are enabled
                            keep_input = True

                            kl_loss = 0  # Initialize KL loss

                            # Loop over the timesteps for DDIM sampling
                            for i, t in tqdm(
                                    enumerate(timesteps),
                                    total=len(timesteps),
                                    disable=not accelerator.is_local_main_process,
                            ):
                                t = torch.tensor([t],
                                                 dtype=inference_dtype,
                                                 device=latent.device
                                                 )
                                t = t.repeat(config.train.batch_size_per_gpu_available)  # Repeat timestep across batch

                                # Use gradient checkpointing if enabled for memory saving
                                if config.grad_checkpoint:
                                    noise_pred_uncond = checkpoint.checkpoint(training_unet, latent, t,
                                                                              train_neg_prompt_embeds,
                                                                              use_reentrant=False).sample
                                    noise_pred_cond = checkpoint.checkpoint(training_unet, latent, t, prompt_embeds,
                                                                            use_reentrant=False).sample

                                    old_noise_pred_uncond = checkpoint.checkpoint(current_unet, latent, t,
                                                                                  train_neg_prompt_embeds,
                                                                                  use_reentrant=False).sample
                                    old_noise_pred_cond = checkpoint.checkpoint(current_unet, latent, t, prompt_embeds,
                                                                                use_reentrant=False).sample

                                else:
                                    # Predict noise for unconditional and conditional prompts without checkpointing
                                    noise_pred_uncond = training_unet(latent, t, train_neg_prompt_embeds).sample
                                    noise_pred_cond = training_unet(latent, t, prompt_embeds).sample

                                    old_noise_pred_uncond = current_unet(latent, t, train_neg_prompt_embeds).sample
                                    old_noise_pred_cond = current_unet(latent, t, prompt_embeds).sample

                                # Truncated backpropagation through time for memory efficiency
                                if config.truncated_backprop:
                                    if config.truncated_backprop_rand:
                                        timestep = random.randint(
                                            config.truncated_backprop_minmax[0],
                                            config.truncated_backprop_minmax[1]
                                        )
                                        if i < timestep:
                                            noise_pred_uncond = noise_pred_uncond.detach()
                                            noise_pred_cond = noise_pred_cond.detach()
                                            old_noise_pred_uncond = old_noise_pred_uncond.detach()
                                            old_noise_pred_cond = old_noise_pred_cond.detach()
                                    else:
                                        if i < config.trunc_backprop_timestep:
                                            noise_pred_uncond = noise_pred_uncond.detach()
                                            noise_pred_cond = noise_pred_cond.detach()
                                            old_noise_pred_uncond = old_noise_pred_uncond.detach()
                                            old_noise_pred_cond = old_noise_pred_cond.detach()

                                # Calculate gradients (difference between conditional and unconditional noise predictions)
                                grad = (noise_pred_cond - noise_pred_uncond)
                                old_grad = (old_noise_pred_cond - old_noise_pred_uncond)

                                # Adjust noise predictions using guidance scale
                                noise_pred = noise_pred_uncond + config.sd_guidance_scale * grad
                                old_noise_pred = old_noise_pred_uncond + config.sd_guidance_scale * old_grad

                                # Perform a DDIM step with KL divergence
                                latent, kl_terms = ddim_step_KL(
                                    pipeline.scheduler,
                                    noise_pred,  # (2,4,64,64),
                                    old_noise_pred,  # (2,4,64,64),
                                    t[0].long(),
                                    latent,
                                    eta=config.sample_eta,  # 1.0
                                )
                                kl_loss += torch.mean(kl_terms).to(inference_dtype)  # Accumulate KL loss

                            # Decode the latent variables into images
                            ims = pipeline.vae.decode(
                                latent.to(pipeline.vae.dtype) / 0.18215).sample  # latent entries around -5 - +7

                            # Calculate the loss using the online loss function
                            loss, rewards = online_loss_fn(ims, config, exp_dataset)
                            loss = loss.mean() * config.train.loss_coeff  # Scale the loss

                            total_loss = loss + config.train.kl_weight * kl_loss  # Combine loss and KL loss

                            rewards_mean = rewards.mean()  # Mean of rewards
                            rewards_std = rewards.std()  # Standard deviation of rewards

                            # Store images and rewards for visualization
                            if len(info_vis["image"]) < config.max_vis_images:
                                info_vis["image"].append(ims.clone().detach())
                                info_vis["rewards_img"].append(rewards.clone().detach())
                                info_vis["prompts"] = list(info_vis["prompts"]) + list(prompts)

                            info["loss"].append(total_loss)  # Record total loss
                            info["KL-entropy"].append(kl_loss)  # Record KL loss

                            info["rewards"].append(rewards_mean)  # Record mean rewards
                            info["rewards_std"].append(rewards_std)  # Record rewards standard deviation

                            # Backward pass to compute gradients
                            accelerator.backward(total_loss)
                            if accelerator.sync_gradients:
                                accelerator.clip_grad_norm_(training_unet.parameters(), config.train.max_grad_norm)
                            optimizer.step()  # Update model parameters
                            optimizer.zero_grad()  # Reset gradients

                            # Check if the accelerator has performed an optimization step
                if accelerator.sync_gradients:
                    assert (inner_iters + 1) % config.train.gradient_accumulation_steps == 0
                    # Log training and evaluation
                    if config.visualize_eval and (global_step % config.vis_freq == 0):

                        all_eval_images = []
                        all_eval_rewards = []
                        if config.same_evaluation:
                            generator = torch.cuda.manual_seed(config.seed)
                            latent = torch.randn(
                                (config.train.batch_size_per_gpu_available * config.max_vis_images, 4, 64, 64),
                                device=accelerator.device, dtype=inference_dtype, generator=generator)
                        else:
                            latent = torch.randn(
                                (config.train.batch_size_per_gpu_available * config.max_vis_images, 4, 64, 64),
                                device=accelerator.device, dtype=inference_dtype)
                        with torch.no_grad():  # Disable gradients for evaluation
                            for index in range(config.max_vis_images):
                                ims, rewards = evaluate(
                                    training_unet,
                                    latent[
                                    config.train.batch_size_per_gpu_available * index:config.train.batch_size_per_gpu_available * (
                                                index + 1)],
                                    train_neg_prompt_embeds,
                                    eval_prompts[
                                    config.train.batch_size_per_gpu_available * index:config.train.batch_size_per_gpu_available * (
                                                index + 1)],
                                    pipeline,
                                    accelerator,
                                    inference_dtype,
                                    config,
                                    eval_loss_fn)

                                all_eval_images.append(ims)
                                all_eval_rewards.append(rewards)

                        eval_rewards = torch.cat(all_eval_rewards)  # Concatenate evaluation rewards
                        eval_reward_mean = eval_rewards.mean()  # Calculate mean evaluation reward
                        eval_reward_std = eval_rewards.std()  # Calculate standard deviation of evaluation rewards
                        eval_images = torch.cat(all_eval_images)  # Concatenate evaluation images
                        eval_image_vis = []
                        if accelerator.is_main_process:
                            # Save evaluation images
                            name_val = config.run_name
                            log_dir = f"logs/{name_val}/eval_vis"
                            os.makedirs(log_dir, exist_ok=True)
                            for i, eval_image in enumerate(eval_images):
                                eval_image = (eval_image.clone().detach() / 2 + 0.5).clamp(0, 1)
                                pil = Image.fromarray(
                                    (eval_image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                                prompt = eval_prompts[i]
                                pil.save(
                                    f"{log_dir}/{outer_loop:01d}_{epoch:03d}_{inner_iters:03d}_{i:03d}_{prompt}.png")
                                pil = pil.resize((256, 256))  # Resize image for logging
                                reward = eval_rewards[i]
                                # eval_image_vis.append(wandb.Image(pil, caption=f"{prompt:.25} | {reward:.2f}"))
                            # accelerator.log({"eval_images": eval_image_vis}, step=global_step)

                    logger.info("Logging")  # Log training progress

                    # Calculate mean of recorded metrics
                    info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                    info = accelerator.reduce(info, reduction="mean")
                    logger.info(f"loss: {info['loss']}, rewards: {info['rewards']}")

                    # Update info dictionary with additional metrics
                    info.update({"outer_loop": outer_loop,
                                 "epoch": epoch,
                                 "inner_epoch": inner_iters,
                                 "eval_rewards": eval_reward_mean,
                                 "eval_rewards_std": eval_reward_std,
                                 "dataset_size": len(exp_dataset),
                                 "dataset_y_avg": torch.mean(exp_dataset.y),
                                 })
                    accelerator.log(info, step=global_step)  # Log the updated info

                    if config.visualize_train:
                        ims = torch.cat(info_vis["image"])  # Concatenate training images
                        rewards = torch.cat(info_vis["rewards_img"])  # Concatenate training rewards
                        prompts = info_vis["prompts"]
                        images = []
                        for i, image in enumerate(ims):
                            image = (image.clone().detach() / 2 + 0.5).clamp(0, 1)
                            pil = Image.fromarray((image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                            pil = pil.resize((256, 256))
                            prompt = prompts[i]
                            reward = rewards[i]
                            # images.append(wandb.Image(pil, caption=f"{prompt:.25} | {reward:.2f}"))

                        # accelerator.log(
                        #     {"images": images},
                        #     step=global_step,
                        # )

                    global_step += 1  # Increment the global step
                    info = defaultdict(list)  # Reset the info dictionary

            # Ensure that we performed an optimization step at the end of the inner epoch
            assert accelerator.sync_gradients

            # Save the model and training state at regular intervals
            if epoch % config.save_freq == 0 and accelerator.is_main_process:
                def save_model_hook(models, weights, output_dir):
                    if isinstance(models[-1], AttnProcsLayers):
                        Unet2d_models[outer_loop + 1].save_attn_procs(output_dir)
                    else:
                        raise ValueError(f"Unknown model type {type(models[-1])}")
                    for _ in range(len(weights)):
                        weights.pop()

                accelerator.register_save_state_pre_hook(save_model_hook)
                accelerator.save_state()

        del optimizer  # Delete the optimizer to free memory

        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear CUDA cache
        gc.collect()  # Run garbage collection


# Entry point for the script
if __name__ == "__main__":
    print("TEST")
    print(torch.cuda.device_count())  # Print the number of CUDA devices
    # os.environ["CUDA_VISIBLE_DEVICES"] = '2'  # Set the visible CUDA device to GPU 2
    app.run(main)  # Run the main function using Abseil's app
