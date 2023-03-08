from tuneavideo.pipelines.pipeline_tuneavideo import TuneAVideoPipeline
from tuneavideo.models.unet import UNet3DConditionModel
from tuneavideo.util import save_videos_grid

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.utils import load_image
from diffusers import UniPCMultistepScheduler

import torch
import cv2
from PIL import Image
import numpy as np
from einops import rearrange
import decord
decord.bridge.set_bridge('torch')


from annotator.util import resize_image, HWC3
from annotator.openpose import OpenposeDetector
from annotator.canny import CannyDetector

apply_canny = CannyDetector()
pply_openpose = OpenposeDetector()


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid



def tvideo_inference():
    pretrained_model_path = "./checkpoints/models--CompVis--stable-diffusion-v1-4/snapshots/3857c45b7d4e78b3ba0f39d4d7f50a2a05aa23d4"
    my_model_path = "./outputs/man-skiing"
    unet = UNet3DConditionModel.from_pretrained(my_model_path, subfolder='unet', torch_dtype=torch.float16).to('cuda')

    tvideo_pipe = TuneAVideoPipeline.from_pretrained(pretrained_model_path, unet=unet, torch_dtype=torch.float16).to("cuda")
    tvideo_pipe.enable_xformers_memory_efficient_attention()
    tvideo_pipe.enable_vae_slicing()


    prompt = "spider man is skiing"
    ddim_inv_latent = torch.load(f"{my_model_path}/inv_latents/ddim_latent-500.pt").to(torch.float16)
    video = tvideo_pipe(prompt, latents=ddim_inv_latent, video_length=24, height=512, width=512, num_inference_steps=50, guidance_scale=12.5).videos

    save_videos_grid(video, f"./{prompt}.gif")


def controlnet_inference():

    image = load_image(
        "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
    )
    image = np.array(image)
    print("load image")

    low_threshold = 100
    high_threshold = 200

    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    print("get canny image")

    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16, cache_dir="./checkpoints")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16, cache_dir="./checkpoints"
    )
    print("download models")
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    pipe.enable_xformers_memory_efficient_attention()

    prompt = ", best quality, extremely detailed"
    prompt = [t + prompt for t in ["Sandra Oh", "Kim Kardashian", "rihanna", "taylor swift"]]
    generator = [torch.Generator(device="cpu").manual_seed(2) for i in range(len(prompt))]

    output = pipe(
        prompt,
        canny_image,
        negative_prompt=["monochrome, lowres, bad anatomy, worst quality, low quality"] * 4,
        num_inference_steps=20,
        generator=generator,
    )
    print("get results")

    image_grid(output.images, 2, 2)


def compose_inference(tvideo_ratio=0.5, savename=None):

    # load config (should be the same as corresponding config file
    config_file = "configs/man-skiing.yaml"
    pretrained_model_path = "./checkpoints/models--CompVis--stable-diffusion-v1-4/snapshots/3857c45b7d4e78b3ba0f39d4d7f50a2a05aa23d4"
    my_model_path = "./outputs/man-skiing"

    sample_start_idx = 0
    sample_frame_rate = 2
    n_sample_frames = 24
    height = 512
    width = 512
    video_path = "data/man-skiing.mp4"

    video_length = n_sample_frames
    num_inference_steps=50 
    guidance_scale=12.5

    control_type = "canny"   # [canny, openpose]

    # load tune-a-video pipeline
    unet = UNet3DConditionModel.from_pretrained(my_model_path, subfolder='unet', torch_dtype=torch.float16).to('cuda')

    tvideo_pipe = TuneAVideoPipeline.from_pretrained(pretrained_model_path, unet=unet, torch_dtype=torch.float16).to("cuda")
    tvideo_pipe.enable_xformers_memory_efficient_attention()
    tvideo_pipe.enable_vae_slicing()
    
    # load controlnet pipeline
    controlnet = ControlNetModel.from_pretrained(f"lllyasviel/sd-controlnet-{control_type}", torch_dtype=torch.float16, cache_dir="./checkpoints")
    controlnet_pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16, cache_dir="./checkpoints"
    )

    # TODO(demi): check whether we can use UniPCMultistep Scheduler for Tune-A-Video too
    # controlnet_pipe.scheduler = UniPCMultistepScheduler.from_config(controlnet_pipe.scheduler.config)
    controlnet_pipe.enable_model_cpu_offload()
    controlnet_pipe.enable_xformers_memory_efficient_attention()

    
    # get controlnet conditional images

    ## load frames from original training video
    video = get_video(video_path, height, width, sample_start_idx, sample_frame_rate, n_sample_frames)
    control_images = get_control_images(video, height, width, n_sample_frames, control_type)  # [f, c, h, w]
    assert control_images.shape == (n_sample_frames, 3, height, width)


    # set parameters 
    #prompt = "spider man is skiing" #+ ", best quality, extremely detailed"
    prompt = "a man is skiing, acrylic painting, trending on pixiv fanbox, palette knife and brush strokes, style of makoto shinkai jamie wyeth james gilleard edward hopper greg rutkowski studio ghibli genshin impact, best quality, extremely detailed"
    ddim_inv_latent = torch.load(f"{my_model_path}/inv_latents/ddim_latent-500.pt").to(torch.float16)
    #print("!!!!!!! ddim_inv_latent.shape=",ddim_inv_latent.shape)
    assert ddim_inv_latent.shape[2] == 24
    ddim_inv_latent = ddim_inv_latent[:, :, :n_sample_frames]

    with torch.no_grad():
        video = _call_compose_inference(tvideo_pipe, controlnet_pipe, tvideo_ratio, prompt, control_images, latents=ddim_inv_latent, video_length=video_length, height=height, width=width, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)

   # video = tvideo_pipe(prompt, latents=ddim_inv_latent, video_length=24, height=512, width=512, num_inference_steps=50, guidance_scale=12.5).videos

    if savename is None:
        savename = f"./{prompt}.gif"
    save_videos_grid(video, f"{savename}")

def get_video(video_path, height, width, sample_start_idx, sample_frame_rate, n_sample_frames):
    vr = decord.VideoReader(video_path, width=width, height=height)
    sample_index = list(range(sample_start_idx, len(vr), sample_frame_rate))[:n_sample_frames]
    #print("video sample index=", sample_index)
    video = vr.get_batch(sample_index)
    #print("video[0]=", video[0])
    return video


def get_control_images(video, height, width, n_sample_frames, control_type, detect_resolution=512):
    assert video.shape == (n_sample_frames, height, width, 3)
    image_resolution = min(height, width)  # HACK(demi): need to rethink about this

    if control_type == "canny":
        detect_resolution = image_resolution
        low_threshold = 100
        high_threshold = 200

    control_images = []
    for i in range(n_sample_frames):
        input_image = video[i].cpu().numpy()
        input_image = HWC3(input_image)

        if control_type == "openpose":
            detected_map, _ = apply_openpose(resize_image(input_image, detect_resolution))
        elif control_type == "canny":
            detected_map = apply_canny(resize_image(input_image, detect_resolution), low_threshold, high_threshold)
        else:
            raise NotImplementedError
        detected_map = HWC3(detected_map)  # TODO(demi): check if pose is correct
        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape
        assert H == height and W == width  # HACK(demi)

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)
        # DEBUG(demi): store conditional images
        Image.fromarray(detected_map).save(f"{control_type}_cond_image_{i}.jpg")

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control_images.append(control)
    control_images = torch.stack(control_images, dim=0)
    assert control_images.shape == (n_sample_frames, height, width, 3)
    control_images = rearrange(control_images, "f h w c -> f c h w").clone()
    return control_images


def _call_compose_inference(tvideo_pipe, controlnet_pipe, tvideo_ratio, prompt, control_images, latents=None, video_length=1, height=512, width=512, num_inference_steps=50, guidance_scale=7.5, callback_steps=1,
    negative_prompt=None, eta=0.0, generator=None,
    callback=None, output_type="tensor",
    num_images_per_prompt=1, num_videos_per_prompt=1,
    controlnet_conditioning_scale=1.0, cross_attention_kwargs=None):
    # NB(demi): skip height, width

    # Check inputs. Raise error if not correct
    tvideo_pipe.check_inputs(prompt, height, width, callback_steps)

    # Define call parameters
    batch_size = 1 if isinstance(prompt, str) else len(prompt)
    device = tvideo_pipe._execution_device

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    do_classifier_free_guidance = guidance_scale > 1.0

    # Encode input prompt
    # TODO(demi): check if tvideo and controlnet text embeddings are the same
    tvideo_text_embeddings = tvideo_pipe._encode_prompt(
        prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt
    )

    # NB(demi): size should assume video_length is batch size
    controlnet_prompt_embeds = controlnet_pipe._encode_prompt(
        [prompt] * video_length,  # TODO(demi): make sure this is correct
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    )
    
    
    # Get controlnet conditioning images
    control_images = torch.stack([control_images for _ in range(batch_size)], dim=0)
    #print("control_images.shape=", control_images.shape)
    assert control_images.shape == (batch_size, video_length, 3, height, width)
    control_images = rearrange(control_images, 'b f c h w -> (b f) c h w')  # assume new batch size = batch size * video length

    cond_images = controlnet_pipe.prepare_image(
        control_images,
        width, height,
        batch_size * num_images_per_prompt,
        num_images_per_prompt,
        device,
        controlnet_pipe.controlnet.dtype,
    )
    if do_classifier_free_guidance:
        cond_images = torch.cat([cond_images] * 2)

    # Prepare timesteps
    scheduler = tvideo_pipe.scheduler
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps

    # Prepare latent variables
    num_channels_latents = tvideo_pipe.unet.in_channels
    assert num_channels_latents == controlnet_pipe.unet.in_channels, "controlnet and tune-a-video has different latent channel size"
    # TODO(demi): check contorlnet num_channel_latents, and latents size
    # TODO(demi): now, default using tune-a-video latent processing

    #print("video_length=", video_length, "latents.shape=", latents.shape)
    latents = tvideo_pipe.prepare_latents(
        batch_size * num_videos_per_prompt,
        num_channels_latents,
        video_length,
        height,
        width,
        tvideo_text_embeddings.dtype,
        device,
        generator,
        latents,
    )
    latents_dtype = latents.dtype

    # Prepare extra step kwargs.
    extra_step_kwargs = tvideo_pipe.prepare_extra_step_kwargs(generator, eta)

    # Denoising loop
    multiplier = 2 if do_classifier_free_guidance else 1
    num_warmup_steps = len(timesteps) - num_inference_steps * scheduler.order
    with tvideo_pipe.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual for tvideo
            tvideo_noise_pred = tvideo_pipe.unet(latent_model_input, t, encoder_hidden_states=tvideo_text_embeddings).sample.to(dtype=latents_dtype)

            # predict the noise residual for controlnet
            # TODO(demi): check relative size between image and latents, now reshape cond_images correspondingly
            # assuming batch size = 
            image = cond_images
            assert  image.shape == (multiplier * batch_size * video_length, 3, height, width)
            c_latents = rearrange(latents, 'b c f h w -> (b f) c h w')
            latent_model_input = torch.cat([c_latents] * 2) if do_classifier_free_guidance else c_latents
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)
            assert latent_model_input.shape[:2] == (batch_size * video_length * multiplier, 4)
            
            
            #print("pass in to controlnet, latent_model_input.shape=", latent_model_input.shape, "prompt_embeds=", controlnet_prompt_embeds.shape,
            #    "image.shpae=", image.shape)

            down_block_res_samples, mid_block_res_sample = controlnet_pipe.controlnet(
                latent_model_input,
                t,
                encoder_hidden_states=controlnet_prompt_embeds,
                controlnet_cond=image,
                return_dict=False,
            )

            # Q(demi): what is controlnet_conditional scale
            down_block_res_samples = [
                down_block_res_sample * controlnet_conditioning_scale
                for down_block_res_sample in down_block_res_samples
            ]
            mid_block_res_sample *= controlnet_conditioning_scale

            # predict the noise residual
            controlnet_noise_pred = controlnet_pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=controlnet_prompt_embeds,
                cross_attention_kwargs=cross_attention_kwargs,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            ).sample

            # TODO(demi): double check if the re-arrangement makes sense
            lc, lh, lw = controlnet_noise_pred.shape[-3:]
            controlnet_noise_pred = controlnet_noise_pred.reshape(multiplier, batch_size, video_length, lc, lh, lw)
            controlnet_noise_pred = rearrange(controlnet_noise_pred, 'm b f c h w -> (m b) c f h w')
            assert tvideo_noise_pred.shape == controlnet_noise_pred.shape
            
            noise_pred = tvideo_noise_pred * tvideo_ratio + controlnet_noise_pred * (1-tvideo_ratio)

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)


            # compute the previous noisy sample x_t -> x_t-1
            #print("7777777777777777")
            #print("noise_pred.shape=", noise_pred.shape, " latents.shape=", latents.shape, " c_latents.shape=", c_latents.shape)
            latents = scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % scheduler.order == 0):
                # Q(demi): how does it update the latents vector?
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)

            #embed()

    # Post-processing
    video = tvideo_pipe.decode_latents(latents)

    # Convert to tensor
    if output_type == "tensor":
        video = torch.from_numpy(video)

    return video


if __name__ == "__main__":
    #tvideo_inference() 
    #controlnet_inference()
    compose_inference(tvideo_ratio=1.0, savename="./tvideo_debug.gif")
    compose_inference(tvideo_ratio=0.5, savename="./compose_debug.gif")

