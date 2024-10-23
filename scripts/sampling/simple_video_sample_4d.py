import os
import io
import sys
import cv2
import yaml
import time
import imageio
import argparse
from PIL import Image
from glob import glob
from tqdm import tqdm
from typing import List, Optional, Union

sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "../../")))
import numpy as np
from fire import Fire

import torch
import torchvision
import torch.nn.functional
import torchvision.transforms as T
from torchvision.utils import save_image

from sgm.modules.encoders.modules import VideoPredictionEmbedderWithEncoder
from scripts.demo.sv4d_helpers import (
    decode_latents,
    load_model,
    initial_model_load,
    read_video,
    run_img2vid,
    prepare_sampling,
    prepare_inputs,
    do_sample_per_step,
    sample_sv3d,
    save_video,
    preprocess_video,
    preprocess_edit_image
)

import warnings
warnings.filterwarnings('ignore')

# Used fo image editing 
from scepter.modules.inference.stylebooth_inference import StyleboothInference
from scepter.modules.utils.config import Config
from scepter.modules.utils.file_system import FS
from scepter.modules.utils.logger import get_logger

# Used for video editing
from FRESCO_src.utils import *
from FRESCO_src.keyframe_selection import get_keyframe_ind
from FRESCO_src.diffusion_hacked import apply_FRESCO_attn, apply_FRESCO_opt, disable_FRESCO_opt
from FRESCO_src.diffusion_hacked import get_flow_and_interframe_paras, get_intraframe_paras
from FRESCO_src.pipe_FRESCO import inference
import diffusers
from diffusers import StableDiffusionPipeline, AutoencoderKL, DDPMScheduler, ControlNetModel



# Initialize first frame editing environment
def setup_image_edit_environment():
    logger = get_logger(name='scepter')
    config_file = 'scepter/methods/studio/scepter_ui.yaml'
    cfg = Config(cfg_file=config_file)
    if 'FILE_SYSTEM' in cfg:
        for fs_info in cfg['FILE_SYSTEM']:
            FS.init_fs_client(fs_info)

    return logger


# Edit the first frame of reference video
def edit_by_stylebooth(output_folder, base_count, input_image, edit_prompt, seed):
    logger = setup_image_edit_environment()
    config_file = 'scepter/methods/studio/inference/edit/stylebooth_tb_pro.yaml'
    cfg = Config(cfg_file=config_file)
    diff_infer = StyleboothInference(logger=logger)
    diff_infer.init_from_cfg(cfg)

    output = diff_infer({'prompt': edit_prompt, 'seed': seed},
                        style_edit_image=input_image,
                        style_guide_scale_text=8.0,
                        style_guide_scale_image=2.0)    # 7.5, 1.5
    save_path = os.path.join(output_folder, f'{base_count:06d}_{edit_prompt}.png')
    save_image(output['images'], save_path)
    # print("Edited image:", output['images'])
    print("Edited image shape:", output['images'].shape)
    print("Edited fisrt frame successfully:", edit_prompt)

    return save_path


# Get video editing models
def get_models(config):
    print('\n' + '=' * 100)
    sys.path.append("./FRESCO_src/ebsynth/deps/gmflow/")
    sys.path.append("./FRESCO_src/EGNet/")
    sys.path.append("./FRESCO_src/ControlNet/")
    
    from gmflow.gmflow import GMFlow
    from model import build_model
    from annotator.hed import HEDdetector
    from annotator.canny import CannyDetector
    from annotator.midas import MidasDetector

    # optical flow
    flow_model = GMFlow(feature_channels=128,
                   num_scales=1,
                   upsample_factor=8,
                   num_head=1,
                   attention_type='swin',
                   ffn_dim_expansion=4,
                   num_transformer_layers=6,
                   ).to('cuda')
    
    checkpoint = torch.load(config['gmflow_path'], map_location=lambda storage, loc: storage)
    weights = checkpoint['model'] if 'model' in checkpoint else checkpoint
    flow_model.load_state_dict(weights, strict=False)
    flow_model.eval() 
    print('create optical flow estimation model successfully!')
    
    # saliency detection
    sod_model = build_model('resnet')
    sod_model.load_state_dict(torch.load(config['sod_path']))
    sod_model.to("cuda").eval()
    print('create saliency detection model successfully!')
    
    # controlnet
    if config['controlnet_type'] not in ['hed', 'depth', 'canny']:
        print('unsupported control type, set to hed')
        config['controlnet_type'] = 'hed'
    controlnet = ControlNetModel.from_pretrained("/data1/huggingface/StableDiffusion/control_v11p_sd15_"+config['controlnet_type'], 
                                                 torch_dtype=torch.float16)
    controlnet.to("cuda") 
    if config['controlnet_type'] == 'depth':
        detector = MidasDetector()
    elif config['controlnet_type'] == 'canny':
        detector = CannyDetector()
    else:
        detector = HEDdetector()
    print('create controlnet model-' + config['controlnet_type'] + ' successfully!')
    
    # diffusion model
    vae = AutoencoderKL.from_pretrained("/data1/huggingface/StableDiffusion/sd-vae-ft-mse", torch_dtype=torch.float16)
    pipe = StableDiffusionPipeline.from_pretrained(config['sd_path'], vae=vae, torch_dtype=torch.float16)
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")
    pipe.scheduler.set_timesteps(config['num_inference_steps'], device=pipe._execution_device)
    
    if config['use_freeu']:
        from src.free_lunch_utils import apply_freeu
        apply_freeu(pipe, b1=1.2, b2=1.5, s1=1.0, s2=1.0)

    frescoProc = apply_FRESCO_attn(pipe)
    frescoProc.controller.disable_controller()
    apply_FRESCO_opt(pipe)
    print('create diffusion model from ' + config['sd_path'] + ' successfully!')
    
    for param in flow_model.parameters():
        param.requires_grad = False    
    for param in sod_model.parameters():
        param.requires_grad = False
    for param in controlnet.parameters():
        param.requires_grad = False
    for param in pipe.unet.parameters():
        param.requires_grad = False
    
    return pipe, frescoProc, controlnet, detector, flow_model, sod_model


def apply_control(x, detector, config):
    if config['controlnet_type'] == 'depth':
        detected_map, _ = detector(x)
    elif config['controlnet_type'] == 'canny':
        detected_map = detector(x, 50, 100)
    else:
        detected_map = detector(x)
    return detected_map


# Edit the frames reference video
def edit_by_FRESCO(output_folder, base_count, input_video, edit_prompt, seed, config_path):
    print('=' * 100)
    print(f'loading video editing config from {config_path} ...' )
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    pipe, frescoProc, controlnet, detector, flow_model, sod_model = get_models(config)
    device = pipe._execution_device
    guidance_scale = 7.5
    do_classifier_free_guidance = guidance_scale > 1
    assert(do_classifier_free_guidance)
    timesteps = pipe.scheduler.timesteps
    cond_scale = [config['cond_scale']] * config['num_inference_steps']
    dilate = Dilate(device=device)
    
    base_prompt = edit_prompt
    if 'Realistic' in config['sd_path'] or 'realistic' in config['sd_path']:
        a_prompt = ', RAW photo, subject, (high detailed skin:1.2), 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3, '
        n_prompt = '(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, mutated hands and fingers:1.4), (deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, amputation'
    else:
        a_prompt = ', best quality, extremely detailed, '
        n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing finger, extra digit, fewer digits, cropped, worst quality, low quality'    

    print('\n' + '=' * 100)    
    video_cap = cv2.VideoCapture(input_video)
    if video_cap.isOpened():
        frame_num = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("Video frame count:", frame_num)
    else:
        print("Failed to open video")
        exit(0)
    keys = get_keyframe_ind(input_video, frame_num, 1, 1)
    
    # you can set extra_prompts for individual keyframe
    # for example, extra_prompts[38] = ', closed eyes' to specify the person frame38 closes the eyes
    extra_prompts = [''] * frame_num
    
    sublists = [keys[i:i+config['batch_size']-2] for i in range(2, len(keys), config['batch_size']-2)]
    sublists[0].insert(0, keys[0])
    sublists[0].insert(1, keys[1])
    if len(sublists) > 1 and len(sublists[-1]) < 3:
        add_num = 3 - len(sublists[-1])
        sublists[-1] = sublists[-2][-add_num:] + sublists[-1]
        sublists[-2] = sublists[-2][:-add_num]

    if not sublists[-2]:
        del sublists[-2]
        
    print('processing %d batches:\nkeyframe indexes'%(len(sublists)), sublists)    
    print('\n' + '=' * 100)
    print('video to video translation...')
    
    batch_ind = 0
    propagation_mode = batch_ind > 0
    imgs = []
    original_images = []
    edit_images = []
    record_latents = []
    video_cap = cv2.VideoCapture(input_video)
    for i in range(frame_num):
        # prepare a batch of frame based on sublists
        success, frame = video_cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = resize_image(frame, 512)
        H, W, C = img.shape
        if i not in sublists[batch_ind]:
            continue
        
        original_images.append(img)
        imgs += [img]
        if i != sublists[batch_ind][-1]:
            continue
        print('processing batch [%d/%d] with %d frames'%(batch_ind+1, len(sublists), len(sublists[batch_ind])))
        
        # prepare input
        batch_size = len(imgs)
        n_prompts = [n_prompt] * len(imgs)
        prompts = [base_prompt + a_prompt + extra_prompts[ind] for ind in sublists[batch_ind]]
        if propagation_mode: # restore the extra_prompts from previous batch
            assert len(imgs) == len(sublists[batch_ind]) + 2
            prompts = ref_prompt + prompts
        prompt_embeds = pipe._encode_prompt(
            prompts,
            device,
            1,
            do_classifier_free_guidance,
            n_prompts,
        ) 
            
        imgs_torch = torch.cat([numpy2tensor(img) for img in imgs], dim=0)
        edges = torch.cat([numpy2tensor(apply_control(img, detector, config)[:, :, None]) for img in imgs], dim=0)
        edges = edges.repeat(1,3,1,1).cuda() * 0.5 + 0.5
        if do_classifier_free_guidance:
            edges = torch.cat([edges.to(pipe.unet.dtype)] * 2)
            
        if config['use_salinecy']:
            saliency = get_saliency(imgs, sod_model, dilate) 
        else:
            saliency = None
        
        # prepare parameters for inter-frame and intra-frame consistency
        flows, occs, attn_mask, interattn_paras = get_flow_and_interframe_paras(flow_model, imgs)
        correlation_matrix = get_intraframe_paras(pipe, imgs_torch, frescoProc, prompt_embeds, seed=seed)
    
        '''
        Flexible settings for attention:
        * Turn off FRESCO-guided attention: frescoProc.controller.disable_controller() 
        Then you can turn on one specific attention submodule
        * Turn on Cross-frame attention: frescoProc.controller.enable_cfattn(attn_mask) 
        * Turn on Spatial-guided attention: frescoProc.controller.enable_intraattn() 
        * Turn on Temporal-guided attention: frescoProc.controller.enable_interattn(interattn_paras)
    
        Flexible settings for optimization:
        * Turn off Spatial-guided optimization: set optimize_temporal = False in apply_FRESCO_opt()
        * Turn off Temporal-guided optimization: set correlation_matrix = [] in apply_FRESCO_opt()
        * Turn off FRESCO-guided optimization: disable_FRESCO_opt(pipe)
    
        Flexible settings for background smoothing:
        * Turn off background smoothing: set saliency = None in apply_FRESCO_opt()
        '''    
        
        # Turn on all FRESCO support
        frescoProc.controller.enable_controller(interattn_paras=interattn_paras, attn_mask=attn_mask)
        apply_FRESCO_opt(pipe, steps = timesteps[:config['end_opt_step']],
                         flows = flows, occs = occs, correlation_matrix=correlation_matrix, 
                         saliency=saliency, optimize_temporal = True)
        
        # gc.collect()
        # torch.cuda.empty_cache()   
        
        # run!
        latents = inference(pipe, controlnet, frescoProc, 
                  imgs_torch, prompt_embeds, edges, timesteps,
                  cond_scale, config['num_inference_steps'], config['num_warmup_steps'], 
                  do_classifier_free_guidance, seed, guidance_scale, config['use_controlnet'],         
                  record_latents, propagation_mode,
                  flows = flows, occs = occs, saliency=saliency, repeat_noise=True)

        with torch.no_grad():
            image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
            image = torch.clamp(image, -1 , 1)
            save_imgs = tensor2numpy(image)
            bias = 2 if propagation_mode else 0
            for ind, num in enumerate(sublists[batch_ind]):
                edit_images.append(save_imgs[ind+bias])
                
        batch_ind += 1
        # current batch uses the last frame of the previous batch as ref
        ref_prompt = [prompts[0], prompts[-1]]
        imgs = [imgs[0], imgs[-1]]
        propagation_mode = batch_ind > 0
        if batch_ind == len(sublists):
            clear_cuda_memory()
            break 

    print('\n' + '=' * 100)
    original_video_save_path = os.path.join(output_folder, f"{base_count:06d}_original_video.mp4")
    edited_video_save_path = os.path.join(output_folder, f"{base_count:06d}_edited_video.mp4")
    with imageio.get_writer(edited_video_save_path, fps=10, format='mp4') as writer:
        for img_array in edit_images:
            writer.append_data(img_array)
        print(f"Edited video saved at {edited_video_save_path}")
    with imageio.get_writer(original_video_save_path, fps=10, format='mp4') as writer:
        for img_array in original_images:
            writer.append_data(img_array)
        print(f"Original video saved at {original_video_save_path}")
    print(len(edit_images), '*', edit_images[0].shape, '\n' + '=' * 100)

    return edit_images

    
# Delete all models and clear CUDA memory
def clear_cuda_memory():
    import gc

    for obj in gc.get_objects():
        if isinstance(obj, torch.nn.Module):
            del obj

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def sample(
    input_path: str = "assets/test_video.mp4",  # Video frames (Can either be image file or folder with image files)
    edit_prompt: str = "Transfrom this image to a low-poly style",
    output_folder: Optional[str] = "outputs/sv4d+edited-multi-view",
    num_steps: Optional[int] = 20,
    sv3d_version: str = "sv3d_u",  # sv3d_u or sv3d_p
    img_size: int = 576, # image resolution
    fps_id: int = 6,
    motion_bucket_id: int = 127,
    cond_aug: float = 1e-5,
    seed: int = 23,
    encoding_t: int = 8,  # Number of frames encoded at a time! This eats most VRAM. Reduce if necessary.
    decoding_t: int = 4,  # Number of frames decoded at a time! This eats most VRAM. Reduce if necessary.
    device: str = "cuda",
    elevations_deg: Optional[Union[float, List[float]]] = 10.0,
    azimuths_deg: Optional[List[float]] = None,
    image_frame_ratio: Optional[float] = 0.917,
    verbose: Optional[bool] = False,
    remove_bg: bool = True,
    FRESCO_config: str = "/data1/Edit4D/FRESCO_src/FRESCO_config.yaml"  # Video translation config
):
    """
    Edit multiple novel-view videos conditioned on a video `input_path` and en edit prompt `edit_prompt`.
    If you run out of VRAM, try decreasing `decoding_t` and `encoding_t`.
    """
    # Set model config
    T = 5  # number of frames per sample
    V = 8  # number of views per sample
    F = 8  # vae factor to downsize image->latent
    C = 4
    H, W = img_size, img_size
    n_frames = 21  # number of input and output video frames
    n_views = V + 1  # number of output video views (1 input view + 8 novel views)
    n_views_sv3d = 21
    subsampled_views = np.array(
        [0, 2, 5, 7, 9, 12, 14, 16, 19]
    )  # subsample (V+1=)9 (uniform) views from 21 SV3D views

    sv4d_model_config = "scripts/sampling/configs/sv4d.yaml"
    base_count = len(glob(os.path.join(output_folder, "*.mp4"))) // 13
    version_dict = {
        "T": T * V,
        "H": H,
        "W": W,
        "C": C,
        "f": F,
        "options": {
            "discretization": 1,
            "cfg": 3.0,
            "sigma_min": 0.002,
            "sigma_max": 700.0,
            "rho": 7.0,
            "guider": 5,
            "num_steps": num_steps,
            "force_uc_zero_embeddings": [
                "cond_frames",
                "cond_frames_without_noise",
                "cond_view",
                "cond_motion",
            ],
            "additional_guider_kwargs": {
                "additional_cond_keys": ["cond_view", "cond_motion"]
            },
        },
    }

    torch.manual_seed(seed)
    os.makedirs(output_folder, exist_ok=True)
    print(f"Input reference video: {input_path}")
    start_time = time.time()

    '''
    # Edit the first frame of the reference video based on the `edit_prompt` by StyleBooth
    reader = imageio.get_reader(input_path)
    first_frame = reader.get_data(0)
    print("First frame shape:", first_frame.shape)
    edit_frame_path = edit_by_stylebooth(output_folder, base_count, Image.fromarray(first_frame), edit_prompt, seed)
    # edit_frame_path = "/data1/Edit4D/outputs/sv4d+edited-multi-view/000000_Apply the style of digital pixel art to this image.png"
    edit_first_frame = preprocess_edit_image(edit_frame_path, remove_bg=True)
    clear_cuda_memory()
    '''

    '''
    # Edit all frames of the reference video based on the `edit_prompt` by FRESCO 
    '''
    edited_video_frames = edit_by_FRESCO(output_folder, base_count, input_path, edit_prompt, seed, FRESCO_config)

    '''
    # Preprocess the edited video frames
    '''
    processed_input_path = preprocess_video(
        edited_video_frames, remove_bg=False, output_folder=output_folder, n_frames=n_frames,
        W=W, H=H, image_frame_ratio=image_frame_ratio,
    )

    '''
    # Preprocess the input reference video frames
    '''
    # processed_input_path = preprocess_video(
    #     input_path, remove_bg=True, output_folder=output_folder, n_frames=n_frames,
    #     W=W, H=H, image_frame_ratio=image_frame_ratio,
    # )

    # read video frames at view 0
    images_v0 = read_video(processed_input_path, n_frames=n_frames, device=device)

    '''
    # calculate `polars_rad` and `azimuths_rad` of the camera to get different viewpoints
    '''
    if isinstance(elevations_deg, float) or isinstance(elevations_deg, int):
        elevations_deg = [elevations_deg] * n_views_sv3d
    assert (
        len(elevations_deg) == n_views_sv3d
    ), f"Please provide 1 value, or a list of {n_views_sv3d} values for elevations_deg! Given {len(elevations_deg)}"
    if azimuths_deg is None:
        azimuths_deg = np.linspace(0, 360, n_views_sv3d + 1)[1:] % 360
    assert (
        len(azimuths_deg) == n_views_sv3d
    ), f"Please provide a list of {n_views_sv3d} values for azimuths_deg! Given {len(azimuths_deg)}"
    polars_rad = np.array([np.deg2rad(90 - e) for e in elevations_deg])
    azimuths_rad = np.array(
        [np.deg2rad((a - azimuths_deg[-1]) % 360) for a in azimuths_deg]
    )

    clear_cuda_memory()
    print('\n' + '=' * 100)
    print('Multi-view generation...')
    # Sample multi-view images of the first frame using SV3D i.e. images at time 0
    images_t0 = sample_sv3d(
        images_v0[0], # edit_first_frame
        n_views_sv3d,
        num_steps,
        sv3d_version,
        fps_id,
        motion_bucket_id,
        cond_aug,
        decoding_t,
        device,
        polars_rad,
        azimuths_rad,
        verbose,
    )   # [21, 3, 576, 576]
    images_t0 = torch.roll(images_t0, 1, 0)  # move conditioning image to first frame
    save_image(images_t0, os.path.join(output_folder, f"{base_count:06d}_t_all.png"))

    # Initialize image matrix
    img_matrix = [[None] * n_views for _ in range(n_frames)]
    for t in range(n_frames):
        img_matrix[t][0] = images_v0[t]
    for i, v in enumerate(subsampled_views):
        img_matrix[0][i] = images_t0[v].unsqueeze(0)

    save_video(
        os.path.join(output_folder, f"{base_count:06d}_t000.mp4"),
        img_matrix[0],
    )
    save_video(
        os.path.join(output_folder, f"{base_count:06d}_v000.mp4"),
        [img_matrix[t][0] for t in range(n_frames)],
    )

    print('\n' + '=' * 100)
    print('4D video frames generation...')
    # Load SV4D model
    model, filter = load_model(
        sv4d_model_config,
        device,
        version_dict["T"],
        num_steps,
        verbose,
    )
    model = initial_model_load(model)
    for emb in model.conditioner.embedders:
        if isinstance(emb, VideoPredictionEmbedderWithEncoder):
            emb.en_and_decode_n_samples_a_time = encoding_t
    model.en_and_decode_n_samples_a_time = decoding_t

    # Interleaved sampling for anchor frames
    t0, v0 = 0, 0
    frame_indices = np.arange(T - 1, n_frames, T - 1)  # [4, 8, 12, 16, 20]
    view_indices = np.arange(V) + 1
    print(f"Sampling anchor frames {frame_indices}")
    image = img_matrix[t0][v0]
    cond_motion = torch.cat([img_matrix[t][v0] for t in frame_indices], 0)
    cond_view = torch.cat([img_matrix[t0][v] for v in view_indices], 0)
    polars = polars_rad[subsampled_views[1:]][None].repeat(T, 0).flatten()
    azims = azimuths_rad[subsampled_views[1:]][None].repeat(T, 0).flatten()
    azims = (azims - azimuths_rad[v0]) % (torch.pi * 2)
    samples = run_img2vid(
        version_dict, model, image, seed, polars, azims, cond_motion, cond_view, decoding_t
    )
    samples = samples.view(T, V, 3, H, W)
    for i, t in enumerate(frame_indices):
        for j, v in enumerate(view_indices):
            if img_matrix[t][v] is None:
                img_matrix[t][v] = samples[i, j][None] * 2 - 1

    # Dense sampling for the rest
    print(f"Sampling dense frames:")
    for t0 in tqdm(np.arange(0, n_frames - 1, T - 1)):  # [0, 4, 8, 12, 16]
        frame_indices = t0 + np.arange(T)
        print(f"Sampling dense frames {frame_indices}")
        latent_matrix = torch.randn(n_frames, n_views, C, H // F, W // F).to("cuda")

        polars = polars_rad[subsampled_views[1:]][None].repeat(T, 0).flatten()
        azims = azimuths_rad[subsampled_views[1:]][None].repeat(T, 0).flatten()
        azims = (azims - azimuths_rad[v0]) % (torch.pi * 2)
        
        # alternate between forward and backward conditioning
        forward_inputs, forward_frame_indices, backward_inputs, backward_frame_indices = prepare_inputs(
            frame_indices, 
            img_matrix, 
            v0, 
            view_indices, 
            model, 
            version_dict, 
            seed, 
            polars, 
            azims
        )
        
        for step in tqdm(range(num_steps)):
            if step % 2 == 1:
                c, uc, additional_model_inputs, sampler = forward_inputs
                frame_indices = forward_frame_indices
            else:
                c, uc, additional_model_inputs, sampler = backward_inputs
                frame_indices = backward_frame_indices
            noisy_latents = latent_matrix[frame_indices][:, view_indices].flatten(0, 1)
                
            samples = do_sample_per_step(
                model,
                sampler,
                noisy_latents,
                c,
                uc,
                step,
                additional_model_inputs,
            )
            samples = samples.view(T, V, C, H // F, W // F)
            for i, t in enumerate(frame_indices):
                for j, v in enumerate(view_indices):
                    latent_matrix[t, v] = samples[i, j]

        img_matrix = decode_latents(model, latent_matrix, img_matrix, frame_indices, view_indices, T)

    # Save output videos
    for v in view_indices:
        vid_file = os.path.join(output_folder, f"{base_count:06d}_v{v:03d}.mp4")
        print(f"Saving {vid_file}")
        save_video(vid_file, [img_matrix[t][v] for t in range(n_frames)])

    # Save diagonal video
    diag_frames = [
        img_matrix[t][(t // (n_frames // n_views)) % n_views] for t in range(n_frames)
    ]

    print('\n' + '=' * 100)
    vid_file = os.path.join(output_folder, f"{base_count:06d}_diag.mp4")
    print(f"Saving {vid_file}")
    save_video(vid_file, diag_frames)
    elapsed_time = time.time() - start_time
    print(f'Totel elapsed time: {elapsed_time:.2f} seconds')


if __name__ == "__main__":
    Fire(sample)
