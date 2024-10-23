## Install
Use the following commands to build a specific environment of Edit4D:
``` shell
conda create -n Edit4D python=3.10
conda activate Edit4D

git clone https://github.com/DINGYANB/Edit4D.git

cd Edit4D
pip install -r requirement.txt
```

## Usage

Download models for **image/video editing**:
- Image editing by [StyleBooth](https://github.com/modelscope/scepter/blob/main/docs/en/tasks/stylebooth.md): the ckpts will be automated download via ModelScope after rununing the main script.

- Video editing by [FRESCO](https://github.com/williamyang1991/FRESCO/blob/main/README.md): use `python FRESCO_src/install.py` to download the corresponding models weights.


To run **Edit4D** on a single input video of 21 frames:
- Download SV3D models (`sv3d_u.safetensors` or `sv3d_p.safetensors`) from [here](https://huggingface.co/stabilityai/sv3d) and SV4D model (`sv4d.safetensors`) from [here](https://huggingface.co/stabilityai/sv4d).

- Run the following command, where `input_path` is the input video path, and `edit_prompt` is the prompt used for editing.

``` shell
python scripts/sampling/simple_video_sample_4d.py --input_path assets/sv4d_videos/test_video1.mp4 --edit_prompt "Apply the style of digital pixel art to this image"
```


**Other parameters includes**:

- `num_steps` : default is 20, can increase to 50 for better quality but longer sampling time.
  
- `sv3d_version` : To specify the SV3D model to generate reference multi-views, set `--sv3d_version=sv3d_u` for SV3D_u or `--sv3d_version=sv3d_p` for SV3D_p.

- `elevations_deg` : To generate novel-view videos at a specified elevation (default elevation is 10) using SV3D_p (default is SV3D_u), e.g., set `--elevations_deg 30.0`.

- **Background removal** : For input videos with plain background, (optionally) use [rembg](https://github.com/danielgatis/rembg) to remove background and crop video frames by setting `--remove_bg=True`. To obtain higher quality outputs on real-world input videos with noisy background, try segmenting the foreground object using [Clipdrop](https://clipdrop.co/) or [SAM2](https://github.com/facebookresearch/segment-anything-2) before running Edit4D.

- **Low VRAM environment** : To run on GPUs with low VRAM, try setting `--encoding_t=1` (of frames encoded at a time) and `--decoding_t=1` (of frames decoded at a time) or lower video resolution like `--img_size=512`.



<div style="display: flex; justify-content: space-between;">
  <div>
    <video width="320" height="240" controls>
      <source src="https://github.com/DINGYANB/Edit4D/blob/main/assets/000000_original_video.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
  </div>
  <div>
    <video width="320" height="240" controls>
      <source src="https://github.com/DINGYANB/Edit4D/blob/main/assets/000000_edited_video.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
  </div>
</div>
