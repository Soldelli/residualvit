# ResidualViT: Efficient Temporally Dense Video Encoding

[📑 Paper](https://arxiv.org/abs/REPLACE) · [🎥 Project Page](https://soldelli.github.io/residualvit/)

Official PyTorch implementation of **ResidualViT for Efficient Temporally Dense Video Encoding**, accepted at **ICCV 2025 (highlight paper)**.  

![pool_figure](https://private-user-images.githubusercontent.com/26504816/490045120-11b28191-11e4-4811-a9ed-4deb91317e06.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NTgwMjY3NzEsIm5iZiI6MTc1ODAyNjQ3MSwicGF0aCI6Ii8yNjUwNDgxNi80OTAwNDUxMjAtMTFiMjgxOTEtMTFlNC00ODExLWE5ZWQtNGRlYjkxMzE3ZTA2LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA5MTYlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwOTE2VDEyNDExMVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWFlM2IwNzlhYTJiN2U3NDA5NjUwZDlhODM1ODZhOGZmMWEwZmQyMzRmN2EyMDI0MmMyZTUwNWFmYTk3NTdiM2ImWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.HHAGej-Eu_H7CveYODLUMn00y-Ds-vx_GpKX4GdXXPU)




# 🧪 Experiments
## • Training

#### 🚀 Installation

```bash
git clone https://github.com/Soldelli/residualvit.git
cd residualvit

python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

# add repo to module path
export PYTHONPATH="$PYTHONPATH:$PWD/src"
```

#### Training Dataset

- Training is performed on WebVid-2.5M ([website]((https://github.com/m-bain/webvid))), pre-sharded with WebDataset
- Videos are preprocessed to 5 FPS and 360p resolution for efficient I/O.
- Use the `--train-data` flag to point to your dataset shards.

To train on a custom dataset, preprocess it into WebDataset format and update the `--train-data` path accordingly.


#### Training Scripts
Training scripts are provided in `/scripts/training`.
Example: run ResidualViT-B32 with token dropping on a SLURM cluster:
```bash
sbatch ./scripts/training/ResidualViT_token_dropping.sh
```
Other configurations:
```bash
sbatch ./scripts/training/ResidualViT_token_merging.sh
sbatch ./scripts/training/ResidualViT_lower_res_inputs.sh
```

Modify script arguments (batch size, learning rate, dataset path, logging path) as required.
Update placeholders (`#PATH_TO_REPOSITORY`, `#PATH_TO_WEBVID`) with your actual paths.


### Pretrained Checkpoints 
| Model             |    Strategy    |   Checkpoint  |
|-------------------|----------------|---------------|
| ResidualViT-B32   | Token Dropping | [Download](https://drive.google.com/file/d/1Kc7da_MnsLgcKY43mIAyO1FBC1S9YHum/view?usp=sharing) |
| ResidualViT-B16   | Token Dropping | [Download](https://drive.google.com/file/d/1Rne6o-O4_bZ4Cn62HhobqMHofaeJyo56/view?usp=sharing) |
| ResidualViT-L14   | Token Dropping | [Download](https://drive.google.com/file/d/1LiMYhOhCFb5dkSXnm-Obrc1VnkdV-9Ht/view?usp=sharing) |



## • Evaluation
Evaluation is conducted using the [zs-video-eval](https://github.com/adobe-research/zs-video-eval)
 repository.


### Installation

```bash
git clone https://github.com/adobe-research/zs-video-eval.git ../
cd  ../zs-video-eval

conda env create -f environment.yml
conda activate sm
pip install --no-dependencies git+https://github.com/Soldelli/residualvit

export PYTHONPATH="$PYTHONPATH:$PWD"
```

### Datasets
The evaluation repo supports:

| Dataset           |    Annotations    |    Videos     |   Motion Vectors    |
|-------------------|------------------ |---------------|---------------------|
| Charades-STA      | [Download](https://drive.google.com/file/d/1guZfHEsZwWCAm4NCttOzuufsLEA9l2jI/view?usp=sharing) | [Website](https://prior.allenai.org/projects/charades) | TBD |
|  ActivityNet-Captions   | [Download](https://drive.google.com/file/d/1ZI8GUuA7Tk5mblTlLVOgp2WTdXYA2lA9/view?usp=sharing) | [Website](http://activity-net.org/) | TBD |


### Inference Scripts
We provide an example of inference script in the `/scripts/evaluation` directory.
Correctly set `#PATH_TO_THE_REPO` and `#PATH_TO_MODEL_FOLDER`.






## 📂 Repository Structure
```bash
residualvit/
├── scripts/          # Training scripts
├── src/              # Core source code
├── LICENSE.md        # Project license
├── MANIFEST.in       # Packaging config 
├── README.md         # Project documentation
├── requirements.txt  # Python dependencies
├── setup.py          # Packaging script 
└── ...
```





## 💡 Citation
If you use this code or find it helpful in your research, please cite our paper:
```bibtex
@inproceedings{soldan2025residualvit,
  title={ResidualViT for Efficient Temporally Dense Video Encoding},
  author={Soldan, Mattia and Caba Heilbron, Fabian and Ghanem, Bernard and Sivic, Josef and Russell, Bryan},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2025}
}
```




## 🙏 Acknowledgements
This repository is built on top of [OpenCLIP](https://github.com/mlfoundations/open_clip), thanks to our collaborators and open-source community.





## 📜 License
This project is licensed under the [Apache License 2.0](LICENSE.txt).