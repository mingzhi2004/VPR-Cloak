# VPR-Cloak

This repository provides a PyTorch implementation of **VPR-Cloak**, from the following paper

[VPR-Cloak](https://openaccess.thecvf.com/content/ICCV2025/papers/Dong_VPR-Cloak_A_First_Look_at_Privacy_Cloak_Against_Visual_Place_ICCV_2025_paper.pdf)  
Shuting Dong*, [Mingzhi Chen*](https://mingzhic.com/), [Feng Lu](https://github.com/Lu-Feng), Hao Yu, Guanghao Li, [Ming Tang](https://mingtang.site/), and [Chun Yuan](https://www.sigs.tsinghua.edu.cn/yc2_en/main.psp)  
Tsinghua University, Pengcheng Laboratory, Southern University of Science and Technology [[`paper`](https://openaccess.thecvf.com/content/ICCV2025/papers/Dong_VPR-Cloak_A_First_Look_at_Privacy_Cloak_Against_Visual_Place_ICCV_2025_paper.pdf)]


--- 

**VPR-Cloak** is an efficient privacy-preserving framework that generates imperceptible, transferable perturbations to protect location privacy against black-box visual place recognition systems in real-world scenarios.

<p align="center">
<img src="figure/VPR-Cloak.png" width="100%">
</p>

## Installation

### Requirements

Create a conda environment and install dependencies:

```bash
conda create -n vpr-cloak python=3.9
conda activate vpr-cloak
```

Install PyTorch and other dependencies:

```bash
pip install torch==2.0.0 torchvision==0.15.0

pip install -r requirements.txt
```

### Dataset Setup

Set the dataset folder path:

```bash
export DATASETS_FOLDER=/path/to/your/datasets
```

Or specify it in the command line using `--eval_datasets_folder`.

### Pretrained Model

Download the pretrained VPR model checkpoint and place it in a directory (e.g., `./checkpoints/`). The model path will be specified using the `--resume` argument.

### Weight Maps (for Weighted Attack)

For the weighted attack method, you need pre-computed attention weight maps:
- **Location**: `./Dinov2_weight/get_weighted_$DATASET/`
- **Format**: `.npy` files (NumPy arrays)
- **Naming**: Weight map files should match the image filenames (e.g., `image001.jpg` → `image001.npy`)
- **Content**: Attention weight maps from DINOv2 model indicating important regions

## Implementation

### 1. Weighted Attack

The weighted attack applies spatially varying perturbations based on attention weight maps.

**Command:**

```bash
python attack_weighted.py \
    --resume /path/to/pretrained_model.pth \
    --eval_datasets_folder /path/to/datasets \
    --eval_dataset_name $DATASET \
    --infer_batch_size 64 \
    --num_workers 10 \
    --save_dir weighted_attack_experiment
```

**Required Files:**
- **Model**: Pretrained VPR model at path specified by `--resume`
- **Weight Maps**: `.npy` files in `./Dinov2_weight/get_weighted_$DATASET/`
  - Format: NumPy array with normalized attention weights [0, 1]
  - Shape: Matches image dimensions (resized during loading)


### 2. Frequency Attack

The frequency attack applies perturbations in the DCT frequency domain to target high-frequency components.

**Command:**

```bash
python attack_frequency.py \
    --resume /path/to/pretrained_model.pth \
    --eval_datasets_folder /path/to/datasets \
    --eval_dataset_name $DATASET \
    --infer_batch_size 16 \
    --num_workers 4 \
    --save_dir frequency_attack_experiment
```

Note: `$DATASET` can be `pitts30k`, `pitts250k`, `tokyo247`, `msls`, etc.

**Required Files:**
- **Model**: Pretrained VPR model at path specified by `--resume`
- No additional weight maps needed

## Evaluation

### 1. VPR Performance Evaluation (`eval.py`)

Evaluate the effectiveness of the attack by measuring retrieval performance (Recall@N) on the perturbed images.

```bash
python eval.py \
    --resume /path/to/pretrained_model.pth \
    --eval_datasets_folder /path/to/attack_output \
    --eval_dataset_name test \
    --test_method hard_resize \
    --recall_values 1 5 10 20 \
    --save_dir evaluation_results
```

### 2. Visual Fidelity Evaluation

Measure the imperceptibility of the perturbations.

```bash
python visual_fidelity.py \
    --reference_dir /path/to/original_queries \
    --target_dir /path/to/attack_output/test/queries
```



## Acknowledgements

Parts of this repo are inspired by the following repositories:

[GSV-Cities](https://github.com/amaralibey/gsv-cities) | [Visual Geo-localization Benchmark](https://github.com/gmberton/deep-visual-geo-localization-benchmark) | [DINOv2](https://github.com/facebookresearch/dinov2)

[CricaVPR](https://github.com/Lu-Feng/CricaVPR) | [SelaVPR](https://github.com/Lu-Feng/SelaVPR) | [VLAD-BuFF](https://github.com/Ahmedest61/VLAD-BuFF) | [DHE-VPR](https://github.com/Lu-Feng/DHE-VPR) | [salad](https://github.com/serizba/salad)




## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{VPRCloak,
  title={VPR-Cloak: A First Look at Privacy Cloak Against Visual Place Recognition},
  author={Dong, Shuting and Chen, Mingzhi and Lu, Feng and Yu, Hao and Li, Guanghao and Wu, Zhe and Tang, Ming and Yuan, Chun},
  booktitle={ICCV},
  year={2025}
}
```

## License

See [LICENSE](LICENSE) file for details.
