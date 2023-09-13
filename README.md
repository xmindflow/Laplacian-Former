# Laplacian-Former: Overcoming the Limitations of Vision Transformers in Local Texture Detection <br> <span style="float: right"><sub><sup>$\text{\textcolor{teal}{MICCAI 2023}}$</sub></sup></span>

[![arXiv](https://img.shields.io/badge/arXiv-2309.00108-b31b1b.svg)](https://arxiv.org/abs/2309.00108)

Vision Transformer (ViT) models have demonstrated a breakthrough in a wide range of computer vision tasks. However, compared to the Convolutional Neural Network (CNN) models, it has been observed that the ViT models struggle to capture high-frequency components of images, which can limit their ability to detect local textures and edge information. As abnormalities in human tissue, such as tumors and lesions, may greatly vary in structure, texture, and shape, high-frequency information such as texture is crucial for effective semantic segmentation tasks. To address this limitation in ViT models, we propose a new technique, Laplacian-Former, that enhances the self-attention map by adaptively re-calibrating the frequency information in a Laplacian pyramid. More specifically, our proposed method utilizes a dual attention mechanism via efficient attention and frequency attention while the efficient attention mechanism reduces the complexity of self-attention to linear while producing the same output, selectively intensifying the contribution of shape and texture features. Furthermore, we introduce a novel efficient enhancement multi-scale bridge that effectively transfers spatial information from the encoder to the decoder while preserving the fundamental features. 

<br>

<p align="center">
  <img src="https://github.com/mindflow-institue/Laplacian-Former/assets/61879630/6b223292-dad3-4e4d-835b-eea070626437" width="700">
</p>

## Citation

```python
@article{azad2023laplacian,
  title={Laplacian-Former: Overcoming the Limitations of Vision Transformers in Local Texture Detection},
  author={Azad, Reza and Kazerouni, Amirhossein and Azad, Babak and Aghdam, Ehsan Khodapanah and Velichko, Yury and Bagci, Ulas and Merhof, Dorit},
  journal={arXiv preprint arXiv:2309.00108},
  year={2023}
}
```

## News
- May 25, 2023: Accepted in **MICCAI 2023**! 🥳🔥

## Train

1) Download the Synapse Dataset from [here](https://drive.google.com/uc?export=download&id=18I9JHH_i0uuEDg-N6d7bfMdf7Ut6bhBi).

2) Run the following code to install the requirements.

    ```python
   pip install -r requirements.txt
    ```

4) Run the below code to train the model on the synapse dataset.
  > <picture>
  >   <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/Mqxx/GitHub-Markdown/main/blockquotes/badge/light-theme/note.svg">
  >   <img alt="Note" src="https://raw.githubusercontent.com/Mqxx/GitHub-Markdown/main/blockquotes/badge/dark-theme/note.svg">
  > </picture><br>
  >
  >    If you want to train the compact version of the model with only three encoders, replace the `train.py` with `train_compact.py`.

  ```python
    python train.py --root_path ./data/Synapse/train_npz --test_path ./data/Synapse/test_vol_h5  --batch_size 24 --eval_interval 20 --max_epochs 400 --dst_fast --resume --model_path [MODEL PATH]
  ```
    
   ```
    --root_path    [Train data path]

    --test_path    [Test data path]

    --eval_interval [Evaluation epoch]

    --dst_fast [Optional] [Load all data into RAM for faster training]

    --resume [Optional] [Resume from checkpoint]

    --model_path [Optional] [Provide the path to the latest checkpoint file for loading the model.]
   ```

> <picture>
>   <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/Mqxx/GitHub-Markdown/main/blockquotes/badge/light-theme/note.svg">
>   <img alt="Note" src="https://raw.githubusercontent.com/Mqxx/GitHub-Markdown/main/blockquotes/badge/dark-theme/note.svg">
> </picture><br>
>
> For information regarding training the skin dataset, please refer to this [link](https://github.com/mindflow-institue/deformableLKA/tree/main/2D).


## Test 

1) Download the learned weights from the below link:

    Dataset   | Model | Download link 
    -----------|-------|----------------
    Synapse   | Laplacian-Former   | [Download](https://drive.google.com/uc?export=download&id=1StaVExctaAFLaq9-onZVu2EtiglimD1n)

<br>

2) Run the below code to test the model on the synapse dataset.
  
    
    ```python
    python test.py --test_path ./data/Synapse/test_vol_h5 --is_savenii --pretrained_path './best_model.pth'
    ```

    ```
    --test_path     [Test data path]
        
    --is_savenii    [Whether to save results during inference]

    --pretrained_path  [Pretrained model path]
    ```


  
## Experiments
For evaluating the performance of the proposed method, two challenging tasks in medical image segmentation have been considered: Synapse Dataset and ISIC 2018 Dataset. The proposed Laplacian-Former achieves superior segmentation performance.
> Our results in the table are updated according to the model weight. 

<p align="center">
  <img style="width: 90%;" alt="Synapse results" src="https://github.com/mindflow-institue/Laplacian-Former/assets/61879630/0c93b40f-30ec-4f93-b026-c1c72becdbec">
  <img style="width: 46%;" alt="ISIC results" src="https://github.com/mindflow-institue/Laplacian-Former/assets/61879630/44e3f784-47c1-4212-aa93-dc3cbc6b6485">
</p>


## References
- [DAEFormer](https://github.com/mindflow-institue/DAEFormer)
- [HiFormer](https://github.com/amirhossein-kz/HiFormer)
