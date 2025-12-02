# OR-AUIT:Enhancing Facial Action Unit Intensity Estimation with Ordinal Regression-Enhanced Transformer
## Submission to The Visual Computer Journal

🔧 Requirements
=
- Python 3
- PyTorch


- Check the required python packages in `requirements.txt`.
```
pip install -r requirements.txt
```

Data and Data Prepareing Tools
=
The Datasets we used:
  * [BP4D](http://www.cs.binghamton.edu/~lijun/Research/3DFE/3DFE_Analysis.html)
  * [DISFA](http://mohammadmahoor.com/disfa-contact-form/)
  * 
Make sure that you download the  pre-trained [mae-face](https://github.com/facebookresearch/mae?tab=readme-ov-file) model to `checkpoints/` 

Train and Test

```
python main_finetune.py --fintune mae_pretrain_vit_base.pth --train_suffix_path  BP4D_224  --train_suffix_path  BP4D_combine_1_2 --batch_size 64 -lr 0.00005 
```
