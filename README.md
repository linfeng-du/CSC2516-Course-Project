# CSC2516-Course-Project

## Preprae data
1. Follow the instructions at https://github.com/Seth-Park/comp-t2i-dataset to download dataset.

2. Download ImageNet labels at https://drive.google.com/file/d/11pcVFcf-HUhNbjlgtsocyf4YwD1B53RR/view?usp=share_link and save it to ./data

3. Run ```python  prepare_imagenet_label.py```

4. OR Download preprocessed sorted ImageNet labels at https://drive.google.com/file/d/10Ke_x0IALG9iCUUFXPL3ZDJMZqBVmOuo/view?usp=share_link and save it to ./data/C-CUB

You can skip step 2 and 3, if you directly download preprocess sorted data following step 4.

## Generate images from CUB captions
```
python random_select.py --condition --gan_type biggan--max_images 100 --k 10
python random_select.py --condition --gan_type studiogan --gan_model_name SAGAN --max_images 100 --k 10
```
If you specify --gan_type as studiogan, then you should also specify --gan_model_name. More supported gan models can be found at https://github.com/lukemelas/pytorch-pretrained-gans.

You can find generated images at ./gen, generated plots at ./plot.