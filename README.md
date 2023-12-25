# Sticker Recommendation System Based on CLIP

click [here](https://drive.google.com/file/d/15_b5tOd3PEJIvYbfhjg35QwjTLBBSR8s/view) to download dataset.

# requirements
- python3.8+, tested in python3.10
- torch, SentenceTransformer, opencv-python, timm, transformers

you don't need to download dataset if you just want to test the stickers recommendation. 

run this command for train:

```bash
python train.py 
```

parameters to add:
- --base_path (path to dataset) 
- --batch_size (batch size)
Please find more parameters in train.py.

run this command to chat with our model:

```bash
python test.py --model_path path_to_model --stickers_path dir_of_stickers --vocab_path path_to_emoji_vocab
```
(Please find more parameters in test.py. you don't need other parameters if you play with model we provide.)

and then input anything in terminal. you can chat with the model multi-turns. It will reply you with stickers in the directory you provide.

if you don't provide --stickers_path, model will use defaul stickers we provide that is not in dataset. I really love these stickers from acfun. I am happy to see my model could use them to chat without any knowledge about it.

have a good play!
