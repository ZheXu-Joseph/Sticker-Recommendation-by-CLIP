import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import argparse
from tqdm import tqdm
from glob import glob

from data import StickerDataset, Vocab, prepare_dataset, plot_losses
from model import StickerClassify
from train import prepare_dataset
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_batch(model,  enc_context, sticker_pixs, k=1, emoji_vocab=None):
    enc_context = enc_context.to(device)
    sticker_pixs = sticker_pixs.to(device)
    sticker_embeddings, emoji_logits = model.sticker_encode(sticker_pixs)
    emoji = emoji_logits.argmax(dim=-1).detach().cpu().numpy()
    
    context_embeddings = model.context_head(enc_context)
    sticker_prob = context_embeddings @ sticker_embeddings.T
    _, indices = sticker_prob.topk(k)
    emoji = [emoji_vocab.id2word(emoji[indices.view(-1)[i]]) for i in range(k)]
    return indices, emoji

def test(model, test_loader):
    model.eval()
    total_val_loss = []
    total_hits1 = 0
    total_hits2 = 0
    total_hits5 = 0
    total_val_num = 10000
    with torch.no_grad():
        for batch in test_loader:
            enc_context = batch["enc_context"].to(device)
            if enc_context.shape[0] != 10: continue
            val_loss, sticker_embeddings, context_embeddings, emoji_logits = model(
                enc_context,
                sticker_pix=batch["sticker_pix"].to(device),
            )
            # sticker_emoji = torch.nn.functional.one_hot(
            #     batch["sticker_emoji_id"], num_classes=500
            # )
            # val_loss = model.emoji_loss(outputs, sticker_emoji)
            # val_loss = outputs["CLIP_loss"]
            total_val_loss.append(val_loss.item())
            # sticker_embeddings = outputs["sticker_embedding"]
            # context_embeddings = outputs["context_embedding"]
            sticker_prob = context_embeddings @ sticker_embeddings.T
            _, indices = sticker_prob.topk(5)
            labels = torch.arange(0,indices.shape[0]).to(device)
            total_hits1 += (indices[:,0].reshape(-1,1) == labels.reshape(-1,1)).any(1).sum().float()
            total_hits2 += (indices[:,:2].reshape(-1,2) == labels.reshape(-1,1)).any(1).sum().float()
            total_hits5 += (indices[:,:5].reshape(-1,5) == labels.reshape(-1,1)).any(1).sum().float()
            # labels = torch.ones(logits.shape[0], dtype=torch.float)
    val_loss = sum(total_val_loss) / len(total_val_loss)
    Hits1 = total_hits1/total_val_num
    Hits2 = total_hits2/total_val_num
    Hits5 = total_hits5/total_val_num
    print(f" valloss:{val_loss} Hits1:{Hits1},Hits2:{Hits2},Hits5:{Hits5}")

def parse_args():
    parser = argparse.ArgumentParser(description="Sticker Classify Model Arguments")
    parser.add_argument("--base_path", type=str, default="./sticker")
    parser.add_argument("--stickers_path", type=str, default='./acfun')
    parser.add_argument("--npy", action="store_true")
    parser.add_argument("--sticker_height", type=int, default=128)
    parser.add_argument("--sticker_width", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--emoji_vocab_size", type=int, default=500)
    parser.add_argument("--train_bert", action="store_true")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--epoch", type=float, default=5)
    parser.add_argument("--max_context_num", type=int, default=5)
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument("--model_path", type=str, default="best_resnet50_multi.pt")
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--vocab_path", type=str, default="./sticker/emoji_vocab")

    


    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    base_path = args.base_path
    stickers_path = args.stickers_path
    stickers = []
    model = StickerClassify(args)
    model.load_state_dict(torch.load(args.model_path))
    model = model.to(device)
    if args.measure:
        test_set = prepare_dataset(args, "test",lang="multi")
        test_dataloader = DataLoader(test_set, batch_size=32)
        test(model, test_dataloader)
    for sticker_path in glob(os.path.join(stickers_path, "*")):
        if args.npy:
            img = np.load(sticker_path)
        else:
            try:
                img = cv2.imread(sticker_path)
                img = cv2.resize(img, (128,128))
            except:
                print(sticker_path)
            if img.shape[-1] > 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        stickers.append(img)

    sticker_pixs = torch.tensor(stickers, dtype=float)
    emoji_vocab = Vocab(args.vocab_path, 500)
    context_encoder = SentenceTransformer("all-MiniLM-L12-v2")

    dialogue = []
    # plt.figure(figsize=(10,10))
    sentence = 'test'
    plt.ion()
    _, axes = plt.subplots(1,5)
    print('hello world.')
    while True and len(sentence) > 1:
        print("Your input: (type 'exit' to quit))")
        sentence = input()
        if sentence == 'exit':
            break
        dialogue.append(sentence)
        enc_context = context_encoder.encode("".join(dialogue[:-5])+sentence, device='cuda',convert_to_tensor=True)
        indices,emoji = evaluate_batch(model, enc_context, sticker_pixs, 5, emoji_vocab)
        prediction = indices.view(-1)
        for i in range(5):
            axes[i].imshow(stickers[prediction[i].item()])
            axes[i].set_title(emoji[i])
        plt.pause(0.1)
        plt.show()
        
    


        