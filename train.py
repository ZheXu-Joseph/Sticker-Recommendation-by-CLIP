import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
import os
import argparse
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt

from data import StickerDataset, Vocab, prepare_dataset, plot_losses
from model import StickerClassify

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(filename='train.log',level=logging.INFO)


def train(model, train_loader, val_loader, optimizer, lr_scheduler, epoch=10, test=False,id=None,best_val_loss=100,lang='en'):
    model = model.to(device)
    train_loss_log = []
    val_loss_log = []
    train_loss = 0
    emoji_loss = 0
    pbar = tqdm(range(epoch))
    for epoch in pbar:
        model.train()
        total_train_loss = []
        total_emoji_loss = []
        for batch in train_loader:
            # positive samples
            enc_context = batch["enc_context"].to(device)
            loss, _, _, emoji_logits = model(
                text_embeddings=enc_context,
                sticker_pix=batch["sticker_pix"].to(device),
            )
            optimizer.zero_grad()
            sticker_emoji = batch["sticker_emoji_id"]
            emoji_loss = model.emoji_loss(emoji_logits, sticker_emoji)
            loss += 0.3 * emoji_loss
            loss.backward()
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step(loss)
            total_train_loss.append(loss.item())
            total_emoji_loss.append(emoji_loss.item())
        train_loss = sum(total_train_loss) / len(total_train_loss)  
        emoji_loss = sum(total_emoji_loss) / len(total_emoji_loss)
        # pbar.set_description(f"trainloss: {train_loss}")
        print(f"trainloss: {train_loss}, emoji_loss:{emoji_loss}")

        # evaluate
        model.eval()
        total_val_loss = []
        total_hits1 = 0
        total_hits2 = 0
        total_hits5 = 0
        total_val_num = 0
        with torch.no_grad():
            for batch in val_loader:
                enc_context = batch["enc_context"].to(device)
                if enc_context.shape[0] != 10: continue
                val_loss, sticker_embeddings, context_embeddings, emoji_logits = model(
                    enc_context,
                    sticker_pix=batch["sticker_pix"].to(device),
                )
                sticker_emoji = batch["sticker_emoji_id"]
                emoji_loss = model.emoji_loss(emoji_logits, sticker_emoji)
                val_loss += 0.3 * emoji_loss
                total_val_loss.append(val_loss.item())
                # sticker_embeddings = outputs["sticker_embedding"]
                # context_embeddings = outputs["context_embedding"]
                sticker_prob = context_embeddings @ sticker_embeddings.T
                _, indices = sticker_prob.topk(5)
                labels = torch.arange(0,indices.shape[0]).to(device)
                total_hits1 += (indices[:,0].reshape(-1,1) == labels.reshape(-1,1)).any(1).sum().float()
                total_hits2 += (indices[:,:2].reshape(-1,2) == labels.reshape(-1,1)).any(1).sum().float()
                total_hits5 += (indices[:,:5].reshape(-1,5) == labels.reshape(-1,1)).any(1).sum().float()
                total_val_num += enc_context.shape[0]
                # labels = torch.ones(logits.shape[0], dtype=torch.float)
        val_loss = sum(total_val_loss) / len(total_val_loss)
        Hits1 = total_hits1/total_val_num
        Hits2 = total_hits2/total_val_num
        Hits5 = total_hits5/total_val_num
        train_loss_log.append(train_loss)
        val_loss_log.append(val_loss)
        print(f"trainloss: {train_loss}, valloss:{val_loss}\
            Hits1:{Hits1},Hits2:{Hits2},Hits5:{Hits5}")
        logging.info(f"trainloss: {train_loss}, valloss:{val_loss}\
            Hits1:{Hits1},Hits2:{Hits2},Hits5:{Hits5}")

        if val_loss < best_val_loss and not test:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"best_{args.model}_{lang}.pt")
            print(f"save model to best_{args.model}_{lang}.pt")
    # plot_losses(train_loss_log, val_loss_log,id=id)
    return model, best_val_loss

def parse_args():
    parser = argparse.ArgumentParser(description="Sticker Classify Model Arguments")
    parser.add_argument("--base_path", type=str, default="./sticker")
    parser.add_argument(
        "--bert", type=str, default="multi-qa-MiniLM-L6-cos-v1"
    )  # multi-qa-mpnet-base-dot-v1, multi-qa-mpnet-base-cos-v1
    parser.add_argument("--sticker_height", type=int, default=128)
    parser.add_argument("--sticker_width", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--emoji_vocab_size", type=int, default=500)
    parser.add_argument("--train_bert", action="store_true")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--epoch", type=float, default=5)
    parser.add_argument("--max_context_num", type=int, default=5)
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument("--lang", type=str, default="multi")

    args = parser.parse_args()
    return args



if __name__ == "__main__":
    print(device)
    args = parse_args()

    model = StickerClassify(args)
    model_path = f"./best_{args.model}_{args.lang}.pt"
    if os.path.exists(model_path):
        print('loading model')
        model.load_state_dict(torch.load(model_path))
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    best_val_loss = 100
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode="min", patience=2, factor=0.5
    # )

 # just prepare and save all train dataset
    # for i in range(10):
    #     train_set = prepare_dataset(args, "train", 30000, i*30000, lang=args.lang)
    #     train_set = 0

    val_set = prepare_dataset(args, "val",lang=args.lang)

    # quick test
    # train_set, val_set = torch.utils.data.random_split(
    #     val_set, [len(val_set) - 2500, 2500]
    # )
    # train_loader = DataLoader(
    #             train_set,
    #             batch_size=args.batch_size,
    #             pin_memory=True,
    #         )


    val_loader = DataLoader(
        val_set,
        batch_size=10,
        pin_memory=True,
    )
    # model, best_val_loss = train(model, train_loader, val_loader, optimizer, lr_scheduler=None, epoch=50,lang=args.lang, test=True)
    # exit()

    # train
    for j in range(2):
        for i in range(10):
            train_set = prepare_dataset(args, "train", 30000, i*30000,lang=args.lang)
            train_loader = DataLoader(
                train_set,
                batch_size=args.batch_size,
                pin_memory=True,
            )
            model, best_val_loss = train(model, train_loader, val_loader, optimizer, lr_scheduler=None, epoch=args.epoch,id=j*100+i, best_val_loss=best_val_loss,lang=args.lang)
            train_set = 0
            train_loader = 0