import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import re
import json
from glob import glob
from sentence_transformers import SentenceTransformer
import random
import matplotlib.pyplot as plt

MAX_CONTEXT_NUM = 15
BATCH_SIZE = 32

SENTENCE_START = "<s>"
SENTENCE_END = "</s>"
PAD_TOKEN = "[PAD]"
UNKNOWN_TOKEN = "[UNK]"
START_DECODING = "[START]"
STOP_DECODING = "[STOP]"


class Vocab(object):
    """Vocabulary class for mapping between words and ids (integers)"""

    def __init__(self, vocab_file, max_size):
        """Creates a vocab of up to max_size words, reading from the vocab_file. If max_size is 0, reads the entire vocab file.

        Args:
          vocab_file: path to the vocab file, which is assumed to contain "<word> <frequency>" on each line, sorted with most frequent word first. This code doesn't actually use the frequencies, though.
          max_size: integer. The maximum size of the resulting Vocabulary."""
        self._word_to_id = {}
        self._id_to_word = {}
        self._count = 0  # keeps track of total number of words in the Vocab

        # [UNK], [PAD], [START] and [STOP] get the ids 0,1,2,3.
        for w in [UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
            self._word_to_id[w] = self._count
            self._id_to_word[self._count] = w
            self._count += 1

        # Read the vocab file and add words up to max_size
        with open(vocab_file, "r", encoding="utf8") as vocab_f:
            for line in vocab_f:
                pieces = line.split()
                if len(pieces) != 2:
                    continue
                w = pieces[0]
                if w in [
                    SENTENCE_START,
                    SENTENCE_END,
                    UNKNOWN_TOKEN,
                    PAD_TOKEN,
                    START_DECODING,
                    STOP_DECODING,
                ]:
                    raise Exception(
                        "<s>, </s>, [UNK], [PAD], [START] and [STOP] shouldn't be in the vocab file, but %s is"
                        % w
                    )
                w = w.lower()
                if w in self._word_to_id:
                    continue
                    # raise Exception('Duplicated word in vocabulary file: %s' % w)
                self._word_to_id[w] = self._count
                self._id_to_word[self._count] = w
                self._count += 1
                if max_size != 0 and self._count >= max_size:
                    print(
                        "max_size of vocab was specified as %i; we now have %i words. Stopping reading."
                        % (max_size, self._count)
                    )
                    break
        print(
            "Finished constructing vocabulary of %i total words. Last word added: %s"
            % (self._count, self._id_to_word[self._count - 1])
        )

    def word2id(self, word):
        """Returns the id (integer) of a word (string). Returns [UNK] id if word is OOV."""
        if word not in self._word_to_id:
            return self._word_to_id[UNKNOWN_TOKEN]
        return self._word_to_id[word]

    def id2word(self, word_id):
        """Returns the word (string) corresponding to an id (integer)."""
        if word_id not in self._id_to_word:
            raise ValueError("Id not found in vocab: %d" % word_id)
        return self._id_to_word[word_id]

    def size(self):
        """Returns the total size of the vocabulary"""
        return self._count


class StickerDataset(Dataset):
    def __init__(
        self,
        emoji_vocab,
        sticker_path,
        context_path,
        max_context_num,
        sticker_height,
        sticker_width,
        num_samples,
        start,
        lang="en",
    ):
        self.emoji_vocab = emoji_vocab
        self.sticker_path = sticker_path
        self.max_context_num = max_context_num
        self.sticker_height = sticker_height
        self.sticker_width = sticker_width
        self.context_path = context_path
        if lang=="en":
            print(f"using english model")
            self.context_encoder = SentenceTransformer("all-MiniLM-L12-v2")
        else:
            print(f"using multi-ling model")
            self.context_encoder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        self.num_samples = num_samples
        self.start = start
        self.data = self.load_data()

    def load_data(self):
        sticker_emoji_mappings = self.load_stickers()
        data = []
        with open(self.context_path, "r", encoding="utf-8") as file:
            counter = 0
            for line in file:
                counter += 1
                if counter < self.start:
                    continue
                item = json.loads(line)
                context_text = self.process_context(
                    item["context"], item.get("reply_to_msg_id")
                )
                sticker_set_id = item["current"]["sticker_set_id"]
                sticker_id = item["current"]["sticker_id"]
                sticker_emoji = item["current"]["sticker_alt"]

                if str(sticker_id) in sticker_emoji_mappings.get(
                    str(sticker_set_id), {}
                ):
                    example = self.create_example(
                        context_text,
                        sticker_set_id,
                        sticker_id,
                        sticker_emoji,
                    )
                    if example is not None:
                        data.append(example)
                print(counter)
                if counter > self.start + self.num_samples:
                    break
        return data

    def process_context(self, context_items, reply_id):
        context_text = []
        reply_text = PAD_TOKEN
        for context_item in context_items:
            text = context_item["text"].replace(" ", "")
            context_text.append(text)
            if context_item["id"] == reply_id:
                reply_text = text
        while len(context_text) < self.max_context_num:
            context_text.append(PAD_TOKEN)
        context_text.append(reply_text)
        return context_text[-self.max_context_num :]

    def create_example(
        self,
        context_text,
        sticker_set_id,
        sticker_id,
        sticker_emoji,
    ):
        sticker_pix = np.load(
            os.path.join(
                self.sticker_path, str(sticker_set_id), str(sticker_id) + ".npy"
            )
        )
        sticker_emoji_id = self.emoji_vocab.word2id(sticker_emoji)

        enc_context = SENTENCE_START + SENTENCE_START.join(context_text) + SENTENCE_END
        enc_context = self.context_encoder.encode(enc_context,device="cuda")
        sticker_pix = torch.tensor(sticker_pix, dtype=torch.float)
        enc_context = torch.tensor(enc_context, dtype=torch.float)
        sticker_emoji_id = torch.tensor(sticker_emoji_id, dtype=torch.int64)
        return {
            "enc_context": enc_context,
            "sticker_pix": sticker_pix,
            "sticker_emoji_id": sticker_emoji_id,
        }

    # def get_negative_samples(
    #     self,
    #     sticker_set_id,
    #     sticker_id,
    #     sticker_emoji_mappings,
    # ):
    #     stickers = sticker_emoji_mappings[str(sticker_set_id)]
    #     negative_sticker_ids = [
    #         sid for sid in stickers.keys() if sid != str(sticker_id)
    #     ]
    #     negative_sticker_ids = negative_sticker_ids[: self.cand_num - 1]
    #     return [
    #         (
    #             # neg id
    #             sid,
    #             # neg pix
    #             np.load(
    #                 os.path.join(self.sticker_path, str(sticker_set_id), sid + ".npy")
    #             ),
    #             # neg emoji id
    #             self.emoji_vocab.word2id(stickers[sid]),
    #             # neg emoji
    #             stickers[sid],
    #         )
    #         for sid in negative_sticker_ids
    #     ]

    def load_stickers(self):
        sticker_emoji_mappings = {}
        for sticker_dir in glob(os.path.join(self.sticker_path, "*")):
            set_id = os.path.basename(sticker_dir)
            with open(
                os.path.join(sticker_dir, "emoji_mapping.txt"), encoding="utf8"
            ) as f:
                stickers = {
                    line.strip().split("\t")[0]: line.strip().split("\t")[1]
                    for line in f
                }
            sticker_emoji_mappings[set_id] = stickers
        return sticker_emoji_mappings

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    

def plot_losses(history_train_loss, history_val_loss,id=None):
    # Set plotting style
    #plt.style.use(('dark_background', 'bmh'))
    plt.figure()
    plt.style.use('bmh')
    plt.rc('axes', facecolor='none')
    plt.rc('figure', figsize=(16, 4))

    # Plotting loss graph
    plt.plot(history_train_loss, label='Train')
    plt.plot(history_val_loss, label='Validation')
    plt.title('Loss Graph')
    plt.legend()
    # plt.show()
    plt.savefig(f'loss{id}.png')
    plt.close()


def prepare_dataset(args, name, num_samples=10000, start=0, lang='en'):
    print(f"preparing {name} set")
    data_path = os.path.join(args.base_path, f"release_{name}.json")
    if lang == 'en':
        lang = ''
    data_set_path = f"./{name}set{start}-{start+num_samples}{lang}.npy"
    if os.path.exists(data_set_path):
        dataset = torch.load(data_set_path)
    else:
        sticker_path = os.path.join(args.base_path, "npy_stickers")
        emoji_vocab = Vocab(os.path.join(args.base_path, "emoji_vocab"), 500)
        dataset = StickerDataset(
            emoji_vocab, sticker_path, data_path, args.max_context_num, 128, 128, num_samples, start,lang=lang
        )
        torch.save(dataset, data_set_path)
    return dataset


if __name__ == "__main__":
    base_path = "./stickerchat"
    sticker_path = os.path.join(base_path, "npy_stickers")
    emoji_vocab = Vocab(os.path.join(base_path, "emoji_vocab"), 500)
    test_path = os.path.join(base_path, "release_test.sjson")
    test_set = StickerDataset(
        emoji_vocab, sticker_path, test_path, MAX_CONTEXT_NUM, 128, 128, 10
    )
    # test_dataloader = DataLoader(test_set, batch_size=1)
