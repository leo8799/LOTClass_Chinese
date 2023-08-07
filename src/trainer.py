import os.path
import sys

import numpy as np

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler


from src.model import LOTClassModel
from config.configs_interface import Configs
from src.logers import LOGS

from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from collections import defaultdict
from joblib import Parallel, delayed
from wobert import WoBertTokenizer
from math import ceil
from multiprocessing import cpu_count


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_stop_words(path):
    stop_words = set()
    with open(path, mode='r', encoding="utf-8") as rf:
        for line in rf:
            stop_words.add(line.strip())
    return stop_words


class LOTClassTrainer(object):
    def __init__(self, args: Configs):
        self.args = args
        self.max_len = args.train_args.MAX_LEN
        self.dataset_dir = args.data.DATASET
        self.model_dir = args.data.MODEL
        self.num_cpus = min(4, cpu_count() - 4) if cpu_count() > 1 else 1
        self.world_size = args.train_args.GPUS
        self.train_batch_size = args.train_args.TRAIN_BATCH
        self.eval_batch_size = args.train_args.EVAL_BATCH
        self.accum_steps = args.train_args.ACCUM_STEP
        eff_batch_size = self.train_batch_size * self.world_size * self.accum_steps
        assert abs(
            eff_batch_size - 256) < 10, \
            "Make sure the effective training batch size is around 256, current: {}".format(eff_batch_size)
        LOGS.log.debug("Effective training batch size:{}".format(eff_batch_size))

        # 加载模型
        self.pretrained_lm = args.train_args.pretrained_weights_path
        self.tokenizer = WoBertTokenizer.from_pretrained(self.pretrained_lm, do_lower_case=True)
        self.vocab = self.tokenizer.get_vocab()
        self.vocab_size = len(self.vocab)
        self.mask_id = self.vocab[self.tokenizer.mask_token]
        self.inv_vocab = {k: v for v, k in self.vocab.items()}  # k,v交换的vocab
        self.read_label_names(args.data.DATASET, args.data.LABEL_NAME_FILE)
        self.num_class = len(self.label_name_dict)
        self.model = LOTClassModel.from_pretrained(self.pretrained_lm,
                                                   output_attentions=False,  # 是否返回注意力tensor
                                                   output_hidden_states=False,  # 是否返回所有隐藏层的hidden_states
                                                   num_labels=self.num_class).to(device)
        self.read_data(args.data.DATASET,
                       args.data.TRAIN_CORPUS,
                       args.data.TEST_CORPUS,
                       args.data.TRAIN_LABEL,
                       args.data.TEST_LABEL)
        self.with_test_label = True if args.data.TEST_LABEL is not None else False
        # self.temp_dir = "tmp_{}".format(self.dist_port)
        self.mcp_loss = nn.CrossEntropyLoss()
        self.st_loss = nn.KLDivLoss(reduction='batchmean')
        self.update_interval = args.train_args.update_interval
        self.early_stop = args.train_args.early_stop

    def read_label_names(self, dataset_dir, label_name_file):
        label_name_file = open(os.path.join(dataset_dir, label_name_file), encoding="utf-8")
        label_names = label_name_file.readlines()
        # 读取的label会包含'\n'使用strip()函数去除，并构建label字典，后面会让模型生成与label名类似的词汇，所以字典的value值为一个list对象
        self.label_name_dict = {i: [word.lower().strip() for word in category_words.strip().split()] for
                                i, category_words in enumerate(label_names)}
        LOGS.log.debug("Label names used for each class are:{}".format(self.label_name_dict))
        self.label2class = {}
        self.all_label_name_ids = [self.mask_id]  # 获取所有label在vocab的id
        self.all_label_names = [self.tokenizer.mask_token]  # 获取所有label名

        # 创建一个label2class的词典，k：label v：class
        for class_idx in self.label_name_dict:  # class_idx为class的下标
            for word in self.label_name_dict[class_idx]:
                self.label2class[word] = class_idx
                # 如果label存在于vocab中
                if word in self.vocab:
                    self.all_label_name_ids.append(self.vocab[word])
                    self.all_label_names.append(word)

    def read_data(self, dataset_dir, train_file, test_file, train_label_file, test_label_file):
        self.train_data, self.label_name_data = self.create_dataset(dataset_dir, train_file, train_label_file,
                                                                    "train.pt",
                                                                    find_label_name=True,
                                                                    label_name_loader_name="label_name_data.pt")
        if test_file is not None:
            self.test_data = self.create_dataset(dataset_dir, test_file, test_label_file, "test.pt")

    def create_dataset(self, dataset_dir, text_file, label_file, loader_name, find_label_name=False,
                       label_name_loader_name=None):
        loader_file = os.path.join(dataset_dir, loader_name)

        # 封装所有corpus，如果有label，则一并封装
        if os.path.exists(loader_file):
            LOGS.log.debug("Loading encoded texts from".format(loader_file))
            data = torch.load(loader_file)
        else:
            LOGS.log.debug("Reading texts from {}".format(os.path.join(dataset_dir, text_file)))
            corpus = open(os.path.join(dataset_dir, text_file), encoding="utf-8")
            docs = [doc.strip() for doc in corpus.readlines()]
            LOGS.log.debug("Converting texts into tensors.")

            # 并行化处理，将数据分成多个chunk，每个进程执行一个chunk
            chunk_size = ceil(len(docs) / self.num_cpus)
            chunks = [docs[x: x + chunk_size] for x in range(0, len(docs), chunk_size)]
            results = Parallel(n_jobs=self.num_cpus)(delayed(self.encode)(docs=chunk) for chunk in chunks)

            # 将结果合并
            input_ids = torch.cat([result[0] for result in results])
            attention_masks = torch.cat([result[1] for result in results])
            LOGS.log.debug("Saving encoded texts into:{}".format(loader_file))

            # 封装成字典
            if label_file is not None:
                LOGS.log.debug("Reading labels from ".format(os.path.join(dataset_dir, label_file)))
                truth = open(os.path.join(dataset_dir, label_file))
                labels = [int(label.strip()) for label in truth.readlines()]
                labels = torch.tensor(labels)
                data = {"input_ids": input_ids, "attention_masks": attention_masks, "labels": labels}
            else:
                data = {"input_ids": input_ids, "attention_masks": attention_masks}
            torch.save(data, loader_file)

        # 封装包含label的corpus
        if find_label_name:
            loader_file = os.path.join(dataset_dir, label_name_loader_name)
            if os.path.exists(loader_file):
                LOGS.log.debug("Loading texts with label names from {}".format(loader_file))
                label_name_data = torch.load(loader_file)
            else:
                LOGS.log.debug("Reading texts from {}".format(os.path.join(dataset_dir, text_file)))
                corpus = open(os.path.join(dataset_dir, text_file), encoding="utf-8")
                docs = [doc.strip() for doc in corpus.readlines()]
                LOGS.log.debug("Locating label names in the corpus.")

                # 并行化处理，将数据分成多个chunk，每个进程执行一个chunk
                chunk_size = ceil(len(docs) / self.num_cpus)
                chunks = [docs[x:x + chunk_size] for x in range(0, len(docs), chunk_size)]
                results = Parallel(n_jobs=self.num_cpus)(
                    delayed(self.label_name_occurrence)(docs=chunk) for chunk in chunks)

                # 将结果合并
                input_ids_with_label_name = torch.cat([result[0] for result in results])
                attention_masks_with_label_name = torch.cat([result[1] for result in results])
                label_name_idx = torch.cat([result[2] for result in results])
                assert len(input_ids_with_label_name) > 0, "No label names appear in corpus!"

                # 封装成字典
                label_name_data = {"input_ids": input_ids_with_label_name,
                                   "attention_masks": attention_masks_with_label_name,
                                   "labels": label_name_idx}
                loader_file = os.path.join(dataset_dir, label_name_loader_name)
                LOGS.log.debug("Saving texts with label names into {}".format(loader_file))
                torch.save(label_name_data, loader_file)
            return data, label_name_data
        else:
            return data

    def encode(self, docs):

        """
        返回一个batch的字典:
        [{inputs_ids: ... , token_type_ids: ... , attention_mask: ...},
         {inputs_ids: ... , token_type_ids: ... , attention_mask: ...}, ...]
        """
        encoded_dict = self.tokenizer.batch_encode_plus(docs,
                                                       add_special_tokens=True,  # 是否添加特别Token:[CLS],[SEP]
                                                       max_length=self.max_len,
                                                       padding='max_length',
                                                       pad_to_max_length=True,
                                                       return_attention_mask=True,
                                                       truncation=True,  # 是否截断太长的句子
                                                       return_tensors='pt')
        input_ids = encoded_dict['input_ids']
        attention_masks = encoded_dict['attention_mask']
        return input_ids, attention_masks

    def label_name_occurrence(self, docs):
        text_with_label = []  # 包含label的docs
        label_name_idx = []  # label在每一个doc的位置下标
        for doc in docs:
            result = self.label_name_in_doc(doc)
            if result is not None:
                text_with_label.append(result[0])
                label_name_idx.append(result[1].unsqueeze(0))
        if len(text_with_label) > 0:
            encoded_dict = self.tokenizer.batch_encode_plus(text_with_label,
                                                            add_special_tokens=True,
                                                            max_length=self.max_len,
                                                            pad_to_max_length=True,
                                                            padding='max_length',
                                                            return_attention_mask=True,
                                                            truncation=True,
                                                            return_tensors='pt')
            input_ids_with_label_name = encoded_dict['input_ids']
            attention_masks_with_label_name = encoded_dict['attention_mask']
            label_name_idx = torch.cat(label_name_idx, dim=0)
        else:
            input_ids_with_label_name = torch.ones(0, self.max_len, dtype=torch.long)
            attention_masks_with_label_name = torch.ones(0, self.max_len, dtype=torch.long)
            label_name_idx = torch.ones(0, self.max_len, dtype=torch.long)
        return input_ids_with_label_name, attention_masks_with_label_name, label_name_idx

    def label_name_in_doc(self, doc):
        # 将doc分词
        doc = self.tokenizer.tokenize(doc)
        label_idx = -1 * torch.ones(self.max_len, dtype=torch.long)  # 构建一个全为-1的tensor，标识label在doc的下标
        new_doc = []  # 最终要返回的新doc
        wordpcs = []  # 每次遍历的单词，由于可能有后缀，所以要用list
        idx = 1  # 0为[CLS]
        # 遍历分词后的doc
        for i, wordpc in enumerate(doc):
            wordpcs.append(wordpc[2:] if wordpc.startswith("##") else wordpc)
            if idx >= self.max_len - 1:  # 超过最大长度，直接停止，注意最后一个token一定是[SEP]
                break
            # 如果下一个token是以##开头的，代表下一个token是当前token的后缀需要将两者作为一个单词（word），
            if i == len(doc) - 1 or not doc[i + 1].startswith("##"):
                word = ''.join(wordpcs)
                if word in self.label2class:  # 判断当前word是不是label，如果是就将label_idx中word当前的下标设为其class
                    label_idx[idx] = self.label2class[word]
                    # 如果label没有在词典里，则用[mask]替换它
                    if word not in self.vocab:
                        wordpcs = [self.tokenizer.mask_token]
                new_word = ''.join(wordpcs)
                if new_word != self.tokenizer.unk_token:
                    idx += len(wordpcs)
                    new_doc.append(new_word)
                wordpcs = []
        if (label_idx >= 0).any():  # 判断doc里面是否有label
            return ''.join(new_doc), label_idx
        else:
            return None

    def category_vocabulary(self, top_pred_num=50, category_vocab_size=100, loader_name="category_vocab.pt"):
        loader_file = os.path.join(self.dataset_dir, loader_name)
        if os.path.exists(loader_file):
            LOGS.log.debug("Loading category vocabulary from ".format(loader_file))
            if loader_name[-3:] == ".pt":
                self.category_vocab = torch.load(loader_file)
            else:
                self.category_vocab = {}
                with open(loader_file, mode='r', encoding="utf-8") as wf:
                    for i, line in enumerate(wf.readlines()):
                        words = line.strip().split(' ')
                        token_words = [self.vocab[w] for w in words if w in self.vocab]
                        self.category_vocab[i] = np.array(token_words)
        else:
            LOGS.log.debug("Constructing category vocabulary.")
            # if not os.path.exists(self.temp_dir):
            #     os.makedirs(self.temp_dir)
            model = self.model
            model.eval()
            label_name_dataset_loader = self.make_dataloader(self.label_name_data, self.eval_batch_size)
            self.category_words_freq = {i: defaultdict(float) for i in range(self.num_class)}
            wrap_label_name_dataset_loader = tqdm(label_name_dataset_loader)
            try:
                for batch in wrap_label_name_dataset_loader:
                    with torch.no_grad():
                        input_ids = batch[0].to(device)
                        input_mask = batch[1].to(device)
                        label_pos = batch[2].to(device)
                        match_idx = label_pos >= 0   # 获取label在句子中的具体下标,是一个列表，因为label可能不止一个
                        predictions = model(input_ids,  # 最后一层是线性层，输出维度是batch_size * max_length * vocab_size, 也就是输出vocab中各个token的概率
                                            pred_mode="mlm",
                                            token_type_ids=None,
                                            attention_mask=input_mask)

                        # prediction[match_idx]:提取所有label的概率分布，维度：label_num(对batch里每一个sample里的label数求和)* vocab_size
                        _, sorted_res = torch.topk(predictions[match_idx], top_pred_num, dim=-1)  # 找出可能性最大的top_pred_num个token，维度：label_num * top_pred_num
                        label_idx = label_pos[match_idx]  # label为哪一类 维度：label_num
                        for i, word_list in enumerate(sorted_res):  # word_list:可能性最大的60个token, i：0~label_num-1
                            for j, word_id in enumerate(word_list):  # token在vocab里的id, j:0~top_pred_num-1
                                self.category_words_freq[label_idx[i].item()][word_id.item()] += 1  # 统计所有batch的结果，每一类频率最高的前top_pred_num个，就是该类构建出来的category_vocabulary
            except RuntimeError as err:
                self.cuda_mem_error(err, "eval")
            self.filter_keywords(category_vocab_size)
            torch.save(self.category_vocab, loader_file)
            with open(loader_file.replace('.pt', '.txt'), mode='w', encoding="utf-8") as wf:
                for i, wk in self.category_vocab.items():
                    wk = wk.tolist()
                    wk = [str(self.inv_vocab[w]) for w in wk]
                    wl = ' '.join(wk)
                    wf.write(wl + '\n')

        for i, category_vocab in self.category_vocab.items():
            LOGS.log.debug("Class {} category vocabulary: {}\n".format(self.label_name_dict[i], [self.inv_vocab[w] for w in category_vocab]))

    # 过滤停止词，和出现在多个分类中的词
    def filter_keywords(self, category_vocab_size=256):
        all_words = defaultdict(list)
        sorted_dicts = {}
        # 根据category_vocab_size过滤频率过少的token
        for i, cat_dict in self.category_words_freq.items():
            sorted_dict = {k: v for k, v in
                            sorted(cat_dict.items(), key=lambda item: item[1], reverse=True)[:category_vocab_size]}
            sorted_dicts[i] = sorted_dict
            for word_id in sorted_dict:
                all_words[word_id].append(i)  # 构造all_words字典，k：token在vocab的id, v：class
        repeat_words = []

        # 有多个类的token记录在repeat_words
        for word_id in all_words:
            if len(all_words[word_id]) > 1:
                repeat_words.append(word_id)
        self.category_vocab = {}
        for i, sorted_dict in sorted_dicts.items():
            self.category_vocab[i] = np.array(list(sorted_dict.keys()))
        stopwords_vocab = load_stop_words(os.path.join(self.args.data.DATASET, self.args.data.stop_words))
        for i, word_list in self.category_vocab.items():
            delete_idx = []
            for j, word_id in enumerate(word_list):
                word = self.inv_vocab[word_id]
                if word in self.label_name_dict[i]:
                    continue

                # isalpha：判断是不是只由字母组成
                if not word.isalpha() or len(word) == 1 or word in stopwords_vocab or word_id in repeat_words:
                    delete_idx.append(j)
            self.category_vocab[i] = np.delete(self.category_vocab[i], delete_idx)

    def make_dataloader(self, data_dict, batch_size):
        if "labels" in data_dict:
            dataset = TensorDataset(data_dict["input_ids"], data_dict["attention_masks"], data_dict["labels"])
        else:
            dataset = TensorDataset(data_dict["input_ids"], data_dict["attention_masks"])
        dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return dataset_loader

    def cuda_mem_error(self, err, mode):
        LOGS.log.debug(err)
        if "CUDA out of memory" in str(err):
            if mode == "eval":
                LOGS.log.debug(
                    "Your GPUs can't hold the current batch size for evaluation, try to reduce `--eval_batch_size`, current: {}".format(self.eval_batch_size)
                )
            else:
                LOGS.log.debug(
                    "Your GPUs can't hold the current batch size for training, try to reduce `--train_batch_size`, current: {}".format(self.train_batch_size)
                )
        sys.exit(1)

    def self_train(self, epochs, loader_name="final_model.pt"):
        loader_file = os.path.join(self.dataset_dir, loader_name)
        if os.path.exists(loader_file):
            LOGS.log.debug("\nFinal model {} found, skip self-training".format(loader_file))
        else:
            rand_idx = torch.randperm(len(self.train_data["input_ids"]))
            # 将self.train_data打乱
            self.train_data = {"input_ids": self.train_data["input_ids"][rand_idx],
                               "attention_masks": self.train_data["attention_masks"][rand_idx]}
            LOGS.log.debug("\nStart self-training.")

            test_dataset_loader = self.make_dataloader(self.test_data,
                                                       self.eval_batch_size) if self.with_test_label else None
            total_steps = int(
                len(self.train_data["input_ids"]) * epochs / (self.world_size * self.train_batch_size * self.accum_steps))
            optimizer = AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=1e-6, eps=1e-8)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1 * total_steps, num_training_steps=total_steps)  # 线性预热

            idx = 0
            if self.early_stop:
                agree_count = 0
            for i in range(int(total_steps / self.update_interval)):
                self_train_dict, idx, agree = self.prepare_self_train_data(self.model, idx)

                if self.early_stop:
                    if 1 - agree < 1e-3:
                        agree_count += 1
                    else:
                        agree_count = 0
                    if agree_count >= 3:
                        break
                self_train_dataset_loader = self.make_dataloader(self_train_dict, self.train_batch_size)
                self.self_train_batches(self.model, self_train_dataset_loader, optimizer, scheduler, test_dataset_loader)
            loader_file = os.path.join(self.model_dir, loader_name)
            LOGS.log.debug(f"Saving final model to {loader_file}")
            torch.save(self.model.state_dict(), loader_file)
    #  根据update_interval 准备数据以及目标分布
    def prepare_self_train_data(self, model, idx):
        target_num = min(self.world_size * self.train_batch_size * self.update_interval * self.accum_steps,
                         len(self.train_data["input_ids"]))
        if idx + target_num >= len(self.train_data["input_ids"]):
            select_idx = torch.cat((torch.arange(idx, len(self.train_data["input_ids"])),
                                    torch.arange(idx + target_num - len(self.train_data["input_ids"]))))
        else:
            select_idx = torch.arange(idx, idx + target_num)
        assert len(select_idx) == target_num
        idx = (idx + len(select_idx)) % len(self.train_data["input_ids"])
        select_dataset = {"input_ids": self.train_data["input_ids"][select_idx],
                          "attention_masks": self.train_data["attention_masks"][select_idx]}
        dataset_loader = self.make_dataloader(select_dataset, self.eval_batch_size)
        input_ids, input_mask, all_preds = self.inference(model, dataset_loader, return_type="data")

        # gather_input_ids = [torch.ones_like(input_ids) for _ in range(self.world_size)]
        # gather_input_mask = [torch.ones_like(input_mask) for _ in range(self.world_size)]
        # gather_preds = [torch.ones_like(preds) for _ in range(self.world_size)]
        # dist.all_gather(gather_input_ids, input_ids)
        # dist.all_gather(gather_input_mask, input_mask)
        # dist.all_gather(gather_preds, preds)
        # input_ids = torch.cat(gather_input_ids, dim=0).cpu()
        # input_mask = torch.cat(gather_input_mask, dim=0).cpu()
        # all_preds = torch.cat(gather_preds, dim=0).cpu()

        weight = all_preds ** 2 / torch.sum(all_preds, dim=0)
        target_dist = (weight.t() / torch.sum(weight, dim=1)).t()  # torch.t(): 求转置
        all_target_pred = target_dist.argmax(dim=-1)
        agree = (all_preds.argmax(dim=-1) == all_target_pred).int().sum().item() / len(all_target_pred)  # early-stop的指标
        self_train_dict = {"input_ids": input_ids, "attention_masks": input_mask, "labels": target_dist}
        return self_train_dict, idx, agree

    # 使用模型推理有三种模式：准备数据（目标分布）、计算准确率、预测
    def inference(self, model, dataset_loader, return_type):
        if return_type == "data":
            all_input_ids = []
            all_input_mask = []
            all_preds = []
        elif return_type == "acc":
            pred_labels = []
            truth_labels = []
        elif return_type == "pred":
            pred_labels = []
        model.eval()
        try:
            for batch in dataset_loader:
                with torch.no_grad():
                    input_ids = batch[0].to(device)
                    input_mask = batch[1].to(device)
                    logits = model(input_ids,
                                   pred_mode="classification",
                                   token_type_ids=None,
                                   attention_mask=input_mask)
                    logits = logits[:, 0, :]
                    if return_type == "data":
                        all_input_ids.append(input_ids)
                        all_input_mask.append(input_mask)
                        all_preds.append(nn.Softmax(dim=-1)(logits))
                    elif return_type == "acc":
                        labels = batch[2]
                        pred_labels.append(torch.argmax(logits, dim=-1).cpu())
                        truth_labels.append(labels)
                    elif return_type == "pred":
                        pred_labels.append(torch.argmax(logits, dim=-1).cpu())
            if return_type == "data":
                all_input_ids = torch.cat(all_input_ids, dim=0)
                all_input_mask = torch.cat(all_input_mask, dim=0)
                all_preds = torch.cat(all_preds, dim=0)
                return all_input_ids, all_input_mask, all_preds
            elif return_type == "acc":
                pred_labels = torch.cat(pred_labels, dim=0)
                truth_labels = torch.cat(truth_labels, dim=0)
                samples = len(truth_labels)
                acc = (pred_labels == truth_labels).float().sum() / samples
                return acc.to(device)
            elif return_type == "pred":
                pred_labels = torch.cat(pred_labels, dim=0)
                return pred_labels
        except RuntimeError as err:
            self.cuda_mem_error(err, "eval")

    # 自训练
    def self_train_batches(self, model, self_train_loader, optimizer, scheduler, test_dataset_loader):
        model.train()
        total_train_loss = 0
        wrap_train_dataset_loader = tqdm(self_train_loader)
        model.zero_grad()
        try:
            for j, batch in enumerate(wrap_train_dataset_loader):
                input_ids = batch[0].to(device)
                input_mask = batch[1].to(device)
                target_dist = batch[2].to(device)
                logits = model(input_ids,
                               pred_mode="classification",
                               token_type_ids=None,
                               attention_mask=input_mask)
                logits = logits[:, 0, :]
                preds = nn.LogSoftmax(dim=-1)(logits)
                loss = self.st_loss(preds.view(-1, self.num_class),
                                    target_dist.view(-1, self.num_class)) / self.accum_steps
                total_train_loss += loss.item()
                loss.backward()
                if (j + 1) % self.accum_steps == 0:
                    # Clip the norm of the gradients to 1.0.
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
            if self.with_test_label:
                acc = self.inference(model, test_dataset_loader, return_type="acc")
                # gather_acc = [torch.ones_like(acc) for _ in range(self.world_size)]
                # dist.all_gather(gather_acc, acc)
                # acc = torch.tensor(gather_acc).mean().item()
            avg_train_loss = torch.tensor([total_train_loss / len(wrap_train_dataset_loader) * self.accum_steps]).to(
                device)
            # gather_list = [torch.ones_like(avg_train_loss) for _ in range(self.world_size)]
            # dist.all_gather(gather_list, avg_train_loss)
            # avg_train_loss = torch.tensor(gather_list)

            LOGS.log.debug(f"lr: {optimizer.param_groups[0]['lr']:.4g}")
            LOGS.log.debug(f"Average training loss: {avg_train_loss.item()}")
            if self.with_test_label:
                LOGS.log.debug(f"Test acc: {acc}")
        except RuntimeError as err:
            self.cuda_mem_error(err, "train")

    def write_results(self, loader_name="final_model.pt", out_file="out.txt"):
        loader_file = os.path.join(self.model_dir, loader_name)
        assert os.path.exists(loader_file)
        LOGS.log.debug(f"\nLoading final model from {loader_file}")
        self.model.load_state_dict(torch.load(loader_file))
        # self.model.to(0)
        test_set = TensorDataset(self.test_data["input_ids"], self.test_data["attention_masks"])
        test_dataset_loader = DataLoader(test_set, sampler=SequentialSampler(test_set), batch_size=self.eval_batch_size)
        pred_labels = self.inference(self.model, test_dataset_loader, return_type="pred")
        out_file = os.path.join(self.dataset_dir, out_file)
        LOGS.log.debug(f"Writing prediction results to {out_file}")
        f_out = open(out_file, 'w')
        for label in pred_labels:
            f_out.write(str(label.item()) + '\n')
