from typing import Union
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import multiprocessing
from transformers import AdamW, get_linear_schedule_with_warmup
import copy
from tqdm import tqdm
from dask import delayed
from sklearn.metrics import confusion_matrix

class DisasterTweetsDataset(Dataset):
    def __init__(self, tweets_df: pd.DataFrame, text_column: str, label_column: str = None) -> None:
        super().__init__()
        self.tweets_df = tweets_df
        self.text_column = text_column
        self.label_column = label_column

    def __len__(self) -> int:
        return self.tweets_df.shape[0]

    def __getitem__(self, idx: int):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.label_column:
            df_row = self.tweets_df.loc[idx, [self.text_column, self.label_column]]
            sample = {'text': df_row[self.text_column], 'label': df_row[self.label_column]}
        else:
            df_row = self.tweets_df.loc[idx, [self.text_column]]
            sample = {'text': df_row[self.text_column]}
        return sample

hf_weights_name = 'roberta-large'
hf_tokenizer = AutoTokenizer.from_pretrained(hf_weights_name,
                                             cache_dir="/Users/bashleig/PycharmProjects/text_clsf_baseline/cache_hf")

def collate_fn(batch):
    if 'label' in batch[0]:
        texts, labels = zip(*[(batch[i]['text'], batch[i]['label']) for i in range(len(batch))])
        result = dict(labels=labels)
    else:
        texts = [batch[i]['text'] for i in range(len(batch))]
        result = {}
    hf_example_ids = hf_tokenizer.batch_encode_plus(list(texts),
                                                    add_special_tokens=True,
                                                    return_attention_mask=True,
                                                    padding='longest')
    return dict(**result, **hf_example_ids)



def add_bert_to_graph(logger, overall_dict):
    for preprocess_type in overall_dict:
        overall_dict[preprocess_type] = {'bert': {'bert': run_bert(logger, preprocess_type)}}

@delayed
def run_bert(logger, preprocess_type):
    logger.info("bert block stared")
    train_df = pd.read_csv('data/03_primary/{}_train.csv'.format(preprocess_type), index_col=0)
    val_df = pd.read_csv('data/03_primary/{}_val.csv'.format(preprocess_type), index_col=0)
    train_indices = train_df.index
    val_indices = val_df.index
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    train_dataset = DisasterTweetsDataset(train_df, 'text', 'label')
    val_dataset = DisasterTweetsDataset(val_df, 'text', 'label')
    hf_weights_name = 'roberta-large'
    # Create tokenizer from pretrained weights


    num_workers = multiprocessing.cpu_count()
    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn,
                              sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn,
                            sampler=val_sampler)

    logger.info("len of train_loader: {}".format(len(train_loader)))
    logger.info("len of train_loader: {}".format(len(val_loader)))

    data_loaders = {'train': train_loader, 'val': val_loader}
    progress_bars = {}
    epoch_stats = {}

    gpu_available = torch.cuda.is_available()
    logger.info("GPU is available: {}".format(gpu_available))
    device = torch.device('cuda' if gpu_available else 'cpu')
    logger.info("Device: {}".format(device))

    if gpu_available:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print("Cuda Device Name:", torch.cuda.get_device_name())

    model = AutoModelForSequenceClassification.from_pretrained(hf_weights_name,
                                                               num_labels=2,
                                                               cache_dir='/Users/bashleig/PycharmProjects/text_clsf_baseline/cache_hf')
    model.to(device)

    num_epochs = 1
    verbose = True

    optimizer = AdamW(model.parameters(), lr=1.5e-6, eps=1e-8)

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=len(train_loader) * num_epochs)

    best_acc = 0.0
    best_loss = float('inf')

    # Weights of best model so far
    best_model_weights = copy.deepcopy(model.state_dict())
    epoch_bar = tqdm(desc='training routine', total=num_epochs,
                     initial=0, position=0, disable=(verbose is not True))
    for split, data_loader in data_loaders.items():
        progress_bars[split] = tqdm(desc=f'split={split}',
                                    total=len(data_loader),
                                    position=1,
                                    disable=(verbose is not True),
                                    leave=True)
        epoch_stats[split] = {'loss': [], 'accuracy': []}

    best_f1 = 0

    for epoch in range(1, num_epochs + 1):
        for split, data_loader in data_loaders.items():
            training_data = []
            tp_total = 0
            fp_total = 0
            fn_total = 0

            epoch_loss = torch.FloatTensor([0.0]).to(device)
            num_correct = torch.LongTensor([0]).to(device)
            total_samples = 0
            is_training = (split == 'train')
            model.train(is_training)
            for batch in data_loader:
                # pass
                with torch.set_grad_enabled(is_training):
                    input_ids = torch.LongTensor(batch['input_ids']).to(device)
                    labels = torch.LongTensor(batch['labels']).to(device)
                    masks = torch.LongTensor(batch['attention_mask']).to(device)

                    optimizer.zero_grad()

                    outputs = model(input_ids, masks, labels=labels)
                    loss = outputs.loss

                    if is_training:
                        loss.backward()
                    epoch_loss += loss
                    _, predictions = torch.max(outputs.logits, 1)
                    num_correct += torch.eq(predictions, labels).sum()
                    total_samples += labels.size(0)
                    # print(confusion_matrix(predictions.cpu().numpy(), labels.cpu().numpy()).ravel())
                    conf_matr = confusion_matrix(predictions.cpu().numpy(), labels.cpu().numpy()).ravel()
                    if len(conf_matr) == 1:
                        tp = conf_matr[0]
                        fp = 0
                        fn = 0
                    else:
                        tn, fp, fn, tp = confusion_matrix(predictions.cpu().numpy(), labels.cpu().numpy()).ravel()
                    tp_total += tp
                    fp_total += fp
                    fn_total += fn

                    if is_training:
                        optimizer.step()
                        scheduler.step()
                    progress_bars[split].update()
            epoch_loss /= len(data_loader)
            epoch_accuracy = num_correct / total_samples

            epoch_prc = tp_total / (tp_total + fp_total)
            epoch_rec = tp_total / (tp_total + fn_total)
            epoch_f1 = (2 * epoch_rec * epoch_prc) / (epoch_rec + epoch_prc)

            epoch_bar.set_postfix(
                {f"{split}_loss": epoch_loss.item(), f"{split}_acc": round(epoch_accuracy.item(), 3)})
            if not is_training:
                training_data.append((epoch_loss.item(), round(epoch_accuracy.item(), 3)))
                # if epoch_accuracy.item() > best_acc:
                #     best_model_weights = copy.deepcopy(model.state_dict())
                #     best_acc = epoch_accuracy.item()
                if epoch_f1 > best_f1:
                    best_model_weights = copy.deepcopy(model.state_dict())
                    best_f1 = epoch_f1

        for bar in progress_bars.values():
            bar.n = 0
            bar.reset()
        epoch_bar.update()
    return best_f1