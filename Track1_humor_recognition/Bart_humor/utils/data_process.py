import json
import pickle
from tqdm import tqdm


def load_vocab(dataset_name):
    speaker_vocab = pickle.load(open('./data/%s/speaker_vocab.pkl' % (dataset_name), 'rb'))
    label_vocab = pickle.load(open('./data/%s/label_vocab.pkl' % (dataset_name), 'rb'))
    return speaker_vocab, label_vocab


def flat_data(dataset_name, train_dev_pred, with_speaker):
    with open('./data/%s/%s_data.json' % (dataset_name, train_dev_pred), encoding='utf-8') as f:
        data_raw = json.load(f)
    data_raw = sorted(data_raw, key=lambda x: len(x), reverse=True)
    final_json = []
    sentence_count = 0
    sent_with_label = 0
    with open('./data/%s/%s_data_flat.json' % (dataset_name, train_dev_pred), "w") as f:
        for index, context in enumerate(tqdm(data_raw, desc="preprocessing %s dataset" % train_dev_pred)):
            for index, sentence in enumerate(context):
                sentence_count += 1
                if "label" in sentence:
                    if with_speaker:
                        sent_with_speaker = sentence["speaker"] + " : " + sentence["text"]
                    else:
                        sent_with_speaker = sentence["text"]
                    new_sentence = {"text": sent_with_speaker, "speaker": sentence["speaker"], "label": sentence["label"]}
                    sent_with_label += 1
                else:
                    continue
                final_json.append(new_sentence)
        json_data = json.dumps(final_json)
        f.write(json_data)


def split_data(dataset_name, train_dev_pred, with_speaker, with_generation):        #以ENLP为例子
    with open('./data/%s/%s_data.json' % (dataset_name, train_dev_pred), encoding='utf-8') as f:
        data_raw = json.load(f)
    data_raw = sorted(data_raw, key=lambda x: len(x), reverse=True)         #按照Dialogue中的utter数量降序排列
    final_json = []
    with open('./data/%s/%s_data_generation.json' % (dataset_name, train_dev_pred), "w") as f:
        for context_index, context in enumerate(tqdm(data_raw, desc="preprocessing %s dataset" % train_dev_pred)):
            context_len = len(context)      #utter数量
            new_context = []
            index = 0
            while index < context_len:
                sentence = context[index]       #utter
                if 'Label' in sentence:         #有标签
                    if index == context_len - 1:    #最后一个utter时下一句是end
                        next_sentence = 'end'
                    else:
                        if with_speaker:
                            next_sentence = context[index + 1]["Speaker"] + " : " + context[index + 1]["Sentence"]      #拼接上说话人标签
                        else:
                            next_sentence = context[index + 1]["Sentence"]
                    if with_speaker:
                        sent_with_speaker = sentence["Speaker"] + " : " + sentence["Sentence"]
                    else:
                        sent_with_speaker = sentence["Sentence"]
                    # if with_generation:
                    new_sentence = {"text": sent_with_speaker, "speaker": sentence["Speaker"], "label": sentence["Label"],
                                    "next_sentence": next_sentence}         #构造出的新数据包含下一句utter
                    # else:
                    #     new_sentence = {"text": sent_with_speaker, "speaker": sentence["speaker"], "label": sentence["label"]}
                    new_context.append(new_sentence)
                    if len(new_context) == 8:           #八个一组，注意这边的处理我们应该是要修改的
                        final_json.append(new_context)
                        new_context = []
                index += 1
            if len(new_context) > 0:
                final_json.append(new_context)

        json_data = json.dumps(final_json)
        f.write(json_data)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--task_name', type=str, required=True, choices=['MELD', 'IEMOCAP', 'DailyDialog', 'EmoryNLP' , 'humor'])
    parser.add_argument('--train_with_generation', type=int, required=True, default=1, help="1: train with auxiliary generation task, 0: verse vice")
    parser.add_argument('--train_with_speaker', type=int, required=True, default=1, help="1: train with speaker, 0: verse vice")

    args = parser.parse_args()

    print("Start preprocess data")

    split_data(args.task_name, 'train', args.train_with_speaker, args.train_with_generation)
    split_data(args.task_name, 'dev', args.train_with_speaker, args.train_with_generation)
    split_data(args.task_name, 'test', args.train_with_speaker, args.train_with_generation)

    print("Preprocess data complete")
