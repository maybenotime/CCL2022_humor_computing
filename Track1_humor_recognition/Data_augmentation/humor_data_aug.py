import nlpaug.augmenter.word as naw
import json

train_path = './train.json'
back_translation_data_path = './back_translation_en_de.json'

def load_train_data(path):
    with open(path,"r") as f:
        data = json.load(f)
    return data


if __name__ == '__main__':
    back_translation_aug = naw.BackTranslationAug(
    from_model_name='facebook/wmt19-en-de', 
    to_model_name='facebook/wmt19-de-en',
    device = 'cuda',
    )
    original_data = load_train_data(train_path)
    augmented_data = []
    for dialogue in original_data:
        augmented_dialogue = []
        for utter in dialogue:
            text = utter['Sentence']
            augmented = back_translation_aug.augment(text)
            utter['Sentence'] = augmented
            augmented_dialogue.append(utter)
        print(augmented_dialogue)
        augmented_data.append(augmented_dialogue)
    
    with open(back_translation_data_path,"w") as w:
        json.dump(augmented_data,w)
    
    print("aug finish")