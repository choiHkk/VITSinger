from tqdm.auto import tqdm
import argparse
import natsort
import random
import json
import glob
import os
import midi_utils


            
def split_trn_val_lines(args):
    note_info = []
    
    val_len = args.val_len
    line_target_dir = args.line_target_dir
    model_name = args.model_name 
    source_dir = os.path.join(args.target_dir, args.data_type)
    assert os.path.isdir(source_dir), print(f"'{source_dir}' is not exist")
    line_paths = natsort.natsorted(glob.glob(os.path.join(source_dir, '*/train.txt')))
    speaker_ids = {}

    os.makedirs(line_target_dir, exist_ok=True)
    tmp = []
    L = len(line_paths)
    for i in tqdm(range(L), total=L):
        line_path = line_paths[i]
        data_name = line_path.split('/')[-2]
        speaker_ids[str(data_name)] = i
        with open(line_path, 'r', encoding='utf8') as f:
            lines = f.read().split('\n')
            lines = [l for l in lines if len(l) > 0]
            for line in lines:
                line = line.split('|')
                line[0] = os.path.join(data_name, 'audio', line[0])
                line.insert(1, str(i))
                note_info.append(list(set(list(map(int, line[-1].split(','))))))
                line = '|'.join(line) + '\n'
                tmp.append(line)
    note_info = list(set(sum(note_info, [])))
    note_min = min(note_info)
    note_max = max(note_info)
    note_min_max = {
        'note_min': int(note_min),
        'note_max': int(note_max)
    }
    with open(os.path.join(line_target_dir, f'note_min_max_{model_name}.json'), 'w', encoding='utf8') as j:
        json.dump(note_min_max, j, ensure_ascii=False, indent=4, sort_keys=False)
    with open(os.path.join(line_target_dir, f'total_speaker_ids_{model_name}.json'), 'w', encoding='utf8') as j:
        json.dump(speaker_ids, j, ensure_ascii=False, indent=4, sort_keys=False)
    random.shuffle(tmp)
    trn_lines = tmp[:-val_len]
    val_lines = tmp[-val_len:]
    random.shuffle(trn_lines)
    random.shuffle(val_lines)

    trn_line_txt = open(os.path.join(line_target_dir, f'{model_name}_train_filelist.txt'), 'w', encoding='utf8')
    for trn_line in trn_lines:
        trn_line_txt.write(trn_line)
    trn_line_txt.close()

    val_line_txt = open(os.path.join(line_target_dir, f'{model_name}_val_filelist.txt'), 'w', encoding='utf8')
    for val_line in val_lines:
        val_line_txt.write(val_line)
    val_line_txt.close()


def main(args):
    data_dir = os.path.join(args.storage_path, args.data_type)
    target_dir = os.path.join(args.target_dir, args.data_type)
    os.makedirs(target_dir, exist_ok=True)
    midi_paths = natsort.natsorted(glob.glob(os.path.join(data_dir, '*/*/*/*/*/*/*/*/*.mid')))
    wav_paths = natsort.natsorted(glob.glob(os.path.join(data_dir, '*/*/*/*/*/*/*/*/*.wav')))
    wav_info_paths = natsort.natsorted(glob.glob(os.path.join(data_dir, '*/*/*/*/*/*/*/*/*.json')))
    
    L = len(midi_paths)
    for i in tqdm(range(L), total=L):
        try:
            midi_utils.preprocess(midi_utils[i], wav_paths[i], wav_info_paths[i], target_dir)
        except:
            print(midi_paths[i])
            
    split_trn_val_lines(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', type=str)
    parser.add_argument('--storage_path', type=str)
    parser.add_argument('--target_dir', type=str)
    parser.add_argument('--line_target_dir', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--val_len', type=int)
    args = parser.parse_args()
    main(args)
    