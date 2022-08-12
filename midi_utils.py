# !pip install mido
# !apt-get install libasound2-dev libjack0 libjack-dev
# !pip install python-rtmidi
# !pip install pretty_midi
# !pip install unihandecode
# !pip install g2pk
from unihandecode import Unihandecoder
from scipy.io import wavfile 
from g2pk import G2p
import numpy as np
import shutil
import json
import mido
import math
import sys
import re
import os
decoder = Unihandecoder()
g2p = G2p()

# support vits only

def get_music(midi_Path, charset='CP949'):
    mid = mido.MidiFile(midi_Path, charset=charset)
    music = []
    current_Lyric = ''
    current_Note = None
    current_Time = 0.0

    # Note on 쉼표
    # From Lyric to message before note on: real note
    for message in list(mid):
        if message.type == 'note_on':
            if message.time < 0.1:
                current_Time += message.time
                if current_Lyric in ['J', 'H', None]:
                    music.append((current_Time, '<X>', 0))
                else:
                    music.append((current_Time, current_Lyric, current_Note))
            else:
                if not current_Lyric in ['J', 'H', None]:
                    music.append((current_Time, current_Lyric, current_Note))
                else:
                    message.time += current_Time
                music.append((message.time, '<X>', 0))
            current_Time = 0.0
            current_Lyric = ''
            current_Note = None
        elif message.type == 'lyrics':
            current_Lyric = message.text.strip()
            current_Time += message.time
        elif message.type == 'note_off':
            current_Note = message.note
            current_Time += message.time
            if current_Lyric == 'H':    # it is temp.
                break
        else:
            current_Time += message.time

    if current_Lyric in ['J', 'H']:
        if music[-1][1] == '<X>':
            music[-1] = (music[-1][0] + current_Time, music[-1][1], music[-1][2])
        else:
            music.append((current_Time, '<X>', 0))
    else:
        music.append((current_Time, current_Lyric, current_Note))
    music = music[1:]
    return music


def get_line_note_duration(music, debug=False, spliter='|'):
    times = [v[0] for v in music[1:-1] if not v[1] == '<X>']
    times = np.array(times)
    mu = times.mean()

    switch = 0
    T = 0
    lines = []
    durations = []
    for v in music:
        d, t, n = v
        T += d
        if t == '<X>':
            if switch == 1:
                line = line[:-1]
                end = line.replace('\n', '').split('-')[-1]
                end = re.sub('[{}]', '', end).split(spliter)[0]
                if end == ' ':
                    line = line.split('-')[:-1]
                    line = '-'.join(line)
                lines.append(line+'\n')
                durations.append(T)
                if debug:
                    print(T, line)
                switch = 0
            line = ''
        else:
            switch = 1
            t = "{" + t + spliter + str(n) + "}-"
            if d > mu:
                s = "{" + ' ' + spliter + str(0) + "}-"
                t = f"{t}{s}"
            line += t

    assert len(lines) == len(durations)
    
    return lines, durations
    
    
def run_text_note_separate(lines, g2p_module=None, decode_module=None, flatten=True, debug=False, spliter='|'):
    if g2p_module is None:
        g2p_module = g2p
    if decode_module is None:
        decode_module = decoder
    # texts, notes = separate(lines, g2p, decoder, flatten=True)
    origin_texts, texts, notes = [], [], []
    for line in lines:
        line = line.replace('\n', '').split('-')
        line = [re.sub('[{}]', '', l) for l in line]
        text = ''
        note = []
        L = len(line)
        for i in range(L):
            text += f"{line[i].split(spliter)[0]}"
            note.append([line[i].split(spliter)[1]])
        origin_text = text
        if g2p_module is not None:
            text = g2p(text)
        text = f'{spliter}'.join(list(text)).strip()
        if decode_module is not None:
            text = decoder.decode(text)
        text = text.split(f'{spliter}')

        assert len(text) == len(note), print(text, note)
        L = len(text)
        for i in range(L):
            note[i] = note[i]*len(text[i])

        text = f'{spliter}'.join(text)
        if flatten:
            text = ''.join(text.split(spliter))
            note = sum(note, [])
            assert len(text) == len(note)
        origin_texts.append(origin_text.strip())
        texts.append(text.strip())
        notes.append(note)
    
    assert len(texts) == len(notes)
    if debug:
        debug_outputs = []
        L = len(texts)
        for i in range(L):
            debug_outputs.append((origin_texts[i], texts[i], notes[i]))
        return debug_outputs
    
    return origin_texts, texts, notes


def run_audio_separate(wav_path, wav_info_path, data_dir, durations, debug_outputs):
    with open(wav_info_path, 'r', encoding='utf8') as j:
        wav_info = json.loads(j.read())
        assert wav_info['기본정보']['Language'] == 'KOR'
        wav_genre = wav_info['음악정보']['SongGenre']
        wav_name = wav_info['전사정보']['LabelText']
        speaker_name = wav_info['화자정보']['SpeakerName']
        speaker_age = wav_info['화자정보']['Age']
        speaker_gender = wav_info['화자정보']['Gender']
        speaker_genre = wav_info['화자정보']['Genre']
        filename = wav_info['파일정보']['FileName']
    
    data_dir = os.path.join(data_dir, filename)

    
    os.makedirs(data_dir, exist_ok=True)

    basename = f"ko-{wav_genre}-{wav_name}-{speaker_name}-{speaker_age}-{speaker_gender}-{speaker_genre}"
    basename = basename.replace(' ', '_')
    sampling_rate, audio = wavfile.read(wav_path)
    lyrics = [v[0] for v in debug_outputs]
    texts = [v[1] for v in debug_outputs]
    notes = [','.join(v[2]) for v in debug_outputs]

    T, S = 0, 0
    lines = open(os.path.join(data_dir, 'train.txt'), 'w', encoding='utf8')
    for i, duration in enumerate(durations):
        current_duration = duration - T
        current_shape = math.floor(current_duration * sampling_rate)

        current_audio = audio[S:S+current_shape]
        current_audio_basename = f"{basename}-{str(i).zfill(5)}.wav"
        target_wav_sub_dir = os.path.join(data_dir, 'audio')
        os.makedirs(target_wav_sub_dir, exist_ok=True)
        target_wav_path = os.path.join(target_wav_sub_dir, current_audio_basename)
        
        wavfile.write(target_wav_path, sampling_rate, current_audio.astype(np.int16))

        line = f'{current_audio_basename}|{lyrics[i]}|{texts[i]}|{notes[i]}\n'
        lines.write(line)

        T += current_duration
        S += current_audio.shape[0]

    lines.close()
    shutil.copy2(wav_path, os.path.join(data_dir, wav_path.split('/')[-1]))
    shutil.copy2(wav_info_path, os.path.join(data_dir, wav_info_path.split('/')[-1]))
    print(f"'{wav_name}' shape miss match: {S / audio.shape[0]}")


def preprocess(midi_path, wav_path, wav_info_path, data_dir, *args):
    music = get_music(midi_path)
    lines, durations = get_line_note_duration(music)
    origin_texts, texts, notes = run_text_note_separate(lines, flatten=True)
    debug_outputs = run_text_note_separate(lines, flatten=True, debug=True)
    run_audio_separate(wav_path, wav_info_path, data_dir, durations, debug_outputs)
    

if __name__ == '__main__':
    import natsort
    from tqdm.auto import tqdm
    import glob
    import os

    nas_path = '/home/choihk/sshfs/nas2/homes/ailab/aihub'
    data_type = '004.MultiSpeakerSingingVoiceVocalData'
    target_dir = f'/home/choihk/sshfs/nas2/homes/ailab/dataset/singing/{data_type}'
    os.makedirs(target_dir, exist_ok=True)

    data_dir = os.path.join(nas_path, data_type)

    midi_paths = natsort.natsorted(glob.glob(os.path.join(data_dir, '*/*/*/*/*/*/*/*/*.mid')))
    wav_paths = natsort.natsorted(glob.glob(os.path.join(data_dir, '*/*/*/*/*/*/*/*/*.wav')))
    wav_info_paths = natsort.natsorted(glob.glob(os.path.join(data_dir, '*/*/*/*/*/*/*/*/*.json')))
    
    L = len(midi_paths)

    for i in tqdm(range(L), total=L):
        try:
            preprocess(midi_paths[i], wav_paths[i], wav_info_paths[i], target_dir)
        except:
            print(midi_paths[i])
