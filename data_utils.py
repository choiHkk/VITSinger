import os
import random
import torch
import torch.utils.data
from tqdm.auto import tqdm

import commons 
from mel_processing import spectrogram_torch, dynamic_range_compression_torch
from utils import load_wav_to_torch, load_filepaths_and_text
from text import text_to_sequence#, cleaned_text_to_sequence
from yin import pitch_calc



"""Multi speaker version"""
class TextAudioSpeakerLoader(torch.utils.data.Dataset):
    """
        1) loads audio, speaker_id, text pairs
        2) normalizes text and converts them to sequences of integers
        3) computes spectrograms from audio files.
    """
    def __init__(self, audiopaths_sid_text, hparams):
        self.audiopaths_sid_text = load_filepaths_and_text(audiopaths_sid_text)
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length  = hparams.filter_length
        self.hop_length     = hparams.hop_length
        self.win_length     = hparams.win_length
        self.sampling_rate  = hparams.sampling_rate
        self.f0_min         = hparams.f0_min
        self.f0_max         = hparams.f0_max
        self.data_dir       = hparams.data_dir
        self.confidence_threshold = hparams.confidence_threshold
        self.gaussian_smoothing_sigma = hparams.gaussian_smoothing_sigma
        
        self.add_blank = hparams.add_blank
        self.min_text_len = getattr(hparams, "min_text_len", 5)
        self.max_text_len = getattr(hparams, "max_text_len", 190)

        random.seed(1234)
        random.shuffle(self.audiopaths_sid_text)
        self._filter()

    def _filter(self):
        print('filtering...')
        """
        Filter text & store spec lengths
        """
        audiopaths_sid_text_new = []
        lengths = []

        for audiopath, sid, origin_text, text, note in tqdm(self.audiopaths_sid_text):
            audiopath = os.path.join(self.data_dir, audiopath)
            if os.path.getsize(audiopath) < self.sampling_rate:
                # print(audiopath, os.path.getsize(audiopath))
                continue
            if self.min_text_len <= len(text) and len(text) <= self.max_text_len:
                audiopaths_sid_text_new.append([audiopath, sid, origin_text, text, note])
                lengths.append(os.path.getsize(audiopath) // (2 * self.hop_length))
                
        self.audiopaths_sid_text = audiopaths_sid_text_new
        self.lengths = lengths
        print(len(self.lengths))

    def get_audio_text_speaker_pair(self, audiopath_sid_text):
        # separate filename, speaker_id and text
        audiopath, sid, origin_text, text, note = (
            audiopath_sid_text[0], audiopath_sid_text[1], audiopath_sid_text[2], 
            audiopath_sid_text[3], audiopath_sid_text[4])
        text = self.get_text(text)
        spec, wav, pitch, silence = self.get_audio(audiopath)
        sid = self.get_sid(sid)
        note = self.get_note(note)
        return (text, spec, wav, pitch, silence, sid, note)

    def get_audio(self, filename):
        audio, sampling_rate = load_wav_to_torch(filename, self.sampling_rate)
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} {} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        pitch = self.get_pitch(audio_norm)
        spec = spectrogram_torch(audio_norm, self.filter_length,
            self.sampling_rate, self.hop_length, self.win_length,
            center=False)
        spec = torch.squeeze(spec, 0)
        silence = self.get_silence(spec)
        pitch = pitch[:spec.size(-1)]
        silence = silence[:spec.size(-1)]
        return spec, audio_norm, pitch, silence

    def get_text(self, text):
        text = text_to_sequence(text)
        if self.add_blank:
            text = commons.intersperse(text, 0)
        text = torch.LongTensor(text)
        return text

    def get_sid(self, sid):
        sid = torch.LongTensor([int(sid)])
        return sid
    
    def get_note(self, note):
        if self.add_blank:
            note = commons.intersperse(note, 0)
        note = torch.LongTensor(note)
        return note
    
    def get_pitch(self, audio): # [B, T]
        pitch = pitch_calc(
            sig=audio.squeeze(0), 
            sr=self.sampling_rate, 
            w_len=self.win_length, 
            w_step=self.hop_length, 
            f0_min=self.f0_min, 
            f0_max=self.f0_max, 
            confidence_threshold=self.confidence_threshold, 
            gaussian_smoothing_sigma=self.gaussian_smoothing_sigma
        ) / self.f0_max # [B, F]
        pitch = torch.from_numpy(pitch)
        return pitch
    
    def get_silence(self, spec, min_db=-4.0):
        spec_log_mean = dynamic_range_compression_torch(spec).mean(dim=0)
        silence = torch.where(
            spec_log_mean < min_db, 
            torch.zeros_like(spec_log_mean), 
            torch.ones_like(spec_log_mean))
        return silence

    def __getitem__(self, index):
        return self.get_audio_text_speaker_pair(self.audiopaths_sid_text[index])

    def __len__(self):
        return len(self.audiopaths_sid_text)
    
    
class TestTextAudioSpeakerLoader(torch.utils.data.Dataset):
    """
        1) loads audio, speaker_id, text pairs
        2) normalizes text and converts them to sequences of integers
        3) computes spectrograms from audio files.
    """
    def __init__(self, sid, audiopaths_sid_text, hparams):
        self.audiopaths_sid_text = load_filepaths_and_text(audiopaths_sid_text)
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length  = hparams.filter_length
        self.hop_length     = hparams.hop_length
        self.win_length     = hparams.win_length
        self.sampling_rate  = hparams.sampling_rate
        self.f0_min         = hparams.f0_min
        self.f0_max         = hparams.f0_max
        self.data_dir       = hparams.data_dir
        self.confidence_threshold = hparams.confidence_threshold
        self.gaussian_smoothing_sigma = hparams.gaussian_smoothing_sigma
        self.sid = sid
        
        self.add_blank = hparams.add_blank
        self.min_text_len = getattr(hparams, "min_text_len", 5)
        self.max_text_len = getattr(hparams, "max_text_len", 190)
        self._filter()

    def _filter(self):
        print('filtering...')
        """
        Filter text & store spec lengths
        """
        audiopaths_sid_text_new = []
        lengths = []

        for audiopath, origin_text, text, note in tqdm(self.audiopaths_sid_text):
            audiopath = os.path.join(self.data_dir, audiopath)
            if os.path.getsize(audiopath) < self.sampling_rate:
                # print(audiopath, os.path.getsize(audiopath))
                continue
            if self.min_text_len <= len(text) and len(text) <= self.max_text_len:
                audiopaths_sid_text_new.append([audiopath, self.sid, origin_text, text, note])
                lengths.append(os.path.getsize(audiopath) // (2 * self.hop_length))
                
        self.audiopaths_sid_text = audiopaths_sid_text_new
        self.lengths = lengths
        print(len(self.lengths))

    def get_audio_text_speaker_pair(self, audiopath_sid_text):
        # separate filename, speaker_id and text
        audiopath, sid, origin_text, text, note = (
            audiopath_sid_text[0], audiopath_sid_text[1], audiopath_sid_text[2], 
            audiopath_sid_text[3], audiopath_sid_text[4])
        text = self.get_text(text)
        spec, wav, pitch, silence = self.get_audio(audiopath)
        sid = self.get_sid(sid)
        note = self.get_note(note)
        return (text, spec, wav, pitch, silence, sid, note)

    def get_audio(self, filename):
        audio, sampling_rate = load_wav_to_torch(filename, self.sampling_rate)
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} {} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        pitch = self.get_pitch(audio_norm)
        spec = spectrogram_torch(audio_norm, self.filter_length,
            self.sampling_rate, self.hop_length, self.win_length,
            center=False)
        spec = torch.squeeze(spec, 0)
        silence = self.get_silence(spec)
        pitch = pitch[:spec.size(-1)]
        silence = silence[:spec.size(-1)]
        return spec, audio_norm, pitch, silence

    def get_text(self, text):
        text = text_to_sequence(text)
        if self.add_blank:
            text = commons.intersperse(text, 0)
        text = torch.LongTensor(text)
        return text

    def get_sid(self, sid):
        sid = torch.LongTensor([int(sid)])
        return sid
    
    def get_note(self, note):
        if self.add_blank:
            note = commons.intersperse(note, 0)
        note = torch.LongTensor(note)
        return note
    
    def get_pitch(self, audio): # [B, T]
        pitch = pitch_calc(
            sig=audio.squeeze(0), 
            sr=self.sampling_rate, 
            w_len=self.win_length, 
            w_step=self.hop_length, 
            f0_min=self.f0_min, 
            f0_max=self.f0_max, 
            confidence_threshold=self.confidence_threshold, 
            gaussian_smoothing_sigma=self.gaussian_smoothing_sigma
        ) / self.f0_max # [B, F]
        pitch = torch.from_numpy(pitch)
        return pitch
    
    def get_silence(self, spec, min_db=-4.0):
        spec_log_mean = dynamic_range_compression_torch(spec).mean(dim=0)
        silence = torch.where(
            spec_log_mean < min_db, 
            torch.zeros_like(spec_log_mean), 
            torch.ones_like(spec_log_mean))
        return silence

    def __getitem__(self, index):
        return self.get_audio_text_speaker_pair(self.audiopaths_sid_text[index])

    def __len__(self):
        return len(self.audiopaths_sid_text)
    

class TextAudioSpeakerCollate():
    """ Zero-pads model inputs and targets
    """
    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        """Collate's training batch from normalized text, audio and speaker identities
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized, sid]
        """
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[1].size(1) for x in batch]),
            dim=0, descending=True)

        max_text_len = max([len(x[0]) for x in batch])
        max_spec_len = max([x[1].size(1) for x in batch])
        max_wav_len = max([x[2].size(1) for x in batch])

        text_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))
        sid = torch.LongTensor(len(batch))

        text_padded = torch.LongTensor(len(batch), max_text_len)
        note_padded = torch.LongTensor(len(batch), max_text_len)
        spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        pitch_padded = torch.FloatTensor(len(batch), max_spec_len)
        silence_padded = torch.LongTensor(len(batch), max_spec_len)
        text_padded.zero_()
        note_padded.zero_()
        spec_padded.zero_()
        wav_padded.zero_()
        pitch_padded.zero_()
        silence_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            text = row[0]
            text_padded[i, :text.size(0)] = text
            text_lengths[i] = text.size(0)

            spec = row[1]
            spec_padded[i, :, :spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = row[2]
            wav_padded[i, :, :wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)
            
            pitch = row[3]
            pitch_padded[i, :pitch.size(0)] = pitch
            
            silence = row[4]
            silence_padded[i, :silence.size(0)] = silence

            sid[i] = row[5]
            
            note = row[6]
            note_padded[i, :note.size(0)] = note

        return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, pitch_padded, silence_padded, sid, note_padded


class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.
  
    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """
    def __init__(self, dataset, batch_size, boundaries, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries
  
        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas
  
    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)
  
        for i in range(len(buckets) - 1, 0, -1):
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i+1)
  
        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (total_batch_size - (len_bucket % total_batch_size)) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket
  
    def __iter__(self):
      # deterministically shuffle based on epoch
      g = torch.Generator()
      g.manual_seed(self.epoch)
  
      indices = []
      if self.shuffle:
          for bucket in self.buckets:
              indices.append(torch.randperm(len(bucket), generator=g).tolist())
      else:
          for bucket in self.buckets:
              indices.append(list(range(len(bucket))))
  
      batches = []
      for i in range(len(self.buckets)):
          bucket = self.buckets[i]
          len_bucket = len(bucket)
          ids_bucket = indices[i]
          num_samples_bucket = self.num_samples_per_bucket[i]
  
          # add extra samples to make it evenly divisible
          rem = num_samples_bucket - len_bucket
          ids_bucket = ids_bucket + ids_bucket * (rem // len_bucket) + ids_bucket[:(rem % len_bucket)]
  
          # subsample
          ids_bucket = ids_bucket[self.rank::self.num_replicas]
  
          # batching
          for j in range(len(ids_bucket) // self.batch_size):
              batch = [bucket[idx] for idx in ids_bucket[j*self.batch_size:(j+1)*self.batch_size]]
              batches.append(batch)
  
      if self.shuffle:
          batch_ids = torch.randperm(len(batches), generator=g).tolist()
          batches = [batches[i] for i in batch_ids]
      self.batches = batches
  
      assert len(self.batches) * self.batch_size == self.num_samples
      return iter(self.batches)
  
    def _bisect(self, x, lo=0, hi=None):
      if hi is None:
          hi = len(self.boundaries) - 1
  
      if hi > lo:
          mid = (hi + lo) // 2
          if self.boundaries[mid] < x and x <= self.boundaries[mid+1]:
              return mid
          elif x <= self.boundaries[mid]:
              return self._bisect(x, lo, mid)
          else:
              return self._bisect(x, mid + 1, hi)
      else:
          return -1

    def __len__(self):
        return self.num_samples // self.batch_size
