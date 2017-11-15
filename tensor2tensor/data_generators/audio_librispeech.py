import os
import soundfile as sf
from tensor2tensor.data_generators import generator_utils
import tensorflow as tf
import tarfile



def _extract_librispeech_to(url, tmp_dir):
  """Extract TIMIT datasets to directory unless directory/timit exists."""

  compressed_filename = os.path.basename(url)
  compressed_filepath = os.path.join(tmp_dir, compressed_filename)
  generator_utils.maybe_download(tmp_dir, compressed_filename, url)

  extracted_dir_path = os.path.join(tmp_dir, 'LibriSpeech', compressed_filename.replace('.tar.gz', ''))
  if not os.path.exists(extracted_dir_path):
    with tf.gfile.GFile(compressed_filepath, 'rb') as f:
      with tarfile.open(fileobj=f, mode="r:gz") as librispeech_compressed:
        librispeech_compressed.extractall(tmp_dir)
  return extracted_dir_path


def _get_audio_data(audio_file):
  np_audio_data, sample_rate = sf.read(audio_file)
  return np_audio_data.tobytes(), len(np_audio_data), np_audio_data.dtype.itemsize, len(np_audio_data.shape)


def _get_text(text_file):
  with open(text_file, 'rt') as f:
    lines = f.readlines()
    audio_texts = dict()
    for line in lines:
      end_of_file_name = line.index(' ')
      audio_file_name = line[:end_of_file_name]
      audio_text = line[end_of_file_name + 1:].strip()
      audio_texts[audio_file_name] = audio_text
  return audio_texts


def _encode_text(text, eos_list):
  encoded_text = [ord(c) for c in text] + eos_list
  return encoded_text


def _get_prefix_of_files(files):
  prefix = None
  contains_txt_file = False
  contains_flac_files = False
  for file_name in files:
    file_prefix = None
    if file_name.endswith('.flac'):
      contains_flac_files = True
      file_prefix = file_name[:file_name.rindex('-')]
    elif file_name.endswith('.trans.txt'):
      contains_txt_file = True
      file_prefix = file_name[:file_name.rindex('.trans.txt')]
    if prefix is None:
      prefix = file_prefix
    elif file_prefix is not None and prefix != file_prefix:
      return None

  if prefix is None:
    return None
  if contains_flac_files and contains_txt_file:
    return prefix


def librispeech_generator(url,
                          tmp_dir,
                          eos_list=None):
  """Data generator for Librispeech transcription problem.

  Args:
    url: url to a dataset
    tmp_dir: path to temporary storage directory.
    eos_list: optional list of end of sentence tokens, otherwise use default
      value `1`.
    vocab_filename: file within `tmp_dir` to read vocabulary from. If this is
      not provided then the target sentence will be encoded by character.
    vocab_size: integer target to generate vocabulary size to.

  Yields:
    A dictionary representing the images with the following fields:
    * inputs: a float sequence containing the audio data
    * audio/channel_count: an integer
    * audio/sample_count: an integer
    * audio/sample_width: an integer
    * targets: an integer sequence representing the encoded sentence
  """
  eos_list = [1] if eos_list is None else eos_list
  dataset_path = _extract_librispeech_to(url, tmp_dir)

  if not os.path.exists(dataset_path):
    raise ValueError('Cannot find the extracted files inside %s!' % dataset_path)
  for root, dirs, files in os.walk(dataset_path):
    prefix = _get_prefix_of_files(files)
    if prefix is not None:
      print('start processing %s ...' % root)
      audio_texts = _get_text(os.path.join(root, prefix + '.trans.txt'))
      for input_file in files:
        if input_file.endswith('.flac'):
          audio_data, sample_count, sample_width, num_channels = _get_audio_data(os.path.join(root, input_file))
          audio_text = audio_texts[input_file[:-len('.flac')]]
          encoded_text = _encode_text(audio_text, eos_list)
          yield {
            "inputs": audio_data,
            "audio/channel_count": [num_channels],
            "audio/sample_count": [sample_count],
            "audio/sample_width": [sample_width],
            "targets": encoded_text
          }
      print('Processing %s done!' % root)


