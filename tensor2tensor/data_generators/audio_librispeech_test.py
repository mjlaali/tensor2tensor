import tensorflow as tf
from tensor2tensor.data_generators import audio_librispeech
import tempfile


class LibrispeechTest(tf.test.TestCase):
  @staticmethod
  def parametrized_test_for_dir_structure(files, expected):
    is_dataset = audio_librispeech._get_prefix_of_files(files)
    assert is_dataset == expected

  def test_if_valid(self):
    files = ['61-70968-0000.flac', '61-70968-0001.flac', '61-70968.trans.txt']
    self.parametrized_test_for_dir_structure(files, '61-70968')

  def test_if_invalid(self):
    files = []
    self.parametrized_test_for_dir_structure(files, None)
    files = ['61-70968-0000.flac', '61-70968-0001.flac']
    self.parametrized_test_for_dir_structure(files, None)
    files = ['61-70968.trans.txt']
    self.parametrized_test_for_dir_structure(files, None)
    files = ['61-70968-0000.flac', '61-70968-0001.flac', '61-70969.trans.txt']
    self.parametrized_test_for_dir_structure(files, None)

  def test_read_text_files(self):
    content = '61-70968-0000 HE BEGAN A CONFUSED COMPLAINT AGAINST\n' +\
              '61-70968-0001 GIVE NOT SO EARNEST A MIND TO THESE MUMMERIES CHILD'
    tmp_file = tempfile.NamedTemporaryFile('w', delete=False)
    tmp_file.write(content)
    tmp_file.close()

    dic_texts = audio_librispeech._get_text(tmp_file.name)
    assert '61-70968-0000' in dic_texts
    assert '61-70968-0001' in dic_texts
    assert dic_texts['61-70968-0000'] == 'HE BEGAN A CONFUSED COMPLAINT AGAINST'
    assert dic_texts['61-70968-0001'] == 'GIVE NOT SO EARNEST A MIND TO THESE MUMMERIES CHILD'
    tmp_file.delete