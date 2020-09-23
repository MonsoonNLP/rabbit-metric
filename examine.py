import json, random
import nltk
from nltk import tokenize

#####
# Prepare sentences data
#####

nltk.download('punkt')

samples = json.loads(open('./data/sample_sentences.json', 'r').read())

new_words = ['masks']
train_sentences = []
test_sentences = []

# returns only one sentence containing the desired word
def one_sentence(comment, word):
  sentences = tokenize.sent_tokenize(comment.replace('\\', '').replace('*', '').replace('?', '? '))
  for sentence in sentences:
    if word in sentence:
      return sentence.replace(word, '_')

for sample in samples['masks']:
  goto_sentence = one_sentence(sample, 'masks')
  if 'http' in goto_sentence:
    continue
  if random.random() < 0.5:
    train_sentences.append(goto_sentence)
  else:
    test_sentences.append(goto_sentence)

#####
# Loading eligible pre-trained model
#####

import torch
from transformers import BertTokenizer, BertForMaskedLM

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

#####
# Updating tokenizer and model to have capacity for new words
#####
tokenizer.add_tokens(new_words)
model.resize_token_embeddings(len(tokenizer))

#####
# Measuring accuracy of this PR
#####

def fill_in_mask(tokenized_text):
  segments_ids = []
  sentence_index = 0
  for token in tokenized_text:
    segments_ids.append(sentence_index)
    if token == '[SEP]':
      sentence_index += 1
  indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
  # Convert inputs to PyTorch tensors
  tokens_tensor = torch.tensor([indexed_tokens])
  segments_tensors = torch.tensor([segments_ids])
  with torch.no_grad():
      outputs = model(tokens_tensor, token_type_ids=segments_tensors)
      predictions = outputs[0]

  return predictions

for word_test in new_words:
  correct = 0
  index = 0

  # todo: training the model

  for sentence in test_sentences[:500]:
    tagged_text = '[CLS] ' + sentence.replace('_', '[MASK]')
    if tagged_text[-5:] != '[SEP]':
      tagged_text += ' [SEP]'

    tokenized_text = tokenizer.tokenize(tagged_text)
    masked_index = tokenized_text.index('[MASK]')
    predictions = fill_in_mask(tokenized_text)
    predicted_index = torch.argmax(predictions[0, masked_index]).item()
    answer = tokenizer.convert_ids_to_tokens([predicted_index])[0]

    # todo: more options for answers
    index += 1
    if answer == word_test:
      correct += 1
    if index % 50 == 0:
      print(str(correct) + ' / ' + str(index))

print(correct / len(test_sentences))
