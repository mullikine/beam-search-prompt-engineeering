
# !pip install transformers
# !pip install torch
# !pip install numpy

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained('gpt2')
model = AutoModelForCausalLM.from_pretrained("gpt2")

# src: https://huggingface.co/transformers/v4.1.1/_modules/transformers/generation_logits_process.html

def set_scores_to_inf_for_banned_tokens(scores, banned_tokens):
    """
    Modifies the scores in place by setting the banned token positions to `-inf`. Banned token is expected to be a
    list of list of banned tokens to ban in tlhe format [[batch index, vocabulary position],...

    Args:
        scores: logits distribution of shape (batch size, vocabulary size)
        banned_tokens: list of list of tokens to ban of length (batch_size)
    """
    banned_mask_list = []
    for idx, batch_banned_tokens in enumerate(banned_tokens):
        for token in batch_banned_tokens:
            banned_mask_list.append([idx, token])
    if not banned_mask_list:
        return scores

    banned_mask = torch.LongTensor(banned_mask_list)
    indices = torch.ones(len(banned_mask))

    banned_mask = (
        torch.sparse.LongTensor(banned_mask.t(), indices, scores.size()).to(scores.device).to_dense().bool()
    )
    scores = scores.masked_fill(banned_mask, -float("inf"))
    return scores

from transformers import LogitsProcessor
import numpy as np

class EvenLogits(LogitsProcessor):
  def __call__(self, input_ids, scores):

    banned_tokens = []
    for beam_index, (beam_input_ids, beam_scores) in enumerate(zip(input_ids, scores)):
      elementwise_length = np.vectorize(len)
      keys = np.array(list(tokenizer.vocab.keys()))
      values = np.array(list(tokenizer.vocab.values()))

      # indexes of tokens that are too long
      indexes = np.where(elementwise_length(keys) % 2 == 0)[0]

      banned_tokens.append(values[indexes])

    scores = set_scores_to_inf_for_banned_tokens(scores, banned_tokens)
    return scores

class ABCLogits(LogitsProcessor):
  def __init__(self, vocab):
    """
    vocab is a dictionary where the keys are tokens
    and the values are the corresponding ids.
    """
    # create an array of tokens
    # remove the 'Ġ' token (used to represent a blank space in the tokenizer)
    self.keys = list(tokenizer.vocab.keys())
    index_to_pop = self.keys.index('Ġ')
    self.keys.pop(index_to_pop)
    self.keys = np.array(self.keys)

    # create an array of ids
    # also remove the 'Ġ' token
    self.values = list(tokenizer.vocab.values())
    self.values.pop(index_to_pop)
    self.values = np.array(self.values)

    # vectorized function used to get the first character of a token
    # ignores leading whitespaces and 'Ġ' tokens
    first_char = lambda x: x.strip('Ġ ')[0].lower()
    self.first_char = np.vectorize(first_char)

    # get the indexes of all IDs that do not start with the given letter
    not_a_indexes = np.where(self.first_char(self.keys) != 'a')
    not_b_indexes = np.where(self.first_char(self.keys) != 'b')
    not_c_indexes = np.where(self.first_char(self.keys) != 'c')

    # create sets of tokens that do not start with 'a', 'b' or 'c'
    self.not_a_values = self.values[not_a_indexes]
    self.not_b_values = self.values[not_b_indexes]
    self.not_c_values = self.values[not_c_indexes]

  def __call__(self, input_ids, scores):
    banned_tokens = []
    # for every beam (partially generated sentence)
    for beam_index, (beam_input_ids, beam_scores) in enumerate(zip(input_ids, scores)):
      # get the last token of this beam
      last_word = tokenizer.decode(beam_input_ids[-1])
      # get the first character of this last token
      starting_char = self.first_char(last_word)
      # if the last token starts with 'a',
      # ban all words that do not start with 'b', etc.
      if starting_char == 'a':
        banned_tokens.append(self.not_b_values)
      elif starting_char == 'b':
        banned_tokens.append(self.not_c_values)
      elif starting_char == 'c':
        banned_tokens.append(self.not_a_values)
      else:
        banned_tokens.append(self.not_a_values)
    # set the scores of all banned tokens over the beams to -inf
    scores = set_scores_to_inf_for_banned_tokens(scores, banned_tokens)
    return scores

from transformers import (
    BeamSearchScorer,
    LogitsProcessorList,
    StoppingCriteriaList,
    MaxLengthCriteria
)
import torch

# how many beams to track during the Viterbi algorithm
num_beams = 10
# how many beams to return after the algorithm
num_return_beams = 10

# the prompt to continue
prompt = 'My cute dog is a'

# tokenizing the prompt
prompt_tokenized = tokenizer(prompt, return_tensors='pt' )
prompt_tokenized = prompt_tokenized['input_ids']

# instantiating a BeamSearchScorer
beam_scorer = BeamSearchScorer(
    batch_size = prompt_tokenized.shape[0],
    num_beams = num_beams,
    num_beam_hyps_to_keep = num_return_beams,
    device=model.device
)

# instantiating a list of LogitsProcessor instances
# using our custom ABCLogits class
logits_processor = LogitsProcessorList([ABCLogits(tokenizer.vocab)])

# running beam search using our custom LogitsProcessor
generated = model.beam_search(
    torch.cat([prompt_tokenized] * num_beams),
    beam_scorer,
    logits_processor = logits_processor,
    stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=12)])
)

# printing the output beams
for index, output_tokenized in enumerate(generated):
  output = tokenizer.decode(output_tokenized)
  print(f'beam {index}: {output}')

# how many beams to track during the Viterbi algorithm
num_beams = 10
# how many beams to return after the algorithm
num_return_beams = 10

# the prompt to continue
prompt = 'My cute dog is a'

# tokenizing the prompt
prompt_tokenized = tokenizer(prompt, return_tensors='pt' )
prompt_tokenized = prompt_tokenized['input_ids']

# instantiating a BeamSearchScorer
beam_scorer = BeamSearchScorer(
    batch_size = prompt_tokenized.shape[0],
    num_beams = num_beams,
    num_beam_hyps_to_keep = num_return_beams,
    device=model.device
)

# instantiating a list of LogitsProcessor instances
# using our custom ABCLogits and EvenLogits class
# logits_processor = LogitsProcessorList([ABCLogits(tokenizer.vocab), EvenLogits()])
logits_processor = LogitsProcessorList([EvenLogits(), ABCLogits(tokenizer.vocab)])

# running beam search using our custom LogitsProcessor
generated = model.beam_search(
    torch.cat([prompt_tokenized] * num_beams),
    beam_scorer,
    logits_processor = logits_processor,
    stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=12)])
)

# printing the output beams
for index, output_tokenized in enumerate(generated):
  output = tokenizer.decode(output_tokenized)
  print(f'beam {index}: {output}')
