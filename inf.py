import pandas as pd
import ast
from InstructABSA.data_prep import DatasetLoader
from InstructABSA.utils import T5Generator, T5Classifier
from instructions import InstructionsHandler
import time
import os

def get_all_filepaths(directory):

  filepaths = []
  for root, _, files in os.walk(directory):
    for file in files:
      filepath = os.path.join(root, file)
      filepaths.append(filepath)

  return filepaths


directory = "../sentiment_score_full_review"
filepaths = get_all_filepaths(directory)

print(filepaths)

# start_time = time.time()

def pre_data(df):
  df['text'] = df['combined_review'].apply(lambda x: bos_instruction + x + eos_instruction)
  return df

for path in filepaths:
  start_time = time.time()
  out_path = 'model' +  path.split('/')[-1][:-4] 
  print("***************************")
  print(out_path)
  df1 = pd.read_csv(path)



  bos_instruction = """Definition: The output will be the aspects (both implicit and explicit) and the aspects sentiment polarity. In cases where there are no aspects the output should be noaspectterm:none.
  Positive example 1-
  input: I charge it at night and skip taking the cord with me because of the good battery life.
  output: battery life:positive,
  Positive example 2-
  input: I even got my teenage son one, because of the features that it offers, like, iChat, Photobooth, garage band and more!.
  output: features:positive, iChat:positive, Photobooth:positive, garage band:positive
  Now complete the following example-
  input: """

  eos_instruction = ' \noutput:'

  id_tr_df = df1.copy(deep=True)
  id_te_df = df1.copy(deep=True)

  #id_tr_df = id_tr_df
  #id_te_df = id_te_df

  instruct_handler = InstructionsHandler()

  instruct_handler.load_instruction_set1()

  loader = DatasetLoader(id_tr_df, id_te_df)
  if loader.train_df_id is not None:
      loader.train_df_id = pre_data(loader.train_df_id)

  if loader.test_df_id is not None:
      loader.test_df_id = pre_data(loader.test_df_id)

  root_path = '../model'

  t5_exp = T5Generator(root_path)

  id_ds, id_tokenized_ds, ood_ds, ood_tokenzed_ds = loader.set_data_for_training_semeval(t5_exp.tokenize_function_inputs)


  id_tr_pred_labels = t5_exp.get_labels(tokenized_dataset = id_tokenized_ds, sample_set = 'train', batch_size = 16)

  loader.train_df_id['prediction_by_model'] =  id_tr_pred_labels
  del loader.train_df_id['text']
  #del loader.train_df_id['post_process_nn']
  out_path = 'model' +  path.split('/')[-1][:-4] 
  loader.train_df_id.to_csv(out_path,index=False)
  print("Saving Done")

  stop_time = time.time()
  print("----------------______--------------------")
  print(stop_time-start_time)

