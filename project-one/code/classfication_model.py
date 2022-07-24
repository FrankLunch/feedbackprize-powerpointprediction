submission_ending = '''    import os
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('--model_paths', nargs='+', required=True)
ap.add_argument("--save_name", type=str, required=True)
ap.add_argument("--max_len", type=int, required=True)
args = ap.parse_args()

    
if ("v3" in args.save_name)|("v2" in args.save_name):
    # https://www.kaggle.com/nbroad/deberta-v2-3-fast-tokenizer
    # The following is necessary if you want to use the fast tokenizer for deberta v2 or v3
    # This must be done before importing transformers
    import shutil
    from pathlib import Path

    transformers_path = Path("/opt/conda/lib/python3.7/site-packages/transformers")

    input_dir = Path("../input/deberta-v2-3-fast-tokenizer")

    convert_file = input_dir / "convert_slow_tokenizer.py"
    conversion_path = transformers_path/convert_file.name

    if conversion_path.exists():
        conversion_path.unlink()

    shutil.copy(convert_file, transformers_path)
    deberta_v2_path = transformers_path / "models" / "deberta_v2"

    for filename in ['tokenization_deberta_v2.py', 'tokenization_deberta_v2_fast.py']:
        filepath = deberta_v2_path/filename
        if filepath.exists():
            filepath.unlink()

        shutil.copy(input_dir/filename, filepath)


        
import gc
import pickle
import numpy as np
import pandas as pd
import transformers
import multiprocessing as mp
from scipy.special import softmax
from torch.utils.data import Dataset
from transformers import (AutoModelForTokenClassification, 
                          AutoTokenizer, 
                          TrainingArguments, 
                          Trainer)

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

NUM_CORES = 16
BATCH_SIZE = 4
MAX_SEQ_LENGTH = args.max_len
PRETRAINED_MODEL_PATHS = args.model_paths
if "debertal_chris" in args.save_name:
    print('==> using -1 in offset mapping...')
if ("v3" in args.save_name)|("v2" in args.save_name):
    print('==> using -1 in offset mapping...')
    
AGG_FUNC = np.mean
print('==> using span token mean...')

#address_api
#TEST_DIR = '../input/feedback-prize-2021/test/'
TEST_DIR ='../input/trypre'

MIN_TOKENS = {
    "Lead": 32,
    "Position": 5,
    "Evidence": 35,
    "Claim": 7,
    "Concluding Statement": 6,
    "Counterclaim": 6,
    "Rebuttal": 6
}

if "chris" not in args.save_name:
    ner_labels = {'O': 0,
                  'B-Lead': 1,
                  'I-Lead': 2,
                  'B-Position': 3,
                  'I-Position': 4,
                  'B-Evidence': 5,
                  'I-Evidence': 6,
                  'B-Claim': 7,
                  'I-Claim': 8,
                  'B-Concluding Statement': 9,
                  'I-Concluding Statement': 10,
                  'B-Counterclaim': 11,
                  'I-Counterclaim': 12,
                  'B-Rebuttal': 13,
                  'I-Rebuttal': 14}
else:
    print("==> Using Chris BIO")
    ner_labels = {'O': 14,
                  'B-Lead': 0,
                  'I-Lead': 1,
                  'B-Position': 2,
                  'I-Position': 3,
                  'B-Evidence': 4,
                  'I-Evidence': 5,
                  'B-Claim': 6,
                  'I-Claim': 7,
                  'B-Concluding Statement': 8,
                  'I-Concluding Statement': 9,
                  'B-Counterclaim': 10,
                  'I-Counterclaim': 11,
                  'B-Rebuttal': 12,
                  'I-Rebuttal': 13}


inverted_ner_labels = dict((v,k) for k,v in ner_labels.items())
inverted_ner_labels[-100] = 'Special Token'

test_files = os.listdir(TEST_DIR)

# accepts file path, returns tuple of (file_ID, txt split, NER labels)
def generate_text_for_file(input_filename):
    curr_id = input_filename.split('.')[0]
    with open(os.path.join(TEST_DIR, input_filename)) as f:
        curr_txt = f.read()

    return curr_id, curr_txt

with mp.Pool(NUM_CORES) as p:
    ner_test_rows = p.map(generate_text_for_file, test_files)
    
if ("v3" in args.save_name)|("v2" in args.save_name):
    from transformers.models.deberta_v2.tokenization_deberta_v2_fast import DebertaV2TokenizerFast
    tokenizer = DebertaV2TokenizerFast.from_pretrained(PRETRAINED_MODEL_PATHS[0])
else:
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_PATHS[0])
# Check is rust-based fast tokenizer
assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)

ner_test_rows = sorted(ner_test_rows, key=lambda x: len(tokenizer(x[1], max_length=MAX_SEQ_LENGTH, truncation=True)['input_ids']))

# tokenize and store word ids
def tokenize_with_word_ids(ner_raw_data):
    # ner_raw_data is shaped (num_examples, 3) where cols are (ID, words, word-level labels)
    tokenized_inputs = tokenizer([x[1] for x in ner_raw_data], 
                                 max_length=MAX_SEQ_LENGTH,
                                 return_offsets_mapping=True,
                                 truncation=True)
    
    tokenized_inputs['id'] = [x[0] for x in ner_raw_data]
    tokenized_inputs['offset_mapping'] = [tokenized_inputs['offset_mapping'][i] for i in range(len(ner_raw_data))]
    
    return tokenized_inputs

tokenized_all = tokenize_with_word_ids(ner_test_rows)

class NERDataset(Dataset):
    def __init__(self, input_dict):
        self.input_dict = input_dict
        
    def __getitem__(self, index):
        return {k:self.input_dict[k][index] for k in self.input_dict.keys() if k not in {'id', 'offset_mapping'}}
    
    def get_filename(self, index):
        return self.input_dict['id'][index]
    
    def get_offset(self, index):
        return self.input_dict['offset_mapping'][index]
    
    def __len__(self):
        return len(self.input_dict['input_ids'])

test_dataset = NERDataset(tokenized_all)

soft_predictions = None
hfargs = TrainingArguments(output_dir='None',
                         log_level='warning',
                         per_device_eval_batch_size=BATCH_SIZE)

for idx, curr_path in enumerate(PRETRAINED_MODEL_PATHS):

    if args.save_name == "longformerwithlstm":
        model = LongformerForTokenClassificationwithbiLSTM.from_pretrained(curr_path)
    elif args.save_name == "debertawithlstm":
        model = DebertaForTokenClassificationwithbiLSTM.from_pretrained(curr_path)
    else:
        model = AutoModelForTokenClassification.from_pretrained(curr_path, trust_remote_code=True)
    trainer = Trainer(model,
                      hfargs,
                      tokenizer=tokenizer)
    
    curr_preds, _, _ = trainer.predict(test_dataset)
    curr_preds = curr_preds.astype(np.float16)
    curr_preds = softmax(curr_preds, -1)

    if soft_predictions is not None:
        soft_predictions = soft_predictions + curr_preds
    else:
        soft_predictions = curr_preds
        
    del model, trainer, curr_preds
    gc.collect()

soft_predictions = soft_predictions / len(PRETRAINED_MODEL_PATHS)

soft_claim_predictions = soft_predictions[:, :, 8]

predictions = np.argmax(soft_predictions, axis=2)
soft_predictions = np.max(soft_predictions, axis=2)

def generate_token_to_word_mapping(txt, offset):
    # GET WORD POSITIONS IN CHARS
    w = []
    blank = True
    for i in range(len(txt)):
        if not txt[i].isspace() and blank==True:
            w.append(i)
            blank=False
        elif txt[i].isspace():
            blank=True
    w.append(1e6)

    # MAPPING FROM TOKENS TO WORDS
    word_map = -1 * np.ones(len(offset),dtype='int32')
    w_i = 0
    for i in range(len(offset)):
        if offset[i][1]==0: continue
        while offset[i][0]>=(w[w_i+1]-("debertal_chris" in args.save_name)-("v3" in args.save_name)\
                             -("v2" in args.save_name) ): w_i += 1
        word_map[i] = int(w_i)

    return word_map

all_preds = []

# Clumsy gathering of predictions at word lvl - only populate with 1st subword pred
for curr_sample_id in range(len(test_dataset)):
    curr_preds = []
    sample_preds = predictions[curr_sample_id]
    sample_offset = test_dataset.get_offset(curr_sample_id)
    sample_txt = ner_test_rows[curr_sample_id][1]
    sample_word_map = generate_token_to_word_mapping(sample_txt, sample_offset)

    word_preds = [''] * (max(sample_word_map) + 1)
    word_probs = dict(zip(range((max(sample_word_map) + 1)),[0]*(max(sample_word_map) + 1)))
    claim_probs = dict(zip(range((max(sample_word_map) + 1)),[0]*(max(sample_word_map) + 1)))

    for i, curr_word_id in enumerate(sample_word_map):
        if curr_word_id != -1:
            if word_preds[curr_word_id] == '': # only use 1st subword
                word_preds[curr_word_id] = inverted_ner_labels[sample_preds[i]]
                word_probs[curr_word_id] = soft_predictions[curr_sample_id, i]
                claim_probs[curr_word_id] = soft_claim_predictions[curr_sample_id, i]
            elif 'B-' in inverted_ner_labels[sample_preds[i]]:
                word_preds[curr_word_id] = inverted_ner_labels[sample_preds[i]]
                word_probs[curr_word_id] = soft_predictions[curr_sample_id, i]
                claim_probs[curr_word_id] = soft_claim_predictions[curr_sample_id, i]

    # Dict to hold Lead, Position, Concluding Statement
    let_one_dict = dict() # K = Type, V = (Prob of start token, start, end)

    # If we see tokens I-X, I-Y, I-X -> change I-Y to I-X
    for j in range(1, len(word_preds) - 1):
        pred_trio = [word_preds[k] for k in [j - 1, j, j + 1]]
        splitted_trio = [x.split('-')[0] for x in pred_trio]
        if all([x == 'I' for x in splitted_trio]) and pred_trio[0] == pred_trio[2] and pred_trio[0] != pred_trio[1]:
            word_preds[j] = word_preds[j-1]

    # B-X, ? (not B), I-X -> change ? to I-X
    for j in range(1, len(word_preds) - 1):
        if 'B-' in word_preds[j-1] and word_preds[j+1] == f"I-{word_preds[j-1].split('-')[-1]}" and word_preds[j] != word_preds[j+1] and 'B-' not in word_preds[j]:
            word_preds[j] = word_preds[j+1]

     # If we see tokens I-X, O, I-X, change center token to the same for stated discourse types
    for j in range(1, len(word_preds) - 1):
        if word_preds[j - 1] in ['I-Lead', 'I-Position', 'I-Concluding Statement'] and word_preds[j-1] == word_preds[j+1] and word_preds[j] == 'O':
            word_preds[j] = word_preds[j-1]

    j = 0 # start of candidate discourse
    while j < len(word_preds): 
        cls = word_preds[j] 
        cls_splitted = cls.split('-')[-1]
        end = j + 1 # try to extend discourse as far as possible

        if word_probs[j] > 0.54: 
            # Must match suffix i.e., I- to I- only; no B- to I-
            while end < len(word_preds) and (word_preds[end].split('-')[-1] == cls_splitted if cls_splitted in ['Lead', 'Position', 'Concluding Statement'] else word_preds[end] == f'I-{cls_splitted}'):
                end += 1
            # if we're here, end is not the same pred as start
            if cls != 'O' and (end - j > MIN_TOKENS[cls_splitted] or max(word_probs[l] for l in range(j, end)) > 0.73): # needs to be longer than class-specified min
                if cls_splitted in ['Lead', 'Position', 'Concluding Statement']:
                    lpc_max_prob = max(word_probs[c] for c in range(j, end))
                    if cls_splitted in let_one_dict: # Already existing, check contiguous or higher prob
                        prev_prob, prev_start, prev_end = let_one_dict[cls_splitted]
                        if cls_splitted in ['Lead', 'Concluding Statement'] and j - prev_end < 49: # If close enough, combine
                            let_one_dict[cls_splitted] = (max(prev_prob, lpc_max_prob), prev_start, end)
                            
                            # Delete other preds that lie inside the joined LC discourse
                            for l in range(len(curr_preds) - 1, 0, -1):
                                check_span = curr_preds[l][2]
                                check_start, check_end = int(check_span[0]), int(check_span[-1])
                                if check_start > prev_start and check_end < end:
                                    del curr_preds[l]
                            
                        elif lpc_max_prob > prev_prob: # Overwrite if current candidate is more likely
                            let_one_dict[cls_splitted] = (lpc_max_prob, j, end)
                    else: # Add to it
                        let_one_dict[cls_splitted] = (lpc_max_prob, j, end)
                else:
                    # Lookback and add preceding I- tokens
                    while j - 1 > 0 and word_preds[j-1] == cls:
                        j = j - 1
                    # Try to add the matching B- tag if immediately precedes the current I- sequence
                    if j - 1 > 0 and word_preds[j-1] == f'B-{cls_splitted}':
                        j = j - 1


                    #############################################################
                    # Run a bunch of adjustments to discourse predictions based on CV 
                    adj_start, adj_end = j, end + 1

                    # Run some heuristics against previous discourse
                    if len(curr_preds) > 0:
                        prev_span = list(map(int, curr_preds[-1][2].split()))
                        prev_start, prev_end = prev_span[0], prev_span[-1]

                        # Join adjacent rebuttals
                        if cls_splitted in 'Rebuttal':                        
                            if curr_preds[-1][1] == cls_splitted and adj_start - prev_end < 32:
                                del curr_preds[-1]
                                combined_list = prev_span + list(range(adj_start, adj_end))                                
                                curr_preds.append((test_dataset.get_filename(curr_sample_id), 
                                                   cls_splitted, 
                                                   ' '.join(map(str, combined_list)),
                                                   AGG_FUNC([word_probs[i] for i in combined_list if i in word_probs.keys()])))
                                j = end
                                continue
                                
                        elif cls_splitted in 'Counterclaim':                        
                            if curr_preds[-1][1] == cls_splitted and adj_start - prev_end < 24:
                                del curr_preds[-1]
                                combined_list = prev_span + list(range(adj_start, adj_end))                                
                                curr_preds.append((test_dataset.get_filename(curr_sample_id), 
                                                   cls_splitted, 
                                                   ' '.join(map(str, combined_list)),
                                                  AGG_FUNC([word_probs[i] for i in combined_list if i in word_probs.keys()])))
                                j = end
                                continue

                        elif cls_splitted in 'Evidence':                        
                            if curr_preds[-1][1] == cls_splitted and 8 < adj_start - prev_end < 25:
                                if max(claim_probs[l] for l in range(prev_end+1, adj_start)) > 0.35:
                                    claim_tokens = [str(l) for l in range(prev_end+1, adj_start) if claim_probs[l] > 0.15]
                                    if len(claim_tokens) > 2:
                                        curr_preds.append((test_dataset.get_filename(curr_sample_id), 
                                                           'Claim', 
                                                           ' '.join(claim_tokens),
                                                           AGG_FUNC([word_probs[int(i)] for i in claim_tokens if int(i) in word_probs.keys()])))
                        # If gap with discourse of same type, extend to it 
                        elif curr_preds[-1][1] == cls_splitted and adj_start - prev_end > 2:
                            adj_start -= 1

                    # Adjust discourse lengths if too long or short
                    if cls_splitted == 'Evidence':
                        if adj_end - adj_start < 45:
                            adj_start -= 9
                        else:
                            adj_end -= 1
                    elif cls_splitted == 'Claim':
                        if adj_end - adj_start > 24:
                            adj_end -= 1
                    elif cls_splitted == 'Counterclaim':
                        if adj_end - adj_start > 24:
                            adj_end -= 1
                        else:
                            adj_start -= 1
                            adj_end += 1
                    elif cls_splitted == 'Rebuttal':
                        if adj_end - adj_start > 32:
                            adj_end -= 1
                        else:
                            adj_start -= 1
                            adj_end += 1
                    adj_start = max(0, adj_start)
                    adj_end = min(len(word_preds) - 1, adj_end)
                    curr_preds.append((test_dataset.get_filename(curr_sample_id), 
                                       cls_splitted, 
                                       ' '.join(map(str, list(range(adj_start, adj_end)))),
                                       AGG_FUNC([word_probs[i] for i in range(adj_start, adj_end) if i in word_probs.keys()])))

        j = end 

    # Add the Lead, Position, Concluding Statement
    for k, v in let_one_dict.items():
        pred_start = v[1]
        pred_end = v[2]

        # Lookback and add preceding I- tokens
        while pred_start - 1 > 0 and word_preds[pred_start-1] == f'I-{k}':
            pred_start = pred_start - 1
        # Try to add the matching B- tag if immediately precedes the current I- sequence
        if pred_start - 1 > 0 and word_preds[pred_start - 1] == f'B-{k}':
            pred_start = pred_start - 1

        # Extend short Leads and Concluding Statements
        if k == 'Lead':
            if pred_end - pred_start < 33:
                pred_end = min(len(word_preds), pred_end + 5)
            else:
                pred_end -= 5
        elif k == 'Concluding Statement':
            if pred_end - pred_start < 23:
                pred_start = max(0, pred_start - 1)
                pred_end = min(len(word_preds), pred_end + 10)
        elif k == 'Position':
            if pred_end - pred_start < 18:
                pred_end = min(len(word_preds), pred_end + 3)

        pred_start = max(0, pred_start)
        if pred_end - pred_start > 6:
            curr_preds.append((test_dataset.get_filename(curr_sample_id), 
                               k, 
                               ' '.join(map(str, list(range(pred_start, pred_end)))),
                               AGG_FUNC([word_probs[i] for i in range(pred_start, pred_end) if i in word_probs.keys()])))

    all_preds.extend(curr_preds)

output_df = pd.DataFrame(all_preds)
output_df.columns = ['id', 'class', 'predictionstring', 'scores']
output_df.to_csv(f'{args.save_name}.csv', index=False)'''

#with open('submission.py', mode='a') as file:
 #   file.write(submission_ending)




#加权盒融合函数
'''
Code taken and modified for 1D sequences from:
https://github.com/ZFTurbo/Weighted-Boxes-Fusion/blob/master/ensemble_boxes/ensemble_boxes_wbf.py
'''
import warnings
import numpy as np

def prefilter_boxes(boxes, scores, labels, weights, thr):
    # Create dict with boxes stored by its label
    new_boxes = dict()

    for t in range(len(boxes)):

        if len(boxes[t]) != len(scores[t]):
            print('Error. Length of boxes arrays not equal to length of scores array: {} != {}'.format(len(boxes[t]), len(scores[t])))
            exit()

        for j in range(len(boxes[t])):
            score = scores[t][j]
            if score < thr:
                continue
            label = labels[t][j]
            box_part = boxes[t][j]

            x = float(box_part[0])
            y = float(box_part[1])

            # Box data checks
            if y < x:
                warnings.warn('Y < X value in box. Swap them.')
                x, y = y, x

            # [label, score, weight, model index, x, y]
            b = [label, float(score) * weights[t], weights[t], t, x, y]
            if label not in new_boxes:
                new_boxes[label] = []
            new_boxes[label].append(b)

    # Sort each list in dict by score and transform it to numpy array
    for k in new_boxes:
        current_boxes = np.array(new_boxes[k])
        new_boxes[k] = current_boxes[current_boxes[:, 1].argsort()[::-1]]

    return new_boxes


def get_weighted_box(boxes, conf_type='avg'):
    """
    Create weighted box for set of boxes
    :param boxes: set of boxes to fuse
    :param conf_type: type of confidence one of 'avg' or 'max'
    :return: weighted box (label, score, weight, model index, x, y)
    """

    box = np.zeros(6, dtype=np.float32)
    conf = 0
    conf_list = []
    w = 0
    for b in boxes:
        box[4:] += (b[1] * b[4:])
        conf += b[1]
        conf_list.append(b[1])
        w += b[2]
    box[0] = boxes[0][0]
    if conf_type == 'avg':
        box[1] = conf / len(boxes)
    elif conf_type == 'max':
        box[1] = np.array(conf_list).max()
    elif conf_type in ['box_and_model_avg', 'absent_model_aware_avg']:
        box[1] = conf / len(boxes)
    box[2] = w
    box[3] = -1 # model index field is retained for consistensy but is not used.
    box[4:] /= conf
    return box


def find_matching_box_quickly(boxes_list, new_box, match_iou):
    """ 
        Reimplementation of find_matching_box with numpy instead of loops. Gives significant speed up for larger arrays
        (~100x). This was previously the bottleneck since the function is called for every entry in the array.

        boxes_list: shape: (N, label, score, weight, model index, x, y)
        new_box: shape: (label, score, weight, model index, x, y)
    """
    def bb_iou_array(boxes, new_box):
        '''
        boxes: shape: (N, x, y)
        new_box: shape: (x, y)
        '''
        # bb interesection over union
        x_min = np.minimum(boxes[:, 0], new_box[0])
        x_max = np.maximum(boxes[:, 0], new_box[0])
        y_min = np.minimum(boxes[:, 1], new_box[1])+1
        y_max = np.maximum(boxes[:, 1], new_box[1])+1

        iou = np.maximum(0, (y_min-x_max)/(y_max-x_min))

        return iou

    if boxes_list.shape[0] == 0:
        return -1, match_iou

    # boxes = np.array(boxes_list)
    boxes = boxes_list

    ious = bb_iou_array(boxes[:, 4:], new_box[4:])

    ious[boxes[:, 0] != new_box[0]] = -1

    best_idx = np.argmax(ious)
    best_iou = ious[best_idx]

    if best_iou <= match_iou:
        best_iou = match_iou
        best_idx = -1

    return best_idx, best_iou


def weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=None, iou_thr=0.55, skip_box_thr=0.0, conf_type='avg', allows_overflow=False):
    '''
    :param boxes_list: list of boxes predictions from each model, each box is 2 numbers.
     It has 3 dimensions (models_number, model_preds, 2)
     Order of boxes: x, y.
    :param scores_list: list of scores for each model
    :param labels_list: list of labels for each model
    :param weights: list of weights for each model. Default: None, which means weight == 1 for each model
    :param iou_thr: IoU value for boxes to be a match
    :param skip_box_thr: exclude boxes with score lower than this variable
    :param conf_type: how to calculate confidence in weighted boxes. 'avg': average value, 'max': maximum value, 'box_and_model_avg': box and model wise hybrid weighted average, 'absent_model_aware_avg': weighted average that takes into account the absent model.
    :param allows_overflow: false if we want confidence score not exceed 1.0

    :return: boxes: boxes coordinates (Order of boxes: x, y).
    :return: scores: confidence scores
    :return: labels: boxes labels
    '''

    if weights is None:
        weights = np.ones(len(boxes_list))
    if len(weights) != len(boxes_list):
        print('Warning: incorrect number of weights {}. Must be: {}. Set weights equal to 1.'.format(len(weights), len(boxes_list)))
        weights = np.ones(len(boxes_list))
    weights = np.array(weights)

    if conf_type not in ['avg', 'max', 'box_and_model_avg', 'absent_model_aware_avg']:
        print('Unknown conf_type: {}. Must be "avg", "max" or "box_and_model_avg", or "absent_model_aware_avg"'.format(conf_type))
        exit()

    filtered_boxes = prefilter_boxes(boxes_list, scores_list, labels_list, weights, skip_box_thr)
    if len(filtered_boxes) == 0:
        return np.zeros((0, 2)), np.zeros((0,)), np.zeros((0,))

    overall_boxes = []
    for label in filtered_boxes:
        boxes = filtered_boxes[label]
        new_boxes = []
        weighted_boxes = np.empty((0,6)) ## [label, score, weight, model index, x, y]
        # Clusterize boxes
        for j in range(0, len(boxes)):
            index, best_iou = find_matching_box_quickly(weighted_boxes, boxes[j], iou_thr)

            if index != -1:
                new_boxes[index].append(boxes[j])
                weighted_boxes[index] = get_weighted_box(new_boxes[index], conf_type)
            else:
                new_boxes.append([boxes[j].copy()])
                weighted_boxes = np.vstack((weighted_boxes, boxes[j].copy()))

        # Rescale confidence based on number of models and boxes
        for i in range(len(new_boxes)):
            clustered_boxes = np.array(new_boxes[i])
            if conf_type == 'box_and_model_avg':
                # weighted average for boxes
                weighted_boxes[i, 1] = weighted_boxes[i, 1] * len(clustered_boxes) / weighted_boxes[i, 2]
                # identify unique model index by model index column
                _, idx = np.unique(clustered_boxes[:, 3], return_index=True)
                # rescale by unique model weights
                weighted_boxes[i, 1] = weighted_boxes[i, 1] *  clustered_boxes[idx, 2].sum() / weights.sum()
            elif conf_type == 'absent_model_aware_avg':
                # get unique model index in the cluster
                models = np.unique(clustered_boxes[:, 3]).astype(int)
                # create a mask to get unused model weights
                mask = np.ones(len(weights), dtype=bool)
                mask[models] = False
                # absent model aware weighted average
                weighted_boxes[i, 1] = weighted_boxes[i, 1] * len(clustered_boxes) / (weighted_boxes[i, 2] + weights[mask].sum())
            elif conf_type == 'max':
                weighted_boxes[i, 1] = weighted_boxes[i, 1] / weights.max()
            elif not allows_overflow:
                weighted_boxes[i, 1] = weighted_boxes[i, 1] * min(len(weights), len(clustered_boxes)) / weights.sum()
            else:
                weighted_boxes[i, 1] = weighted_boxes[i, 1] * len(clustered_boxes) / weights.sum()
        
        # REQUIRE BBOX TO BE PREDICTED BY AT LEAST 2 MODELS
        #for i in range(len(new_boxes)):
        #    clustered_boxes = np.array(new_boxes[i])
        #    if len(np.unique(clustered_boxes[:, 3])) > 1:
        #        overall_boxes.append(weighted_boxes[i])
                
        overall_boxes.append(weighted_boxes) # NOT NEEDED FOR "REQUIRE TWO MODELS" ABOVE
    overall_boxes = np.concatenate(overall_boxes, axis=0) # NOT NEEDED FOR "REQUIRE TWO MODELS" ABOVE
    #overall_boxes = np.array(overall_boxes) # NEEDED FOR "REQUIRE TWO MODELS" ABOVE
    overall_boxes = overall_boxes[overall_boxes[:, 1].argsort()[::-1]]
    boxes = overall_boxes[:, 4:]
    scores = overall_boxes[:, 1]
    labels = overall_boxes[:, 0]
    return boxes, scores, labels





#运行各个模型，训练并生成预测csv文件


#with open('submission.py', mode='a') as file:
 #   file.write(submission_ending)
#use subprocess
import subprocess
subprocess.run(['python3','generate_preds.py','--model_paths','../input/deberta-v2-xlarge/deberta-v2-xlarge-v6000-f0/checkpoint-7500','../input/deberta-v2-xlarge/deberta-v2-xlarge-v6003-f3/checkpoint-9000','--save_name','deberta_v2','--max_len','1536'])

#get_ipython().system('python3 generate_preds.py --model_paths ../input/deberta-v2-xlarge/deberta-v2-xlarge-v6000-f0/checkpoint-7500                                         ../input/deberta-v2-xlarge/deberta-v2-xlarge-v6003-f3/checkpoint-9000                             --save_name deberta_v2 --max_len 1536')





import pandas as pd
import numpy as np, os
deberta_v2_csv = pd.read_csv("deberta_v2.csv").dropna()


# # Ensemble Models with WBF
# We will now read in the 10 submission files generated above and apply WBF to ensemble them. After applying WBF, it is important to remove predictions with confidence score below threshold. This is explained [here][1]. 
# 
# If only 1 model out of 10 models makes a certain span prediction, that prediction will still be present in WBF's outcome. However that prediction will have a very low confidence score because that model's confidence score will be averaged with 9 zero confidence scores. We found optimal confidence scores per class by analyzing our CV OOF score. For each class, we vary the threshold and compute the corresponding class metric score.
# 
# ![](https://raw.githubusercontent.com/cdeotte/Kaggle_Images/main/Mar-2022/conf_scores.png)
# 
# [1]: https://www.kaggle.com/c/tensorflow-great-barrier-reef/discussion/307609




import os
#模型已经训练好了，下面是对测试集的处理
#address_api
#TEST_DIR = '../input/feedback-prize-2021/test/'#读取路径
TEST_DIR ='../input/trypre'
test_files = os.listdir(TEST_DIR)
v_ids = [f.replace('.txt','') for f in test_files]




#将所有预测综合，通过加权盒模型得到综合submmsion
import math

class_to_label = {
    'Claim': 0, 
    'Evidence': 1, 
    'Lead':2, 
    'Position':3, 
    'Concluding Statement':4,
    'Counterclaim':5, 
    'Rebuttal':6
}

# Threshold found from CV
label_to_threshold = {
    0 : 0.275, #Claim
    1 : 0.375, #Evidence
    2 : 0.325, #Lead
    3 : 0.325, #Position
    4 : 0.4, #Concluding Statement
    5 : 0.275, #Counterclaim
    6 : 0.275 #Rebuttal
}

label_to_class = {v:k for k, v in class_to_label.items()}

def preprocess_for_wbf(df_list):
    boxes_list=[]
    scores_list=[]
    labels_list=[]
    
    for df in df_list:
        scores_list.append(df['scores'].values.tolist())
        labels_list.append(df['class'].map(class_to_label).values.tolist())
        predictionstring = df.predictionstring.str.split().values
        df_box_list = []
        for bb in predictionstring:
            df_box_list.append([int(bb[0]), int(bb[-1])])
        boxes_list.append(df_box_list)
    return boxes_list, scores_list, labels_list

def postprocess_for_wbf(idx, boxes_list, scores_list, labels_list):
    preds = []
    for box, score, label in zip(boxes_list, scores_list, labels_list):
        if score > label_to_threshold[label]: 
            start = math.ceil(box[0])
            end = int(box[1])
            preds.append((idx, label_to_class[label], ' '.join([str(x) for x in range(start, end+1)])))
    return preds

def generate_wbf_for_id(i):
    df10 = deberta_v2_csv[deberta_v2_csv['id']==i]
    
    boxes_list, scores_list, labels_list = preprocess_for_wbf([df10])
    nboxes_list, nscores_list, nlabels_list = weighted_boxes_fusion(boxes_list, scores_list, labels_list, iou_thr=0.33, conf_type='avg')

    return postprocess_for_wbf(i, nboxes_list, nscores_list, nlabels_list)





import multiprocessing as mp

#with mp.Pool(2) as p:
 #   list_of_list = p.map(generate_wbf_for_id, v_ids)
list_of_list = map(generate_wbf_for_id, v_ids)
preds = [x for sub_list in list_of_list for x in sub_list]





sub = pd.DataFrame(preds)
sub.columns = ["id", "class", "predictionstring"]
#sub.to_csv('submission.csv', index=False)
#sub









# # Visualize Test Predictions
# Below we visualize the test predictions


from pathlib import Path
from spacy import displacy
#address_api
#test_path = Path('../input/feedback-prize-2021/test')
test_path = Path('../input/trypre')
colors = {
            'Lead': '#8000ff',
            'Position': '#2b7ff6',
            'Evidence': '#2adddd',
            'Claim': '#80ffb4',
            'Concluding Statement': 'd4dd80',
            'Counterclaim': '#ff8042',
            'Rebuttal': '#ff0000',
            'Other': '#007f00',
         }

def get_test_text(ids):
    with open(test_path/f'{ids}.txt', 'r') as file: data = file.read()
    return data

def visualize(df):
    ids = df["id"].unique()
    for i in range(len(ids)):
        ents = []
        example = ids[i]
        curr_df = df[df["id"]==example]
        text = " ".join(get_test_text(example).split())
        splitted_text = text.split()
        for i, row in curr_df.iterrows():
            predictionstring = row['predictionstring']
            predictionstring = predictionstring.split()
            wstart = int(predictionstring[0])
            wend = int(predictionstring[-1])
            ents.append({
                             'start': len(" ".join(splitted_text[:wstart])), 
                             'end': len(" ".join(splitted_text[:wend+1])), 
                             'label': row['class']
                        })
        ents = sorted(ents, key = lambda i: i['start'])

        doc2 = {
            "text": text,
            "ents": ents,
            "title": example
        }

        options = {"ents": ['Lead', 'Position', 'Evidence', 'Claim', 'Concluding Statement', 'Counterclaim', 'Rebuttal'], "colors": colors}
        displacy.render(doc2, style="ent", options=options, manual=True, jupyter=True)
        
def output(df):
    ids = df["id"].unique()
    
    result=pd.DataFrame(columns=['id','class','predictionstring','discourse_text'])
    
    for i in range(len(ids)):#对于每个论文次数
        ents = []
        example = ids[i]#example存储了这个论文的id号
        curr_df = df[df["id"]==example]#截取了id等于该论文的所有行列
        text = " ".join(get_test_text(example).split())#将文章先分开，再合并
        splitted_text = text.split()#数组形式的文章
        discourse_text=[]
        for i in range(curr_df.shape[0]):
            predictionstring = curr_df.iloc[i][2]
            predictionstring = predictionstring.split()
            wstart = int(predictionstring[0])
            wend = int(predictionstring[-1])
            #curr_df.iloc[i][2]=" ".join(splitted_text[wstart:wend-1])
            discourse_text.append(" ".join(splitted_text[wstart:wend-1]))
            #curr_df.iloc[i][2]=" asdasd"
        curr_df['discourse_text']=discourse_text
       # return curr_df
        result=pd.concat([result,curr_df])
    return result
        





#if len(sub["id"].unique())==5:
text_csv=output(sub)
del text_csv['predictionstring']
text_csv.to_csv("submmsion.csv ")


#a.to_csv("submmsion.csv ")







