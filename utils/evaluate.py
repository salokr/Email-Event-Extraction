from timeout import timeout
from string_utils import extract_span_close_to_trigger, exact_match_on_index, get_arg_span_indices_from_email
import ast, re
from tqdm import tqdm
from fuzzywuzzy import fuzz
import numpy as np
def safeDivide(numerator, denominator):
	if(denominator==0):
		return 0
	return numerator/denominator

def getScores(numerator_precision, numerator_recall, denominator_prediction, denominator_gold):
	precision = safeDivide(numerator_precision, denominator_prediction)
	recall = safeDivide(numerator_recall, denominator_gold)
	return {'Precision': precision, 'Recall': recall, 'F1':safeDivide(2*precision*recall,(precision+recall))}



def get_arg_scores(predicted_arg, gold_arg):
	if(predicted_arg is not None and len(predicted_arg['span_indices'])>0 and predicted_arg['span_indices'][0] not in gold_arg["span_indices"]):
		return 0, 0, 0, None#just a quick eval step
	try:
		overlaps = get_arg_span_indices_from_email(gold_arg["span"], predicted_arg["span"])
	except:
		overlaps = []
	precision, recall, F1 = 0, 0, 0
	total_gold_arg_count = len(gold_arg['span_indices'])
	total_predicted_arg_count = len(predicted_arg['span_indices'])
	total_matched_count = len(overlaps[0]['span_indices']) if len(overlaps)>0 else 0
	precision = safeDivide(total_matched_count, total_predicted_arg_count)
	recall = safeDivide(total_matched_count, total_gold_arg_count)
	f_score = safeDivide((2*precision*recall), (precision+recall))
	return precision, recall, f_score, overlaps


#for multiple members, we need to split them into different arguments
def post_process_members(arg_list):
	new_arg_list = []
	for arg_dict in arg_list:
		if(arg_dict.get("meta_srs") is not None):
			new_arg_list.append(arg_dict)
			continue
		span = arg_dict["span"]
		type_ = arg_dict["type"]
		if(type_.lower().find("member")>=0):
			arg_members = span.split(" and ")
			for arg_idx, arg_member in enumerate(arg_members):
				new_arg_dict = {}
				a_n = type_ + "_" + str(arg_idx)
				new_arg_dict["span"] = arg_member.strip()
				new_arg_dict["type"] = a_n
				if(arg_dict.get("indices") is not None):
					new_arg_dict["indices"] = arg_dict["indices"]
				new_arg_list.append(new_arg_dict)
		else:
			new_arg_list.append(arg_dict)
	return new_arg_list

# save dumps for more analysis
def json_dump(ob, fn, tp):
	fn, tp = str(fn).replace("/", "_"), str(tp).replace("/", "_")
	with open(f"./analyaze/analyze_{tp}_{fn}", "w") as f:
		json.dump(ob, f, indent = 4)


def trigger_scores(threads, desc = ""):
	predicted_trigger_count, gold_trigger_count = 0, 0#
	trigger_identification_count, trigger_class_count = 0, 0#
	trigger_identification_distinct_count, trigger_class_distinct_count = 0, 0
	#
	predicted_argument_count, gold_argument_count = 0, 0
	arg_identification_prec, arg_identification_rec, arg_class_prec, arg_class_rec = 0, 0, 0, 0
	#
	skipped_trigger, skipped_argument = [], []
	accepted_trigger, accepted_argument = [], []


	trigger_scores_p,trigger_scores_r,trigger_scores_f, argument_scores_p_id, argument_scores_r_id, argument_scores_f_id = [], [], [], [], [], []
	argument_scores_p_cl, argument_scores_r_cl, argument_scores_f_cl = [], [], [] 
	for _, email in  tqdm(enumerate(threads), total = len(threads), desc="Evaluating end2end..."):
		email_trigger_recall, email_trigger_precision, email_trigger_f_score = [], [] , []
		email_arg_recall_id, email_arg_precision_id, email_arg_f_score_id = [], [] , []
		email_arg_recall_class, email_arg_precision_class, email_arg_f_score_class = [], [] , []
		email["email_body"] = ' '.join(email["email_body"]).replace('?', ' ?').replace(".", " .").replace("'", " '").strip().split()

		predicted_triggers = [e for e in email['predicted_triggers'] if(e.get("span") is not None and e["span"]).strip()!=""]
		gold_triggers = [e for e in email['gold_triggers'] if(e.get("span") is not None and e["span"]).strip()!=""]

		for x in gold_triggers:
			x['arguments'] = post_process_members(x['arguments'])
		for x in predicted_triggers:
			x['arguments'] = post_process_members(x['arguments'])
		#######
		predicted_trigger_count += len(predicted_triggers)
		gold_trigger_count += len(gold_triggers)
		distinct_gold_trigger_index_id, distinct_gold_trigger_index_class = set(), set()
		gold_argument_count += sum([len(x['arguments']) for x in gold_triggers])
		number_of_gold_arguments = sum([len(x['arguments']) for x in gold_triggers])
		number_of_predicted_arguments = len(predicted_triggers)
		#######
		for predicted_trigger in predicted_triggers:
			if(predicted_trigger["span"].strip()==""):###
				print("Found a blank")
				continue###
			predicted_argument_count += len(predicted_trigger["arguments"])
			best_trigger_index = -1
			if(len(predicted_trigger["indices"])<=0):#for generative model we'll not have indices but for seq labeling we can provide indices apriori
				#to speed up, if the trigger is not in the email ignore it (happens in generative models for initial epochs)
				if(fuzz.token_set_ratio(' '.join(email['email_body']), predicted_trigger["span"])<=30):
					# for debugging 
					# skipped_trigger.append({"trigger_skipped": predicted_trigger, "email_body": email["email_body"], "all_golds": str(gold_triggers)})
					continue
				#extracting the sequences might take long time due to the usage of LCS algorithm. This happens for BART in initial rounds. To handle them, we set a time-out
				try:
					matched_trigger_details = get_arg_span_indices_from_email(email['email_body'], predicted_trigger["span"].split())
				except:
					matched_trigger_details = []#time out occured
			else:
				matched_trigger_details = [{"span": predicted_trigger["span"], "span_indices": predicted_trigger['indices']}]
			if(len(matched_trigger_details)<=0):#there are no overlapping triggers
				#for debugging
				# skipped_trigger.append({"trigger_not_found": predicted_trigger, "email_body": email["email_body"], "all_golds": str(gold_triggers)})
				continue
			predicted_trigger_index = matched_trigger_details[0]["span_indices"]
			predicted_trigger["indices"] = predicted_trigger_index
			
			for gold_index, gold_trigger in enumerate(gold_triggers):
				if(len(gold_trigger['indices'])<=0):
					matched_gold_trigger_details = get_arg_span_indices_from_email(email['email_body'], gold_trigger["span"].split())
					gold_trigger['indices'] = matched_gold_trigger_details[0]["span_indices"]
				#
				found = None
				matched_trigger = exact_match_on_index(predicted_trigger_index, gold_trigger["indices"])#make sure that we have different triggers
				if(matched_trigger):
					found = gold_trigger
					best_trigger_index = gold_index
					accepted_trigger.append({"For trigger": predicted_trigger, "Matched_GT": gold_triggers[best_trigger_index], "all_args": str(gold_triggers)})
					trigger_identification_count += 1
					distinct_gold_trigger_index_id.add(gold_index)
					#
					if(gold_triggers[best_trigger_index]["type"] == predicted_trigger["type"]):
						trigger_class_count += 1
						distinct_gold_trigger_index_class.add(gold_index)
						#Argument Evaluation, if the trigger has already been matched should we still do arg calculation? 
						predicted_arguments = predicted_trigger["arguments"]
						gold_arguments = gold_trigger["arguments"]
						distinct_gold_arg_index2recall, distinct_gold_arg_class_index2recall = dict(), dict()
						#################For Sequence Labeling We need to analyze MSRs seaprately, generatives can be done by template choice#################
						predicted_meta_srs, ground_truth_meta_srs = None, None
						for p_idx, p in enumerate(predicted_arguments):
							if(p.get("meta_srs") is not  None):
								predicted_meta_srs = p["meta_srs"]
								break
						for g_idx, g in enumerate(gold_arguments):
							if(g.get("meta_srs") is not  None):
								ground_truth_meta_srs = g["meta_srs"]
								break
						if(predicted_meta_srs is not None):
							arg_identification_prec += int(predicted_meta_srs==ground_truth_meta_srs)
							arg_class_prec += int(predicted_meta_srs==ground_truth_meta_srs)
							distinct_gold_arg_index2recall["meta_srs"] = [int(predicted_meta_srs==ground_truth_meta_srs)]
							distinct_gold_arg_class_index2recall["meta_srs"] = [int(predicted_meta_srs==ground_truth_meta_srs)]
							_ = predicted_arguments.pop(p_idx)
						if(ground_truth_meta_srs is not None):
							_ = gold_arguments.pop(g_idx)
						for args in predicted_arguments:
							email_body = email['email_body']
							arg_search = [x.strip() for x in args["span"].split()] if type(args["span"])==type('') else args["span"]
							if((args.get("indices") is None or len(args['indices'])<=0) and (' '.join(arg_search) in["is or will be delivered to", "is or will be performed by", "is or requested to be updated to"] or fuzz.token_set_ratio(' '.join(email_body), ' '.join(arg_search))<=40)):
								skipped_argument.append({"skipped_argument": args, "email_body": email_body, "all_args": str(gold_arguments)})
								continue
							try:
								arg_span_index, arg_role_type = extract_span_close_to_trigger(email['email_body'], [x.strip() for x in args["span"].split()] if type(args["span"])==type('') else args["span"], predicted_trigger_index, span_indices = args.get("indices")), args["type"]#extract spans closest to triggers
							except:
								arg_span_index = None
							if(arg_span_index is None):
								skipped_argument.append({"unfound_argument": args, "email_body": email_body, "all_args": str(gold_arguments)})
								continue#the arg has no match?
							args["span_indices"] = arg_span_index['span_indices'] if type(arg_span_index) == type({}) else arg_span_index
							args["indices"] = args["span_indices"]
							best_arg_index, best_arg_score, best_precision, best_recall, best_overlap = -1, 0, 0, 0, None
							for gold_arg_index, gold_arg in enumerate(gold_arguments):
								if(args["span"]==gold_arg["span"]):
									#to speed things up no need to match now
									gold_arg["indices"] = args["indices"]
									best_arg_score, best_recall, best_precision, best_arg_index = 1, 1, 1, gold_arg_index
									break
								if(gold_arg.get("indices") is None):
									matched_gold_arg_details = get_arg_span_indices_from_email(email['email_body'], gold_arg["span"].split())
									gold_arg['indices'] = matched_gold_arg_details[0]['span_indices']
								try:
									_prec, _recall, _f1, overlaps = get_arg_scores(arg_span_index, gold_arg)
								except:
									_prec, _recall, _f1, overlaps = 0, 0, 0, []
								if(_f1 > best_arg_score):
									best_arg_score = _f1
									best_recall = _recall
									best_precision = _prec
									best_arg_index = gold_arg_index
									best_overlap = overlaps 
							accepted_argument.append({"For arg": args, "best_matched_arg": gold_arguments[best_arg_index] if best_arg_index!=-1 else None, "all_args": str(gold_arguments)})
							arg_identification_prec += best_precision
			
							##p-score
							# print(best_precision)
							email_arg_precision_id.append(best_precision)

							if(distinct_gold_arg_index2recall.get(best_arg_index) is None):
								distinct_gold_arg_index2recall[best_arg_index] = []
							distinct_gold_arg_index2recall[best_arg_index].append(best_recall)
							arg_role_type = re.sub("_\d+","",arg_role_type)

							if(best_arg_index!=-1):
								gold_arguments[best_arg_index]['type'] = re.sub("_\d+", "", gold_arguments[best_arg_index]['type'])
							if(best_arg_index!=-1 and arg_role_type == gold_arguments[best_arg_index]['type']):
								arg_class_prec += best_precision
								if(distinct_gold_arg_class_index2recall.get(best_arg_index) is None):
									distinct_gold_arg_class_index2recall[best_arg_index] = []
								distinct_gold_arg_class_index2recall[best_arg_index].append(best_recall)
								email_arg_precision_class.append(best_precision)
							else:
								arg_class_prec += 0#if model has predicted something, while gold has nothing, it should be accounted for as in precision but not in recall 
								email_arg_precision_class.append(best_precision)

						arg_identification_rec += sum([max(scores) for scores in distinct_gold_arg_index2recall.values()])
						
						##p-score
						email_arg_recall_id.append([max(scores) for scores in distinct_gold_arg_index2recall.values()])
						
						arg_class_rec += sum([max(scores) for scores in distinct_gold_arg_class_index2recall.values()])
						
						##p-score
						email_arg_recall_class.append(sum([max(scores) for scores in distinct_gold_arg_class_index2recall.values()]))
					break#if we have matched a trigger no need to go further
		trigger_identification_distinct_count += len(distinct_gold_trigger_index_id)
		trigger_class_distinct_count +=  len(distinct_gold_trigger_index_class)
		###
		argument_scores_p_id.append(np.mean(email_arg_precision_id))#
		argument_scores_p_cl.append(np.mean(email_arg_precision_class))#

	total_argument_identification_scores = getScores(arg_identification_prec, arg_identification_rec, predicted_argument_count, gold_argument_count)
	total_argument_class_scores = getScores(arg_class_prec, arg_class_rec, predicted_argument_count, gold_argument_count)
	total_trigger_identification_scores = getScores(trigger_identification_count, trigger_identification_distinct_count, predicted_trigger_count, gold_trigger_count)
	total_trigger_class_scores = getScores(trigger_class_count, trigger_class_distinct_count, predicted_trigger_count, gold_trigger_count)
	print("Trigger Identification: Precision {:.3f} Recall {:.3f} F1 {:.3f}".format(total_trigger_identification_scores["Precision"], total_trigger_identification_scores["Recall"], total_trigger_identification_scores["F1"]))
	print("Trigger Class: Precision {:.3f} Recall {:.3f} F1 {:.3f}".format(total_trigger_class_scores["Precision"], total_trigger_class_scores["Recall"], total_trigger_class_scores["F1"]))
	print("Argument Identification: Precision {:.3f} Recall {:.3f} F1 {:.3f}".format(total_argument_identification_scores["Precision"], total_argument_identification_scores["Recall"], total_argument_identification_scores["F1"]))
	print("Argument Class: Precision {:.3f} Recall {:.2f} F1 {:.3f}".format(total_argument_class_scores["Precision"], total_argument_class_scores["Recall"], total_argument_class_scores["F1"]))	
	# json_dump(accepted_trigger, desc, "triggers_accepted")
	# json_dump(skipped_trigger, desc, "triggers_skipped")
	# json_dump(accepted_argument, desc, "args_accepted")
	# json_dump(skipped_argument, desc, "args_skipped")
	return {"EM_trigger_id_scores": total_trigger_identification_scores, "EM_trigger_class_scores": total_trigger_class_scores, "EM_arg_id_scores": total_argument_identification_scores, "EM_arg_class_scores":total_argument_class_scores, "threads": threads}

import json
from argparse import ArgumentParser
parser = ArgumentParser()


if __name__ == "__main__":
	parser.add_argument('-f', '--filename',required = True)
	args = parser.parse_args()
	print(args.filename)
	f = open(args.filename)
	jsn = json.load(f)
	trigger_scores(jsn, args.filename)	

