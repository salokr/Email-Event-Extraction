from glob import glob
import ast
full_data_address = './data/full_data'
train_data_address = './data/train'
dev_data_address = './data/dev'
test_data_address = './data/test'






def count_emails(full_data):
	print("Total number of Threads: ", len(full_data))
	for thread in full_data:





def collect_stats(address):
	event_counts = {}
	email_counts, non_event_emails, event_emails, word_count, trigger_word_count, trigger_count = 0, 0, 0, 0, 0, 0
	all_files = glob(f"{address}/*.json")
	for file in all_files:
		jsn = json.load(open(file))
		email_counts += len(jsn['sentences'])
		sentence_word_counts = sum([len(sen) for sen in jsn['sentences']])
		word_count += sentence_word_counts
		for idx, (turns, sentence) in enumerate(zip(jsn['events'], jsn['sentences'])):
			for event_type in jsn['events'][turns]:#each turn is grouped by same event type
				if(event_counts.get(event_type.strip()) is None):
					event_counts[event_type.strip()] = 0
				labels, triggers, metaSRs = jsn['events'][turns][event_type]['labels'], jsn['events'][turns][event_type]['triggers'], jsn['events'][turns][event_type]['extras']
				event_counts[event_type.strip()] += len(labels)
				if(event_type=="O" or event_type==""):
					non_event_emails += 1
				else:
					event_emails += len(labels)
				for trigger in triggers:
					# print(trigger)
					# [ast.literal_eval(x) if x.strip()!='' else {'words':"", 'indices': ''} for x in triggers]
					if(trigger.strip()!=""):
						while(type(trigger)!=type({})):
							trigger = ast.literal_eval(trigger)
					else:
						trigger = {'words':"", 'indices': ''}
					trigger_word_count += len(trigger['indices'].split())
				trigger_count += len(triggers)
	dict_sum = sum(event_counts.values())
	for key, value in event_counts.items():
		value = value/dict_sum
		event_counts[key] = value
	print(json.dumps(event_counts, indent = 4))
	print('-'*100)
	print("Email counts: ", email_counts)
	print("Non event emails: ", non_event_emails)
	print("Event Emails: ", event_emails)
	print("Avg word counts:", word_count/email_counts)
	print("Avg words in triggers: ", trigger_word_count/trigger_count)
	print('-'*100)


collect_stats(full_data_address)
collect_stats(train_data_address)
collect_stats(dev_data_address)
collect_stats(test_data_address)



