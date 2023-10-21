#######
## Utilities file for string operations such as to find indices from an email for generative models
## LCS: Longest common subsequence
#######
from timeout import timeout
import copy
def get_random_word(word):
    return word + "1"


def findLCS(x, y, m, n): 
    s = set()
    if m == 0 or n == 0:
        s.add("")
        return s
    if x[m - 1] == y[n - 1]:
        tmp = findLCS(x, y, m - 1, n - 1)
        for string in tmp:
            s.add(string +x[m - 1] + " idx: " + str(m-1)+" [SEP] ")
    else: 
        if L[m - 1][n] >= L[m][n - 1]:
            s = findLCS(x, y, m - 1, n) 
        if L[m][n - 1] >= L[m - 1][n]:
            tmp = findLCS(x, y, m, n - 1)
            for i in tmp:
                s.add(i)
    return s

def LCS(x, y, m, n):
    for i in range(m + 1):
        for j in range(n + 1):
            #print(x[i-1], y[j-1], '<<<<<')
            if i == 0 or j == 0:
                L[i][j] = 0
            elif x[i - 1] == y[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j],
                              L[i][j - 1])
    return L[m][n]
 

# x = "A G T G A T G".split()
# y = "G T T A G".split()
# m = len(x)
# n = len(y)
# print("LCS length is", LCS(x, y, m, n))
# subsequences = findLCS(x, y, m, n)

fun = lambda x: {"span":x[0].strip(), "index": int(x[1])}
N, L = 0, []
@timeout(300)
def getLCS(text, tokens):
    #assert False
    if(type(text)==type("")):
        text = text.split()
    if(type(tokens)==type("")):
        tokens = tokens.split()
    #print('>>>text', text)
    #print('tokens>>>', tokens)
    global N, L
    N = 1030
    L = [[0 for i in range(N)]
        for j in range(N)]
    max_len = LCS(text, tokens, len(text), len(tokens))
    subsequences = findLCS(text, tokens, len(text), len(tokens))
    #print(subsequences)
    min_sum = float('inf')
    min_sum_index = -1
    for s_idx, subsequence in enumerate(subsequences):
        sub_parts = subsequence.split('[SEP]')
        word2index = [fun(sp.split('idx:')) for sp in sub_parts if sp.strip()!='']
        sum_ = sum([word2index[idx]['index']-word2index[idx-1]['index'] for idx in range(1, len(word2index))])
        #print(word2index, sum_)
        if(sum_ < min_sum):
            min_sum = sum_
            min_w2_index = word2index
    output = {"span":"", 'span_indices': []}
    for x in min_w2_index:
        output["span"] += x['span'] + " "
        output["span_indices"].append(x['index'])
    return max_len, output

#getLCS("A G T G A T G".split(), "G T T A G".split())


import copy
def get_arg_span_indices_from_email(text, tokens, msg = ""):
    if(type(text)==type('')):
        text = text.split()
    if(type(tokens)==type('')):
        tokens=tokens.split()
    text, tokens = [x.strip() for x in text], [x.strip() for x in tokens]
    seq_len, seq = getLCS(text, tokens)
    seq['span'] = seq.pop('span').strip()
    final_dict = []
    orig_seq, orig_len = copy.deepcopy(seq), seq_len
    if(seq_len==1):
        order_dict = []
        for x in tokens:
            x_indices = [idx for idx, word in enumerate(text) if(word==x)]
            xx = [{'span':x, 'span_indices': [idd]} for idd in x_indices]
            order_dict.extend(xx)
    else:
        max_seq_len = seq_len
        all_max_subsequences = {}
        for span, span_indices in zip(seq['span'].split(), seq['span_indices']):
            for other_span_indices in [idx for idx, yy in enumerate(text) if yy==span][::-1]:#find all occurance of the current word
                xx = copy.deepcopy(text)
                word_to_replace = span#get a random word
                while(seq['span'].find(word_to_replace)!=-1):#get a random word
                    #print('trying: ', seq['span'])
                    word_to_replace = get_random_word(word_to_replace)#get a random word
                xx[other_span_indices] = word_to_replace
                seq_len, seq = getLCS(xx, tokens)
                #print('XXXXXXX', seq, max_seq_len, xx)
                if(seq_len==max_seq_len):
                    if(all_max_subsequences.get(seq['span']) is None):
                        all_max_subsequences[seq['span']] = []    
                    all_max_subsequences[seq['span']].append(seq)
        order_dict = {}
        for spans in all_max_subsequences:
            spans = all_max_subsequences[spans]
            for span_id in spans:
                span_indices = span_id['span_indices']
                sum_ = sum([span_indices[i] - span_indices[i-1] for i in range(1, len(span_indices))])   
                if(order_dict.get(sum_) is None):
                    order_dict[sum_] = []
                order_dict[sum_].append(span_id)
        order_dict = order_dict[min(order_dict.keys())] if len(order_dict)>0 else {}    
    for x in (order_dict):
        found = False
        for y in final_dict:
            if(x['span']==y['span'] and x['span_indices']==y['span_indices']):
                found = True
                break
        if(not found):
            final_dict.append(x)
    if(len(final_dict)==0 and orig_len>0):
        final_dict = [orig_seq]
    return(final_dict)



def calulate_distance_from_trigger(span, email, trigger):
    if(type(span)==type({})):
        indices = span["span_indices"]
        span_start, span_end = indices[0], indices[-1]
    else:
        span_start, span_end = span[0], span[-1]#1
    if(type(trigger)==type({})):
        trigger_start, trigger_end = trigger["span"]
    else:
        trigger_start, trigger_end = trigger[0], trigger[-1]
    if(span_start < trigger_start):
        distance = abs(trigger_start-span_end)
    else:
        distance = abs(span_start-trigger_end)
    return distance


def extract_span_close_to_trigger(email, span, trigger, span_indices=None):  
    if(type(span)==""):
        span=span.split()
    span_indices = get_arg_span_indices_from_email(email, span, msg="ee") if span_indices is None else [span_indices]
    min_distance = float('inf')
    ret_span = None
    for spans_index in span_indices:
        distance = calulate_distance_from_trigger(spans_index, email, trigger)
        if(distance<min_distance):
            min_distance = distance
            ret_span = spans_index
    return ret_span

def exact_match_on_index(predicted_index, ground_truth_index):
    return predicted_index == ground_truth_index


# x, orig_y = ['aa', 'e', 'f', 'g', 'aa', 'b', 'c', 'dd', 'dd'], ['aa', 'e', 'f'] 
# print('>>>>', get_arg_span_indices_from_email(x, orig_y))



# x, orig_y = ['O', "neal", ',', 'pls', 'call', 'me', 'at', '53211', '.', 'Thanks', ',', 'Sherry'], ["O", "neal"] 
# print('>>>>', get_arg_span_indices_from_email(x, orig_y))


# x = ['Hi', 'Teb,', 'I', 'am', 'trying', 'to', 'verify', 'if', 'Jeanette', 'Doll', 'is', 'still', 'on', 'assignment.', 'Listed', 'below', 'are', 'the', 'Co', '#', 's', 'and', 'CC', '#', 's', 'that', 'Jeanette', 'has', 'been', 'charging', 'her', 'time', 'to.', 'Could', 'you', 'validate', 'if', 'these', 'cost', 'center', 'numbers', 'are', 'still', 'active', 'or', 'inactive?.', 'Is', 'the', 'Company', 'Name', 'and', 'Department', 'the', 'same?', 'Also,', 'please', 'let', 'me', 'know', 'if', 'there', 'are', 'any', 'cost', 'center', 'numbers', 'that', 'need', 'to', 'be', 'added', 'or', 'deleted.', 'Transwestern', 'GPG-TW', 'Rates', '0060', '111004', 'Lastly,', 'is', 'there', 'a', 'tentative', 'end', 'date', 'for', 'this', 'assignment?', 'Thanks!', 'Liz', 'LeGros']
# orig_y = ['Teb']
# print('>>>>', get_arg_span_indices_from_email(x, orig_y))
# x, orig_y = ['aa', 'e', 'f', 'g', 'aa', 'b', 'c', 'dd', 'dd'], ['aa', 'e', 'f'] 
# print('>>>>', get_arg_span_indices_from_email(x, orig_y))


# x = ['Hi', 'Teb,', 'I', 'am', 'trying', 'to', 'verify', 'if', 'Jeanette', 'Doll', 'is', 'still', 'on', 'assignment.', 'Listed', 'below', 'are', 'the', 'Co', '#', 's', 'and', 'CC', '#', 's', 'that', 'Jeanette', 'has', 'been', 'charging', 'her', 'time', 'to.', 'Could', 'you', 'validate', 'if', 'these', 'cost', 'center', 'numbers', 'are', 'still', 'active', 'or', 'inactive?.', 'Is', 'the', 'Company', 'Name', 'and', 'Department', 'the', 'same?', 'Also,', 'please', 'let', 'me', 'know', 'if', 'there', 'are', 'any', 'cost', 'center', 'numbers', 'that', 'need', 'to', 'be', 'added', 'or', 'deleted.', 'Transwestern', 'GPG-TW', 'Rates', '0060', '111004', 'Lastly,', 'is', 'there', 'a', 'tentative', 'end', 'date', 'for', 'this', 'assignment?', 'Thanks!', 'Liz', 'LeGros']
# orig_y = ['Listed', 'below']
# print('>>>>', get_arg_span_indices_from_email(x, orig_y))


# x, orig_y = ['aa', 'b', 'c', 'dd', 'x', 'y', 'aa', 'b', 'c', 'dd'], ['aa', 'b', 'c', 'dd']
# print(getOverlappingSpan(x, orig_y))

# x, orig_y = ['aa', 'e', 'f', 'g', 'aa', 'b', 'c', 'dd', 'dd'], ['b', 'b', 'aa'] 
# print(getOverlappingSpan(x, orig_y))


# x = ['aa', 'e', 'f', 'g', 'aa', 'b', 'c', 'dd', 'dd']
# orig_y = ['aa', 'b', 'c', 'dd']
# print(getOverlappingSpan(x, orig_y))

# x, orig_y = ['aa', 'b', 'c', 'dd', 'x', 'y', 'z', 'aa', 'b', 'c', 'dd'], ['aa', 'b', 'c', 'dd']
# print(getOverlappingSpan(x, orig_y))

# x, orig_y = ['aa', 'b', 'c', 'dd', 'x', 'y', 'z', 'b', 'c', 'dd'], ['aa', 'b', 'c', 'dd']
# print(getOverlappingSpan(x, orig_y))

# x, orig_y = ['aa', 'e', 'f', 'g', 'aa', 'b', 'c', 'dd', 'dd'], ['aa', 'b', 'c', 'dd'] 
# print(getOverlappingSpan(x, orig_y))

# x, orig_y = ['aa', 'e', 'f', 'g', 'aa', 'b', 'c', 'dd', 'dd'], ['aa', 'b', 'dd'] 
# print(getOverlappingSpan(x, orig_y))

# x, orig_y = ['aa', 'e', 'f', 'g', 'aa', 'b', 'c', 'dd', 'dd'], ['b', 'b', 'aa'] 
# print(getOverlappingSpan(x, orig_y))

# x, orig_y = ['aa', 'e', 'f', 'g', 'aa', 'b', 'c', 'dd', 'dd'], ['k', 'k', 'a'] 
# print(getOverlappingSpan(x, orig_y))

# x, orig_y = ['e', 'f', 'g', 'b', 'c', 'dd', 'dd'], ['aa', 'b', 'dd'] 
# print(getOverlappingSpan(x, orig_y))




