from collections import defaultdict
import pprint

# Input data
sentences = [
    "The_DET cat_NOUN sleeps_VERB",
    "A_DET dog_NOUN barks_VERB",
    "The_DET dog_NOUN sleeps_VERB",
    "My_DET dog_NOUN runs_VERB fast_ADV",
    "A_DET cat_NOUN meows_VERB loudly_ADV",
    "Your_DET cat_NOUN runs_VERB",
    "The_DET bird_NOUN sings_VERB sweetly_ADV",
    "A_DET bird_NOUN chirps_VERB"
]

# Initialize
transition_counts = defaultdict(lambda: defaultdict(int))
emission_counts = defaultdict(lambda: defaultdict(int))
tag_counts = defaultdict(int)

# Training
for sentence in sentences:
    tokens = sentence.split()
    tags = []
    words = []

    for token in tokens:
        word, tag = token.rsplit("_", 1)
        words.append(word)
        tags.append(tag)
        emission_counts[tag][word] += 1
        tag_counts[tag] += 1

    prev_tag = "START"
    for tag in tags:
        transition_counts[prev_tag][tag] += 1
        prev_tag = tag
    transition_counts[prev_tag]["END"] += 1

# Normalize to probabilities
def normalize_counts(counts):
    probs = {}
    for prev in counts:
        total = sum(counts[prev].values())
        probs[prev] = {curr: count / total for curr, count in counts[prev].items()}
    return probs

transition_probs = normalize_counts(transition_counts)
emission_probs = normalize_counts(emission_counts)

# Tag set from training
tag_set = list(emission_counts.keys())

# Viterbi algorithm
def viterbi(sentence, transition_probs, emission_probs, tag_set):
    viterbi = [{}]
    backpointer = [{}]

    # Initialization
    for tag in tag_set:
        if tag in transition_probs['START'] and sentence[0] in emission_probs.get(tag, {}):
            viterbi[0][tag] = transition_probs['START'][tag] * emission_probs[tag][sentence[0]]
            backpointer[0][tag] = 'START'

    # Recursion
    for t in range(1, len(sentence)):
        viterbi.append({})
        backpointer.append({})
        for curr_tag in tag_set:
            if sentence[t] in emission_probs.get(curr_tag, {}):
                max_prob = 0
                best_prev = None
                for prev_tag in viterbi[t-1]:
                    trans = transition_probs.get(prev_tag, {}).get(curr_tag, 0)
                    emit = emission_probs[curr_tag].get(sentence[t], 0)
                    prob = viterbi[t-1][prev_tag] * trans * emit
                    if prob > max_prob:
                        max_prob = prob
                        best_prev = prev_tag
                if best_prev:
                    viterbi[t][curr_tag] = max_prob
                    backpointer[t][curr_tag] = best_prev

    # Termination
    last_probs = viterbi[-1]
    if not last_probs:
        return "No valid tag sequence found."
    best_last_tag = max(last_probs, key=last_probs.get)
    best_path = [best_last_tag]

    for t in range(len(sentence) - 1, 0, -1):
        best_path.insert(0, backpointer[t][best_path[0]])

    return list(zip(sentence, best_path))

# Probabilities
print("Transition Probabilities (T):")
pprint.pprint({k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in transition_probs.items()})
print("\nEmission Probabilities (E):")
pprint.pprint({k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in emission_probs.items()})

# User input
user_input = input("\nEnter a sentence (words separated by space, must match training vocab): ").strip()
words = user_input.split()

# Tag the sentence
result = viterbi(words, transition_probs, emission_probs, tag_set)
print("\nTagged Sentence:")
print(result)
