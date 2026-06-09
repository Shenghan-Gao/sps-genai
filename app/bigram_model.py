import random
from collections import defaultdict


class BigramModel:
    def __init__(self, corpus):
        self.bigram_dict = defaultdict(list)
        self.build_model(corpus)

    def build_model(self, corpus):
        for sentence in corpus:
            words = sentence.lower().split()
            for i in range(len(words) - 1):
                self.bigram_dict[words[i]].append(words[i + 1])

    def generate_text(self, start_word, length):
        current_word = start_word.lower()
        generated_words = [current_word]

        for _ in range(length - 1):
            next_words = self.bigram_dict.get(current_word)

            if not next_words:
                break

            current_word = random.choice(next_words)
            generated_words.append(current_word)

        return " ".join(generated_words)