from pretokenization import pre_tokenize
from typing import Optional


class PreToken:
    def __init__(self, word: bytes, count: int):
        self.count = count
        self.word = word
        self.tokens = [word[i:i+1] for i in range(len(word))]
        #print(self.tokens)

class VocabList:
    def __init__(self, vocab: list[bytes]):
        self.vocab : list[bytes] = vocab
        self.vocab_set = set(vocab)

    def add(self, word: bytes) -> None:
        self.vocab_set.add(word)
        self.vocab.append(word)

    def get_vocab_dict(self) -> dict[int, bytes]:
        vocab_dict: dict[int, bytes] = {i: s for i, s in enumerate(self.vocab)}
        return vocab_dict

    def exists(self, word: bytes) -> bool:
        return word in self.vocab_set

def find_largest_token(count_dict: dict[bytes, int]) -> Optional[bytes]:
    if len(count_dict) == 0:
        return None
    #print(count_dict)
    res = max(count_dict.items(), key=lambda kv: (kv[1], kv[0]))[0]
    if count_dict[res] <= 0:
        return None
    return res


def decrease_cnt(token: bytes, val: int, count_dict: dict[bytes, int]):
    if token in count_dict:
        #print(f"token : {token},v = {val} before : {count_dict[token]}")
        count_dict[token] -= val
        #print(f"token : {token} after : {count_dict[token]}")
        if count_dict[token] <= 0:
            count_dict.pop(token)
        #print(count_dict)


def increase_cnt(token: bytes, pre_token: PreToken, count_dict: dict[bytes, int], token_idx: dict[bytes, set[PreToken]]):
    if token not in count_dict:
        count_dict[token] = 0
    count_dict[token] += pre_token.count
    if token not in token_idx:
        token_idx[token] = set()
    token_idx[token].add(pre_token)


def replace_iter(largest_token: bytes, pre_token: PreToken, count_dict: dict[bytes, int], token_idx: dict[bytes, set[PreToken]]):
    found = False
    for i in range(len(pre_token.tokens) -1):
        if largest_token != pre_token.tokens[i] + pre_token.tokens[i + 1]:
            continue
        if i > 0:
            new_token = pre_token.tokens[i - 1] + largest_token
            print(f"new token = {new_token}, pre = {pre_token.tokens[i - 1]}, largest = {largest_token}")
            decrease_cnt(pre_token.tokens[i - 1] + pre_token.tokens[i], pre_token.count, count_dict)
            increase_cnt(new_token, pre_token, count_dict, token_idx)
        if i < len(pre_token.tokens) - 2:
            new_token = largest_token + pre_token.tokens[i + 2]
            decrease_cnt(pre_token.tokens[i + 1] + pre_token.tokens[i + 2], pre_token.count, count_dict)
            increase_cnt(new_token, pre_token, count_dict, token_idx)
        #print(f"decreasing token {largest_token} by {pre_token.count}")
        decrease_cnt(largest_token, pre_token.count, count_dict)
        pre_token.tokens = pre_token.tokens[:i] + [pre_token.tokens[i] + pre_token.tokens[i+1]] + pre_token.tokens[i+2:]
        found = True
        break
    return found


def replace(largest_token: bytes, pre_token: PreToken, count_dict: dict[bytes, int],
                 token_idx: dict[bytes, set[PreToken]]):
    #print(f"replacing token : {pre_token.tokens}, largest_token = {largest_token}")
    while replace_iter(largest_token, pre_token, count_dict, token_idx):
        break



def merge(count_dict: dict[bytes, int], vocab: VocabList, token_idx: dict[bytes, set[PreToken]]) -> bool:
    largest_token = find_largest_token(count_dict)
    print(f"Merging vocab : {largest_token}")
    if largest_token is None:
        return False
    if vocab.exists(largest_token):
        if largest_token in count_dict:
            count_dict.pop(largest_token)
        return True
    print(f"Adding vocab : {largest_token} , type : {type(largest_token)}")
    vocab.add(largest_token)

    if largest_token not in token_idx:
        print(f"Error !!!!, largest_token : {largest_token} not found in token_idx")
        return True

    for pre_token in token_idx[largest_token]:
        replace(largest_token, pre_token, count_dict, token_idx)

    token_idx.pop(largest_token)
    if largest_token in count_dict:
        count_dict.pop(largest_token)
    return True


INITIAL_VOCAB_SIZE: int = 256


def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[
    dict[int, bytes], list[tuple[bytes, bytes]]]:
    merges: list[tuple[bytes, bytes]] = []
    vocab: list[bytes] = [bytes([i]) for i in range(INITIAL_VOCAB_SIZE)]

    vocab.extend([s.encode("utf-8") for s in special_tokens])

    vocab_list = VocabList(vocab)

    # print(f"vocab: {vocab}")

    pre_tokens = pre_tokenize(input_path, 4, special_tokens)
    token_idx: dict[bytes, set[PreToken]] = {}
    count_dict: dict[bytes, int] = {}

    idx = 0
    for k, v in pre_tokens.items():
        idx += 1
        print(f"parsing token {idx}th%: {k}: {v}")
        tokens = []
        k_bytes = k.encode("utf-8")
        if len(k_bytes) < 2:
            if not vocab_list.exists(k_bytes):
                print(f"!!!! Error one byte token {k_bytes} is not in vocab")
            continue

        for i in range(len(k_bytes) - 1):
            tokens.append(k_bytes[i:i + 2])
        # print(tokens)
        pre_token = PreToken(k_bytes, v)
        for token in tokens:
            if token not in token_idx:
                token_idx[token] = set()
            token_idx[token].add(pre_token)

            if token not in count_dict:
                count_dict[token] = 0
            count_dict[token] += v

    # print(count_dict)
    # for k, v in token_idx.items():
    #    print(f"{k} : {len(v)}")

    merge_count = vocab_size - INITIAL_VOCAB_SIZE - len(special_tokens)
    merge_count = max(merge_count, 0)
    print(f"LOG: start merging {merge_count} iterations")
    for i in range(merge_count):
        res = merge(count_dict, vocab_list, token_idx)
        if not res:
            print(f"Warning, merge completed before reaching the target number of vocab, current idx = {i}")

    vocab_dict = vocab_list.get_vocab_dict()
    print(vocab_dict)
    return vocab_dict, merges


if __name__ == "__main__":
    train_bpe("./tests/fixtures/tinystories_sample.txt", 400, ["<|endoftext|>"])
    #train_bpe("./tests/fixtures/simple_test.txt", 400, ["<|endoftext|>"])
