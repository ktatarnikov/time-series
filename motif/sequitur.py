
class Rule:
    def __init__(self, token, production):
        self.token = token
        self.production = production
        middle = production.find(":")
        self.left = production[0:middle]
        self.right = production[middle + 1:len(production)]
        self.positions = []

    def occurence_count(self):
        return len(self.positions)

    def add_occurence(self, start, stop):
        self.positions.append((start, stop))

    def get_positions(self):
        return self.positions

    def to_string(self):
        return f"{self.token} -> {self.production} : {len(self.positions)}"

class Terminal:
    def __init__(self, digram_index, sym, pos, prev = None, next = None):
        self.digram_index = digram_index
        self.sym = sym
        self.prev = prev
        self.next = next
        self.pos = pos
        if prev is not None:
            prev.next = self
        if next is not None:
            next.prev = self

    def digram_with_prev(self):
        if self.prev is not None:
            return self.prev.sym + ":" + self.sym, self.prev
        return None, None

    def replace_with_rule(self, rule_sym):
        digrams_to_check_next = self._relink_next(rule_sym)
        digrams_to_check_prev = self._relink_prev(rule_sym)
        return digrams_to_check_next + digrams_to_check_prev

    def _relink_next(self, rule_sym):
        result = []
        if self.next is not None:
            removed = self.next
            digram_to_remove = self.next.join_with_next()
            if digram_to_remove is not None:
                digram_to_add = rule_sym.sym + ":" + self.next.next.sym
                self.digram_index._unindex_digram(digram_to_remove, removed)
                self.digram_index._index_digram(digram_to_add, rule_sym)
                result.append(digram_to_add)
                rule_sym.next = self.next.next
                self.next.next.prev = rule_sym
        return result

    def _relink_prev(self, rule_sym):
        result = []
        if self.prev is not None:
            digram_to_remove = self.prev.join_with_next()
            if digram_to_remove is not None:
                digram_to_add = self.prev.sym + ":" + rule_sym.sym
                self.digram_index._unindex_digram(digram_to_remove, self.prev)
                self.digram_index._index_digram(digram_to_add, self.prev)
                result.append(digram_to_add)
                rule_sym.prev = self.prev
                self.prev.next = rule_sym
        return result

    def join_with_previous(self):
        if self.prev is not None:
            return self.prev.sym + ":" + self.sym
        return None

    def join_with_next(self):
        if self.next is not None:
            return self.sym + ":" + self.next.sym
        return None

    def __repr__(self):
        return self.sym

    def sequence(self):
        next_sequence = self.next.sequence() if self.next is not None else ""
        return f"{self.sym}{next_sequence}"

    def get_start_stop_positions(self):
        start = self.pos
        stop = self.next.pos if self.next is not None else -1
        return start, stop

class DigramIndex:
    def __init__(self):
        self.digrams = dict()

    def get_occurences(self, digram_sym):
        return self.digrams[digram_sym]

    def keys(self):
        return self.digrams.keys()

    def get_index(self):
        return self.digrams

    def _index_digram(self, digram_sym, symbol):
        if digram_sym not in self.digrams:
            self.digrams[digram_sym] = []
        self.digrams[digram_sym].append(symbol)
        return self.digrams[digram_sym]

    def _unindex_digram(self, digram_sym, symbol):
        if digram_sym in self.digrams:
            digram_pos = self.digrams[digram_sym]
            digram_pos.remove(symbol)
            if len(digram_pos) == 0:
                del self.digrams[digram_sym]

class SymbolIndex:
    def __init__(self):
        self.symbols = dict()

    def get_positions(self, sym):
        return self.symbols[sym]

    def keys(self):
        return self.symbols.keys()

    def add_symbol(self, sym, pos):
        if sym not in self.symbols:
            self.symbols[sym] = []
        self.symbols[sym].append(pos)

class Grammar:
    def __init__(self):
        self.rules = dict()
        self.rules_by_token = dict()
        self.current_token = 0

    def rule_exists(self, digram_sym):
        return True if digram_sym in self.rules else False

    def keys(self):
        return self.rules.keys()

    def get_rule(self, digram_sym):
        return self.rules[digram_sym]

    def get_rule_by_token(self, token):
        return self.rules_by_token[token]

    def get_rules(self):
        return [rule.to_string() for rule in self.rules.values()]

    def _insert_new_rule(self, digram_sym):
        if digram_sym not in self.rules:
            token = self._next_token()
            rule = Rule(token, digram_sym)
            self.rules[digram_sym] = rule
            self.rules_by_token[token] = rule
        return self.rules[digram_sym]

    def _next_token(self):
        self.current_token += 1
        return f"R{self.current_token:#04}"

class Sequence:
    def __init__(self, digram_index, root):
        self.digram_index = digram_index
        self.root = root
        self.last = root

    def add_terminal(self, sym, pos):
        self.last = Terminal(digram_index = self.digram_index, sym = sym, prev = self.last, pos = pos)
        digram_sym, begin_symbol = self.last.digram_with_prev()
        self.digram_index._index_digram(digram_sym, begin_symbol)
        return digram_sym, begin_symbol

    def replace_with_rule(self, symbol, digram_sym, rule):
        start, _ = symbol.get_start_stop_positions()
        rule_sym = Terminal(digram_index = self.digram_index, sym = rule.token, pos = start)
        removed_digrams = symbol.replace_with_rule(rule_sym)
        if symbol == self.root:
            self.root = rule_sym
        if symbol.next == self.last:
            self.last = rule_sym
        return removed_digrams

class Sequitur:
    def __init__(self):
        self.digrams = DigramIndex()
        self.rules = Grammar()
        self.symbols = SymbolIndex()

    def get_rules(self):
        return self.rules.get_rules()

    def get_tokens(self):
        return self.rules.keys()

    def get_symbols(self):
        return self.symbols

    def get_grammar(self):
        return self.rules

    def get_digrams(self):
        return self.digrams.keys()

    def get_digrams_and_occurences(self):
        return self.digrams.get_index()

    def induce(self, word: list):
        self.symbols.add_symbol(word[0], 0)
        root = Terminal(digram_index = self.digrams, sym = word[0], pos = 0)
        seq = Sequence(digram_index = self.digrams, root = root)
        word_i = 1
        prev = word[0]
        while word_i < len(word):
            sym = word[word_i]
            if prev == sym:
                word_i += 1
                continue
            self.symbols.add_symbol(sym, word_i)
            digram_sym, begin_symbol = seq.add_terminal(sym, word_i)
            all_occurences = self.digrams.get_occurences(digram_sym)
            if self.rules.rule_exists(digram_sym):
                self.replace_terminal_with_rule(begin_symbol, digram_sym, seq)
            else:
                self.keep_digram_invariant(digram_sym, seq)
            word_i += 1
        # print(f"last S: {seq.root.sequence()}")

    def keep_digram_invariant(self, digram_sym, seq):
        occurences = self.digrams.get_occurences(digram_sym)
        if len(occurences) > 1 and not self.rules.rule_exists(digram_sym):
            rule = self.rules._insert_new_rule(digram_sym)
            for symbol in occurences:
                self.replace_terminal_with_rule(symbol, digram_sym, seq)

    def replace_terminal_with_rule(self, symbol, digram_sym, seq):
        rule = self.rules.get_rule(digram_sym)
        digrams_to_check = seq.replace_with_rule(symbol, digram_sym, rule)
        start, stop = symbol.get_start_stop_positions()
        rule.add_occurence(start, stop)
        for digram in digrams_to_check:
            occurences = self.digrams.get_occurences(digram)
            if len(occurences) > 1:
                self.keep_digram_invariant(digram, seq)

class SequiturError(Exception):
    pass
