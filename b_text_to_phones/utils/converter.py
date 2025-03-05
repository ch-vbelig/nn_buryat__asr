class Converter:
    def __init__(self, alphabet_path=None):
        self.el_to_index = {}
        self.index_to_el = {0: "SOS", 1: "EOS"}
        self.n_elements = 2
        if alphabet_path:
            self.alpha_path = alphabet_path
            self.addElementsFromAlphabet()

    def addElement(self, element):
        if element not in self.el_to_index:
            self.el_to_index[element] = self.n_elements
            self.index_to_el[self.n_elements] = element
            self.n_elements += 1

    def addElementsFromAlphabet(self):
        if not self.alpha_path:
            raise RuntimeError('No value provided for self.alpha_path')

        with open(self.alpha_path, encoding='utf-8') as fp:
            lines = fp.readlines()
            elements = [line.strip() for line in lines]

            for el in elements:
                self.addElement(el)
