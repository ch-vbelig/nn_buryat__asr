class PhoneConverter:
    def __init__(self, path):
        self.fpath = path
        self.el_to_index = {"<SPACE>": 0}
        self.sil_token = 0
        self.index_to_el = {0: "<SPACE>"}
        self.n_elements = len(self.index_to_el)
        self._add_elements()

    def _add_elements(self):
        with open(self.fpath) as fp:
            lines = fp.readlines()
            phones = [p.strip() for p in lines]

            for phone in phones:
                self._add_element(phone)

    def _add_element(self, element):
        if element not in self.el_to_index:
            self.el_to_index[element] = self.n_elements
            self.index_to_el[self.n_elements] = element
            self.n_elements += 1
