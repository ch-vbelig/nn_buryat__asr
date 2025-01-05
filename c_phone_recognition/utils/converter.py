class PhoneConverter:

    def __init__(self, fpath):
        """
        :attr self.fpath (str): path to the phone_set.txt file, each line contains a single phone (space token excluded)
        :attr self.sil_idx (int): index of <SIL> token
        :attr self.space_idx (int): index of <SPACE> token
        :attr self.phone_to_index (dict): phone-to-index dictionary
        :attr self.index_to_phone (dict): index-to-phone dictionary
        :attr self.n_elements (int): number of elements in dictionary
        """
        self.fpath = fpath
        self.sil_idx = 0
        self.space_idx = 1
        self.phone_to_index = {"<SIL>": 0, "<SPACE>": 1}
        self.index_to_phone = {0: "<SIL>", 1: "<SPACE>"}
        self.n_elements = len(self.index_to_phone)

        self._add_elements()

    def _add_elements(self):
        with open(self.fpath) as fp:
            lines = fp.readlines()
            phones = [p.strip() for p in lines]

            for phone in phones:
                self._add_element(phone)

    def _add_element(self, element):
        if element not in self.phone_to_index:
            self.phone_to_index[element] = self.n_elements
            self.index_to_phone[self.n_elements] = element
            self.n_elements += 1
