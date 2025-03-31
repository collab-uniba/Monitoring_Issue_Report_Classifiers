class LabelMapper:
    """
    Maintains consistent label mapping across training and testing.
    Labels are sorted alphabetically and mapped to sequential integers.
    """
    def __init__(self, label_set=None):
        if label_set:
            sorted_labels = sorted(label_set)
            self.label_to_id = {label: idx for idx, label in enumerate(sorted_labels)}
            self.id_to_label = dict(enumerate(sorted_labels))
        else:
            self.label_to_id = {}
            self.id_to_label = {}
    
    def map_labels(self, labels):
        return labels.map(self.label_to_id)
    
    def inverse_map(self, ids):
        return [self.id_to_label[id] for id in ids]
    
    @property
    def num_labels(self):
        return len(self.label_to_id)