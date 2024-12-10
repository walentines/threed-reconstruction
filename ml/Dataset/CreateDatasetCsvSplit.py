import os
import random
import pandas as pd

class CreateDatasetCsvSplit:
    def __init__(self, root_path, validation_split, test_split, csv_root_name):
        self.root_path = root_path
        self.validation_split = validation_split
        self.test_split = test_split
        self.csv_root_name = csv_root_name
        self.train_split = 1.0 - (validation_split + test_split)
    
    def __dict_to_csv(self, dict, mode):
        df = pd.DataFrame(dict)
        df.to_csv(f'{self.csv_root_name}_{mode}.csv', index=False)

    def __compute_csv(self):
        dirs = os.listdir(self.root_path)
        train_csv = {'videoid': []}
        val_csv = {'videoid': []}
        test_csv = {'videoid': []}
        for dir in dirs:
            rand_number = random.random()
            if rand_number < self.train_split:
                train_csv['videoid'].append(dir)
            elif rand_number >= self.train_split and rand_number < self.train_split + self.validation_split:
                val_csv['videoid'].append(dir)
            else:
                test_csv['videoid'].append(dir)
        
        self.__dict_to_csv(train_csv, mode='train')
        self.__dict_to_csv(val_csv, mode='val')
        self.__dict_to_csv(test_csv, mode='test')
    
    def __call__(self):
        self.__compute_csv()

if __name__ == '__main__':
    csv_generator = CreateDatasetCsvSplit('/mnt/hddmount1/sei2clj/HQ339', 0.1, 0.1, 'csv')
    csv_generator()
