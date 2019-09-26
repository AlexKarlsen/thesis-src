import pandas as pd
import os

class Logger():
    def __init__(self, name, n_exits, log_path='logging'):
        self.log_path = log_path
        self.name = name
        self.cols = []
        for v in ['train', 'test']:
            for u in ['loss', 'accuracy', 'time']:
                self.cols +=  ['exit-'+ str(index)+ '-' + v + '-' + u  for index in range(n_exits)]
        self.DataFrame = pd.DataFrame(columns=self.cols)

    def log(self, train_loss, train_acc, train_time, test_loss, test_acc, test_time):
        data = []
        data = train_loss + train_acc + train_time + test_loss + test_acc + test_time
        data_dict = dict(zip(self.cols, data))

        # continously log results
        self.DataFrame = self.DataFrame.append(data_dict, ignore_index=True)
        self.DataFrame.to_csv(os.path.join(self.log_path, self.name + '.csv'))