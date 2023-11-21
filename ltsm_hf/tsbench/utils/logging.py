import os
import csv

class Logger(object):
    """ Logger saves the running results and helps make plots from the results
    """

    def __init__(self, log_dir):
        """ Initialize the labels, legend and paths of the plot and log file.

        Args:
            log_path (str): The path the log files
        """
        self.log_dir = log_dir

    def __enter__(self):
        self.txt_path = os.path.join(self.log_dir, 'log.txt')
        self.csv_path = os.path.join(self.log_dir, 'performance.csv')

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.txt_file = open(self.txt_path, 'w')
        self.csv_file = open(self.csv_path, 'w')
        fieldnames = ["epoch", "train_loss", "validation_loss", "time"]
        self.writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.writer.writeheader()

        return self

    def log(self, text):
        """ Write the text to log file then print it.
        Args:
            text(string): text to log
        """
        self.txt_file.write(text+'\n')
        self.txt_file.flush()
        print(text)

    def log_performance(self, epoch, train_loss, validation_loss, time):
        """ Log a point in the curve
        Args:
            episode (int): the episode of the current point
            reward (float): the reward of the current point
        """
        self.writer.writerow({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "time": time,
        })

    def __exit__(self, type, value, traceback):
        if self.txt_path is not None:
            self.txt_file.close()
        if self.csv_path is not None:
            self.csv_file.close()
