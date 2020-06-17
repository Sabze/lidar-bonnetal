class TrainTracker:

    def __init__(self):
        self.counter = 0
        self.logfiles = []

    def update(self, logfile):
        self.logfiles.append(logfile)
        self.counter += 1