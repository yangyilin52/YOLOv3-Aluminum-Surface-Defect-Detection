import matplotlib.pyplot

class LineChart:
    def __init__(self, title, xLabel, yLabel):
        self.figure = matplotlib.pyplot.figure(figsize = (8, 6), dpi = 144)
        self.ax = self.figure.add_subplot(1, 1, 1)

        self.title = title
        self.xLabel = xLabel
        self.yLabel = yLabel
        self.data = {}

    def addLine(self, lineLabel):
        if lineLabel not in self.data.keys():
            self.data[lineLabel] = []

    def addData(self, lineLabel, data):
        if lineLabel in self.data.keys():
            self.data[lineLabel].append(data)

    def clearChart(self):
        self.data = {}

    def saveAsImage(self, filepath):
        self.ax.set_title(self.title)
        self.ax.set_xlabel(self.xLabel)
        self.ax.set_ylabel(self.yLabel)
        for k, v in self.data.items():
            xList = list(range(1, len(v) + 1))
            self.ax.plot(xList, v, label = k)
        if len(self.data) > 0:
            self.ax.legend()
        self.figure.savefig(filepath)
        self.ax.clear()
