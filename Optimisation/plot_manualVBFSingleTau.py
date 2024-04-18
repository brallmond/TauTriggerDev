import pandas as pd
import json
import numpy as np

def plot(xaxis, yaxis, parameters, x_title, y_title, min_x, max_x, min_y, max_y, save_path,
        data=None, params_to_mark=[], plot_text=False):
    import matplotlib
    matplotlib.use("PDF")
    from matplotlib import pyplot as plt
    plt.figure()
    ax = plt.subplot()
    try:
        ax = data.plot(x=xaxis, y=yaxis, kind="scatter", ax=ax)
        if min_x and max_x:
           ax.set_xlim(min_x, max_x)
        if min_y and max_y:
           ax.set_ylim(min_y, max_y)
    except:
        plt.scatter(xaxis, yaxis, marker="o")
    for (x, y, label) in zip(data[xaxis], data[yaxis], parameters):
        if label in params_to_mark:
            plt.scatter(x, y, marker="o", color="r")
        if not plot_text and label not in params_to_mark:
            continue
        plt.annotate(label, # this is the text
             (x, y), # this is the point to label
             textcoords="offset points", # how to position the text
             xytext=(0, 10), # distance from text to points (x,y)
             ha='center', # horizontal alignment can be left, right or center
             size=5)
    plt.xlabel(x_title)
    plt.ylabel(y_title)

    x_text=0.05
    y_text=0.9
    plt.text(x_text, 1.02, "CMS", fontsize='large', fontweight='bold',
        transform=ax.transAxes)
    upper_text = "private work"
    plt.text(x_text + 0.1, 1.02, upper_text, transform=ax.transAxes)
    # text = [self.dataset.process.label.latex, self.category.label.latex]
    # for t in text:
        # plt.text(x_text, y_text, t, transform=ax.transAxes)
        # y_text -= 0.05

    plt.savefig(save_path)
    plt.close('all')

if __name__ == '__main__':

    df = pd.read_pickle("/afs/cern.ch/work/b/ballmond/public/TauTriggerDev/results_VBFSingleTau.pickle")   

    print(df)

    plot("rate", "eff", df["params"], "Rate (Hz)", "Efficiency", 0, 100, 0., 1., "plot_VBFSingleTau.pdf", data=df, params_to_mark=[["deeptau"], [0.4, 0.3]])
