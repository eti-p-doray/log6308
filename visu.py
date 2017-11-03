# encoding: UTF-8
# Copyright 2016 Google.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#plt.style.use(["ggplot", "tensorflowvisu.mplstyle"])
#import matplotlib
#matplotlib.use('macosx') #this is the default on mac
#print("matplotlib version: " + matplotlib.__version__)
import matplotlib.animation as animation
from matplotlib import rcParams
import math
tf.set_random_seed(0)

# number of percentile slices for histogram visualisations
HISTOGRAM_BUCKETS = 7


# n = HISTOGRAM_BUCKETS (global)
# Buckets the data into n buckets so that there are an equal number of data points in
# each bucket. Returns n+1 bucket boundaries. Spreads the reaminder data.size % n more
# or less evenly among the central buckets.
# data: 1-D ndarray containing float data, MUST BE SORTED in ascending order
#    n: integer, the number of desired output buckets
# return value: ndarray, 1-D vector of size n+1 containing the bucket boundaries
#               the first value is the min of the data, the last value is the max
def probability_distribution(data):
    n = HISTOGRAM_BUCKETS
    data.sort()
    bucketsize = data.size // n
    bucketrem  = data.size % n
    buckets = np.zeros([n+1])
    buckets[0] = data[0]  # min
    buckets[-1] = data[-1]  # max
    buckn = 0
    rem = 0
    remn = 0
    k = 0
    cnt = 0 # only for assert
    lastval = data[0]
    for i in range(data.size):
        val = data[i]
        buckn += 1
        cnt += 1
        if buckn > bucketsize+rem : ## crossing bucket boundary
            cnt -= 1
            k += 1
            buckets[k] = (val + lastval) / 2
            if (k<n+1):
                cnt += 1
            buckn = 1 # val goes into the new bucket
            if k >= (n - bucketrem) // 2 and remn < bucketrem:
                rem = 1
                remn += 1
            else:
                rem = 0
        lastval = val
    assert i+1 == cnt
    return buckets

def _empty_collection(collection):
    tempcoll = []
    for a in (collection):
        tempcoll.append(a)
    for a in (tempcoll):
        collection.remove(a)

def _display_time_histogram(ax, xdata, ydata, color):
    _empty_collection(ax.collections)
    midl = HISTOGRAM_BUCKETS//2
    midh = HISTOGRAM_BUCKETS//2
    for i in range(int(math.ceil(HISTOGRAM_BUCKETS/2.0))):
        ax.fill_between(xdata, ydata[:,midl-i], ydata[:,midh+1+i], facecolor=color, alpha=1.6/HISTOGRAM_BUCKETS)
        if HISTOGRAM_BUCKETS % 2 == 0 and i == 0:
            ax.fill_between(xdata, ydata[:,midl-1], ydata[:,midh], facecolor=color, alpha=1.6/HISTOGRAM_BUCKETS)
            midl = midl-1

class NetflixDataVis:
    xmax = 0
    y2max = 0
    x1 = []
    y1 = []
    z1 = []
    x2 = []
    y2 = []
    z2 = []
    x3 = []
    w3 = np.zeros([0,HISTOGRAM_BUCKETS+1])
    b3 = np.zeros([0,HISTOGRAM_BUCKETS+1])
    im1 = np.full((28*10,28*10,3),255, dtype='uint8')
    im2 = np.full((28*10,28*10,3),255, dtype='uint8')
    _animpause = False
    _animation = None
    _mpl_figure = None
    _mlp_init_func = None
    _mpl_update_func = None
    _color4 = None
    _color5 = None

    def __set_title(self, ax, title, default=""):
        if title is not None and title != "":
            ax.set_title(title, y=1.02) # adjustment for plot title bottom margin
        else:
            ax.set_title(default, y=1.02) # adjustment for plot title bottom margin

    #  retrieve the color from the color cycle, default is 1
    def __get_histogram_cyclecolor(self, colornum):
        clist = rcParams['axes.prop_cycle']
        ccount = 1 if (colornum is None) else colornum
        colors = clist.by_key()['color']
        for i, c in enumerate(colors):
            if (i == ccount % 3):
                return c

    def __init__(self, title1=None, title2=None, title3=None, title4=None, title5=None, title6=None, histogram4colornum=None, histogram5colornum=None, dpi=70):
        self._color4 = self.__get_histogram_cyclecolor(histogram4colornum)
        self._color5 = self.__get_histogram_cyclecolor(histogram5colornum)
        fig = plt.figure(figsize=(19.20, 10.80), dpi=dpi)
        plt.gcf().canvas.set_window_title("MNIST")
        fig.set_facecolor('#FFFFFF')
        ax1 = fig.add_subplot(231)
        ax2 = fig.add_subplot(232)
        ax3 = fig.add_subplot(233)
        ax4 = fig.add_subplot(234)
        ax5 = fig.add_subplot(235)
        ax6 = fig.add_subplot(236)

        self.__set_title(ax1, title1, default="Accuracy")
        self.__set_title(ax2, title2, default="Cross entropy loss")
        self.__set_title(ax3, title3, default="Training digits")
        self.__set_title(ax4, title4, default="Weights")
        self.__set_title(ax5, title5, default="Biases")
        self.__set_title(ax6, title6, default="Test digits")

        #ax1.set_figaspect(1.0)

        # TODO: finish exporting the style modifications into a stylesheet
        line1, = ax1.plot(self.x1, self.y1, label="training accuracy")
        line2, = ax1.plot(self.x2, self.y2, label="test accuracy")
        legend = ax1.legend(loc='lower right') # fancybox : slightly rounded corners
        legend.draggable(True)

        line3, = ax2.plot(self.x1, self.z1, label="training loss")
        line4, = ax2.plot(self.x2, self.z2, label="test loss")
        legend = ax2.legend(loc='upper right') # fancybox : slightly rounded corners
        legend.draggable(True)

        ax3.grid(False) # toggle grid off
        ax3.set_axis_off()
        imax1 = ax3.imshow(self.im1, animated=True, cmap='binary', vmin=0.0, vmax=1.0, interpolation='nearest', aspect=1.0)

        ax6.grid(False) # toggle grid off
        ax6.axes.get_xaxis().set_visible(False)
        imax2 = ax6.imshow(self.im2, animated=True, cmap='binary', vmin=0.0, vmax=1.0, interpolation='nearest', aspect=1.0)
        ax6.locator_params(axis='y', nbins=7)
        # hack...
        ax6.set_yticks([0, 280-4*56, 280-3*56, 280-2*56, 280-56, 280])
        ax6.set_yticklabels(["100%", "98%", "96%", "94%", "92%", "90%"])

        def _init():
            ax1.set_xlim(0, 10)  # initial value only, autoscaled after that
            ax2.set_xlim(0, 10)  # initial value only, autoscaled after that
            ax4.set_xlim(0, 10)  # initial value only, autoscaled after that
            ax5.set_xlim(0, 10)  # initial value only, autoscaled after that
            ax1.set_ylim(0, 1)    # important: not autoscaled
            #ax1.autoscale(axis='y')
            ax2.set_ylim(0, 2)  # important: not autoscaled
            return imax1, imax2, line1, line2, line3, line4


        def _update():
            # x scale: iterations
            ax1.set_xlim(0, self.xmax+1)
            ax2.set_xlim(0, self.xmax+1)
            ax4.set_xlim(0, self.xmax+1)
            ax5.set_xlim(0, self.xmax+1)

            # four curves: train and test accuracy, train and test loss
            line1.set_data(self.x1, self.y1)
            line2.set_data(self.x2, self.y2)
            line3.set_data(self.x1, self.z1)
            line4.set_data(self.x2, self.z2)

            #images
            imax1.set_data(self.im1)
            imax2.set_data(self.im2)

            # histograms
            _display_time_histogram(ax4, self.x3, self.w3, self._color4)
            _display_time_histogram(ax5, self.x3, self.b3, self._color5)

            #return changed artists
            return imax1, imax2, line1, line2, line3, line4

        def _key_event_handler(event):
            if len(event.key) == 0:
                return
            else:
                keycode = event.key

            # pause/resume with space bar
            if keycode == ' ':
                self._animpause = not self._animpause
                if not self._animpause:
                    _update()
                return

            # [p, m, n] p is the #of the subplot, [n,m] is the subplot layout
            toggles = {'1':[1,1,1], # one plot
                       '2':[2,1,1], # one plot
                       '3':[3,1,1], # one plot
                       '4':[4,1,1], # one plot
                       '5':[5,1,1], # one plot
                       '6':[6,1,1], # one plot
                       '7':[12,1,2], # two plots
                       '8':[45,1,2], # two plots
                       '9':[36,1,2], # two plots
                       'escape':[123456,2,3], # six plots
                       '0':[123456,2,3]} # six plots

            # other matplotlib keyboard shortcuts:
            # 'o' box zoom
            # 'p' mouse pan and zoom
            # 'h' or 'home' reset
            # 's' save
            # 'g' toggle grid (when mouse is over a plot)
            # 'k' toggle log/lin x axis
            # 'l' toggle log/lin y axis

            if not (keycode in toggles):
                return

            for i in range(6):
                fig.axes[i].set_visible(False)

            fignum = toggles[keycode][0]
            if fignum <= 6:
                fig.axes[fignum-1].set_visible(True)
                fig.axes[fignum-1].change_geometry(toggles[keycode][1], toggles[keycode][2], 1)
                ax6.set_aspect(25.0/40) # special case for test digits
            elif fignum < 100:
                fig.axes[fignum//10-1].set_visible(True)
                fig.axes[fignum//10-1].change_geometry(toggles[keycode][1], toggles[keycode][2], 1)
                fig.axes[fignum%10-1].set_visible(True)
                fig.axes[fignum%10-1].change_geometry(toggles[keycode][1], toggles[keycode][2], 2)
                ax6.set_aspect(1.0) # special case for test digits
            elif fignum == 123456:
                for i in range(6):
                    fig.axes[i].set_visible(True)
                    fig.axes[i].change_geometry(toggles[keycode][1], toggles[keycode][2], i+1)
                ax6.set_aspect(1.0) # special case for test digits

            plt.draw()

        fig.canvas.mpl_connect('key_press_event', _key_event_handler)

        self._mpl_figure = fig
        self._mlp_init_func = _init
        self._mpl_update_func = _update

    def _update_xmax(self, x):
        if (x > self.xmax):
            self.xmax = x

    def _update_y2max(self, y):
        if (y > self.y2max):
            self.y2max = y

    def append_training_curves_data(self, x, accuracy, loss):
        self.x1.append(x)
        self.y1.append(accuracy)
        self.z1.append(loss)
        self._update_xmax(x)

    def append_test_curves_data(self, x, accuracy, loss):
        self.x2.append(x)
        self.y2.append(accuracy)
        self.z2.append(loss)
        self._update_xmax(x)
        self._update_y2max(accuracy)

    def get_max_test_accuracy(self):
        return self.y2max

    def append_data_histograms(self, x, datavect1, datavect2, title1=None, title2=None):
        self.x3.append(x)
        datavect1.sort()
        self.w3 = np.concatenate((self.w3, np.expand_dims(probability_distribution(datavect1), 0)))
        datavect2.sort()
        self.b3 = np.concatenate((self.b3, np.expand_dims(probability_distribution(datavect2), 0)))
        self._update_xmax(x)

    def update_image1(self, im):
        self.im1 = im

    def update_image2(self, im):
        self.im2 = im

    def is_paused(self):
        return self._animpause

    def animate(self, compute_step, iterations, train_data_update_freq=20, test_data_update_freq=100, one_test_at_start=True, more_tests_at_start=False, save_movie=False):

        def animate_step(i):
            if (i == iterations // train_data_update_freq): #last iteration
                compute_step(iterations, True, True)
            else:
                for k in range(train_data_update_freq):
                    n = i * train_data_update_freq + k
                    request_data_update = (n % train_data_update_freq == 0)
                    request_test_data_update = (n % test_data_update_freq == 0) and (n > 0 or one_test_at_start)
                    if more_tests_at_start and n < test_data_update_freq: request_test_data_update = request_data_update
                    compute_step(n, request_test_data_update, request_data_update)
                    # makes the UI a little more responsive
                    plt.pause(0.001)
            if not self.is_paused():
                return self._mpl_update_func()

        self._animation = animation.FuncAnimation(self._mpl_figure, animate_step, int(iterations // train_data_update_freq + 1), init_func=self._mlp_init_func, interval=16, repeat=False, blit=False)

        if save_movie:
            mywriter = animation.FFMpegWriter(fps=24, codec='libx264', extra_args=['-pix_fmt', 'yuv420p', '-profile:v', 'high', '-tune', 'animation', '-crf', '18'])
            self._animation.save("./tensorflowvisu_video.mp4", writer=mywriter)
        else:
            plt.show(block=True)
