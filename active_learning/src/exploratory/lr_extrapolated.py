import numpy as np

class Line(object):

    def __init__(self,coor1,coor2):
        self.coor1 = coor1
        self.coor2 = coor2

    @property
    def distance(self):
        x1,y1 = self.coor1
        x2,y2 = self.coor2
        return ((x2-x1)**2+(y2-y1)**2)**0.5    

    @property
    def slope(self):
        x1,y1 = self.coor1
        x2,y2 = self.coor2
        return (float(y2-y1))/(x2-x1)

    @property
    def yintercept(self):
        x, y = self.coor1
        return y - self.slope * x

if __name__ == "__main__":
    linear = False
    if linear:
        coor1 = (0, 0.1)
        coor2 = (7500000, 0.01)
        coor3 = (12500000, 0.001)
        l = Line(coor2, coor1)
        #l = Line(coor3, coor2)
        print l.distance
        print l.slope
        get_y = lambda x: x * l.slope + l.yintercept
        xs = range(coor1[0], coor2[0], 100000)
        #xs = range(coor2[0], coor3[0], 100000)
        ys = [get_y(x) for x in xs]
        for x, y in zip(xs, ys):
            print "\"{}\": {},".format(x, y)
    else:
        #CIFAR 10
        old_xs = [0, 7500000, 12500000]
        old_ys = [0.1, 0.01, 0.001] 
        xs = range(0, 12500000, 100000)

        #IMAGENET
        old_xs = [0, 8600700, 17201400]
        old_ys = [0.1, 0.01, 0.001] 
        xs = range(0, 17201400, 100000)

        get_y = lambda x: np.interp(x, old_xs, old_ys)
        ys = [get_y(x) for x in xs]
        for x, y in zip(xs, ys):
            print "\"{}\": {},".format(x, y)





