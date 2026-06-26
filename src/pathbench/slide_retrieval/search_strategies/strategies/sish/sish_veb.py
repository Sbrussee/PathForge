# The implementation of VEB tree is borrowed from:
# https://github.com/erikwaing/VEBTree
import math
import logging

logger = logging.getLogger(__name__)

class VEB:
    """van Emde Boas tree used by the SISH retrieval backend."""

    def high(self, x):
        return int(math.floor(x / math.sqrt(self.u)))

    def low(self, x):
        return int((x % math.ceil(math.sqrt(self.u))))

    def index(self, x, y):
        return int((x * math.floor(math.sqrt(self.u))) + y)

    def __init__(self, u):
        if u < 0:
            raise Exception("u cannot be less than 0 --- u = " + str(u))
        self.u = 2
        while self.u < u:
            self.u *= self.u
        self.min = None
        self.max = None
        if (u > 2):
            self.clusters = [None for i in range(self.high(self.u))]  # VEB(self.high(self.u))
            self.summary = None  # VEB(self.high(self.u))

    def member(self, x):
        if x == self.min or x == self.max:  # found it as the minimum or maximum
            return True
        elif self.u <= 2:					# has not found it in the "leaf"
            return False
        else:
            cluster = self.clusters[self.high(x)]
            if cluster is not None:
                return cluster.member(self.low(x))  # looks for it in the clusters inside
            else:
                return False

    def successor(self, x):
        if self.u <= 2:
            if x == 0 and self.max == 1:
                return 1
            else:
                return None
        elif self.min is not None and x < self.min: # x is less than everything in the tree, returns the minimum
            return self.min
        else:
            h = self.high(x)
            low_val = self.low(x)
            maxlow = None
            cluster = self.clusters[h]
            if cluster is not None:
                maxlow = cluster.max
            if maxlow is not None and low_val < maxlow:
                offset = cluster.successor(low_val)
                return self.index(h, offset)
            else:
                succcluster = None
                if self.summary is not None:
                    succcluster = self.summary.successor(h)
                if succcluster is None:
                    return None
                else:
                    cluster2 = self.clusters[succcluster]
                    offset = 0
                    if cluster2 is not None:
                        offset = cluster2.min
                    return self.index(succcluster, offset)

    def predecessor(self, x):
        if self.u <= 2:
            if x == 1 and self.min == 0:
                return 0
            else:
                return None
        elif self.max is not None and x > self.max:
            return self.max
        else:
            h = self.high(x)
            low_val = self.low(x)
            minlow = None
            cluster = self.clusters[h]
            if cluster is not None:
                minlow = cluster.min
            if minlow is not None and low_val > minlow:
                offset = cluster.predecessor(low_val)
                if offset is None:
                    offset = 0
                return self.index(h, offset)
            else:
                predcluster = None
                if self.summary is not None:
                    predcluster = self.summary.predecessor(h)
                if predcluster is None:
                    if self.min is not None and x > self.min:
                        return self.min
                    else:
                        return None
                else:
                    cluster2 = self.clusters[predcluster]
                    offset = 0
                    if cluster2 is not None:
                        offset = cluster2.max
                    return self.index(predcluster, offset)

    def emptyInsert(self, x):
        self.min = x
        self.max = x

    def insert(self, x):
        if self.min is None:
            self.emptyInsert(x)
        else:
            if x < self.min:
                temp = self.min
                self.min = x
                x = temp
            if self.u > 2:
                h = self.high(x)
                if self.clusters[h] is None:
                    self.clusters[h] = VEB(self.high(self.u))
                if self.summary is None:
                    self.summary = VEB(self.high(self.u))
                if self.clusters[h].min is None:
                    self.summary.insert(h)
                    self.clusters[h].emptyInsert(self.low(x))
                else:
                    self.clusters[h].insert(self.low(x))
            if x > self.max:
                self.max = x
