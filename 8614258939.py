import numpy as np
import struct
import sys
# import time

class MNIST:
    """
    provide a manipulation interface of MNIST data:
        load, columnize, split, show
    """
    def __init__(self, img_file_name, lbl_file_name, D, N):
        ## file paths
        self._img_file_name = img_file_name
        self._lable_file_name = lbl_file_name
        ## raw data
        self._N = N
        self._D = D
        self._Num = 1000
        self._imgs = self.load_imgs(self._img_file_name)
        self._lables = self.lable_read(self._lable_file_name)
        ## processed data
        [self._train_imgs, self._test_imgs, self._train_lables, self._test_lables] = \
            self.split_raw_data()
        [self._transformed_trian_imgs, self._transformed_test_imgs] = self.pca_transform()

    def load_imgs(self, file_name):
        ## read bin-file
        f = open(file_name, 'rb')
        buff = f.read(16 + 784 * self._Num);
        f.close()
        ## load headers & prepare struct
        offset = 0
        magic, imageNum, rows, cols = struct.unpack_from('>IIII', buff, offset)
        offset += struct.calcsize('>IIII')
        images = np.empty((self._Num, rows * cols))
        image_size = rows * cols
        fmt = '>' + str(image_size) + 'B'
        ## load imgs
        for i in range(self._Num):
            images[i] = np.array(struct.unpack_from(fmt, buff, offset))  ## .reshape(rows,cols)
            offset += struct.calcsize(fmt)
        ## return img list
        return np.array(images[:self._Num], dtype=int)

    def lable_read(self, file_name):
        ## read bin-file
        f = open(file_name, 'rb')
        buff = f.read(8 + 1 * self._Num)
        f.close()
        ## load headers & prepare struct
        offset = 0
        magic, lableNum = struct.unpack_from('>II', buff, offset)
        offset += struct.calcsize('>II')
        lables = np.empty(self._Num)
        fmt = '>' + str(1) + 'B'
        ## load lables
        for i in range(self._Num):
            lables[i] = np.array(struct.unpack_from(fmt, buff, offset))
            offset += struct.calcsize(fmt)
        ## return
        return np.array([int(e) for e in lables][:self._Num], dtype=int)

    def split_raw_data(self):
        # return [self._imgs[self._N : ], self._imgs[ : self._N],
        #         self._lables[self._N : ], self._lables[ : self._N]]
        return [self._imgs[int(self._N):], self._imgs[: int(self._N)],
                self._lables[int(self._N):], self._lables[: int(self._N)]]

    def pca_transform(self):
        from sklearn.decomposition import PCA
        pca = PCA(n_components=self._D, svd_solver='full')
        pca.fit(self._train_imgs)
        return [pca.transform(self._train_imgs), pca.transform(self._test_imgs)]

    #     def img_show(self, index):
    #         import matplotlib.pyplot as plt
    #         plt.imshow(self._imgs[index].reshape(28, 28), cmap="Greys")
    #         print("index = {}, label = {}".format(index, self._lables[index]))

    def get_data(self):
        return [self._imgs, self._lables]

    def get_transformed_data(self):
        return [self._transformed_trian_imgs, self._transformed_test_imgs,
                self._train_lables, self._test_lables]


class KNN:
    """
    provide a KNN training and data manipulating interface
    """
    def __init__(self, train_img, test_img, train_lable, test_lable, K, N):
        self._Num = 1000
        self._K = K
        self._N = N
        self._train_img = train_img
        self._test_img = test_img
        self._train_lable = train_lable
        self._test_lable = test_lable

    def train():
        return

    def predict(self, print_acc, print_res):
        ## find neighbors
        dist = [[np.sqrt(np.sum(np.square(train_img[index] - test_point))) for index in range(self._Num - self._N)] for test_point in test_img]
        neighbors = [np.argsort(dist_line)[ : self._K] for dist_line in dist]
        ## vote
        votes = np.zeros([self._N, 10])
        for i in range(self._N):
            for j in range(self._K):
                neighbor_ind = neighbors[i][j]
                lable = train_lable[neighbor_ind]
                votes[i][lable] = votes[i][lable] + (1 / dist[i][neighbor_ind])
        votes = np.array([np.argsort(votes_line)[-1] for votes_line in votes])
        ## calculate accuracy
        correction = sum(votes == test_lable)
        acc = correction / self._N
        res = [[votes[i], test_lable[i]] for i in range(self._N)]
        np.savetxt("8614258939.txt", res, fmt="%d %d");
        if print_acc:
            print("_> acc={}".format(acc) + "(" + "{}/{}".format(correction, self._N) + ")")
            print("_> error:")
            for i in range(len(res)):
                pair = res[i]
                if pair[0] != pair[1]:
                    print("    {}: {}-{}".format(i, pair[0], pair[1]))
        if print_res:
            print("--- res ---")
            for i in range(self._N):
                print("    {} {}".format(votes[i], test_lable[i]))



"""
--- main ---
"""
# start = time.clock()

argv = sys.argv
K = int(argv[1])
D = int(argv[2])
N = int(argv[3])
mnist_path = argv[4] + "/"

img_path = mnist_path + "train-images.idx3-ubyte"
lbl_path = mnist_path + "train-labels.idx1-ubyte"
mnist = MNIST(img_path, lbl_path, D, N)
[train_img, test_img, train_lable, test_lable] = mnist.get_transformed_data()
knn = KNN(train_img, test_img, train_lable, test_lable, K, N)
knn.predict(print_acc=True, print_res=False)

# end = time.clock()
# print("time consume: {}".format(end - start))
