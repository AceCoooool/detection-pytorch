import numpy as np


class Visdom(object):
    def __init__(self, display_id=0, num=3):
        self.idx = display_id
        self.num = num
        self.win = dict()
        if display_id > 0:
            import visdom
            self.vis = visdom.Visdom(port=8097)

    def create_vis_line(self, xlabel, ylabel, title, legend, name):
        self.win[name] = self.vis.line(
            X=np.zeros((1, 3)),
            Y=np.zeros((1, 3)),
            opts=dict(xlabel=xlabel, ylabel=ylabel, title=title, legend=legend)
        )

    def update_vis_line(self, iter, loc, conf, name, update_type, epoch_size=1):
        self.vis.line(
            X=np.ones((1, 3)) * iter,
            Y=np.array([loc, conf, loc + conf]).reshape((1, 3)) / epoch_size,
            win=self.win[name],
            update=update_type
        )
        # initialize
        if iter == 0:
            self.vis.line(
                X=np.zeros((1, 3)),
                Y=np.array([loc, conf, loc + conf]).reshape((1, 3)),
                win=self.win[name],
                update=True
            )

    def create_vis_image(self, size, title, name):
        self.win[name] = self.vis.image(
            np.random.randn(*size),
            opts=dict(title=title)
        )

    def update_vis_image(self, image, name):
        self.vis.image(image, win=self.win[name])


if __name__ == '__main__':
    import cv2

    vis = Visdom(1)
    # vis_legend = ['Loc Loss', 'Conf Loss', 'Total Loss']
    # vis.create_vis_line('Iteration', 'Loss', 'SSD', vis_legend, 'iter')
    # vis.create_vis_line('Epoch', 'Loss', 'SSD', vis_legend, 'epoch')
    # for i in range(20):
    #     loc, conf = 10 + i, 20 + i
    #     vis.update_vis_line(i, loc, conf, 'iter', 'append')
    # # loc, conf = 10, 20
    # # vis.update_vis_line(0, loc, conf, 'iter', 'append')
    vis.create_vis_image((3, 64, 64), 'image', 'img')
    img = cv2.cvtColor(cv2.imread('/home/ace/Pictures/cat.jpg'), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (64, 64)).transpose((2, 0, 1))
    # print(img.shape)
    vis.update_vis_image(img, 'img')
