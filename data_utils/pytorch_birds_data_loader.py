import torch
import numpy as np

from few_shot_cub_data_generator import FewshotBirdsDataGenerator

class PytorchBirdsDataLoader(object):

    def __init__(self, n_episodes, n_way, n_query, n_support, mode='train'):
        self.generator = FewshotBirdsDataGenerator(
            batch_size=n_query+n_support,
            episode_length=1,
            episode_width=None,
            image_dim=(299, 299, 3) # TODO: change this
        )
        self.n_episodes = n_episodes
        self.n_way = n_way
        self.n_query = n_query
        self.n_support = n_support
        print('num test classes:', len(self.generator.test_data.keys()))
        print('examples for each class:')
        for k in self.generator.test_data:
            print('k:', k, 'num examples:', len(self.generator.test_data[k].keys()))
        print('num examples:', sum([len(self.generator.test_data[k].keys()) for k in self.generator.test_data]))

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            data = None
            if self.mode == 'train':
                data = self.generator.train_data
            elif self.mode == 'val':
                data = self.generator.val_data
            elif self.mode == 'test':
                data = self.generator.test_data
            # TODO: allow some way to select the split
            yield self._sample_episode(data)

    def _sample_episode(self, dataset):
        # dataset is a dictionary of classes where each class
        # you want to return a dictionary which for each epoch maps
        # qs: query points, xs: support points N_cxSxCxHxW
        H, W, C = self.generator.image_dim
        # we flip the C indices at the end
        xq = np.zeros(shape=(self.n_way, self.n_query, C, H, W))
        xs = np.zeros(shape=(self.n_way, self.n_support, C, H, W))
        episode = {'xq' : None, 'xs' : None}
        # select the classes
        ys = np.array(dataset.keys())
        y_idxs = np.random.choice(len(ys), self.n_way)
        y_sample = ys[y_idxs]
        for i, c in enumerate(y_sample):
            # extract example configs
            total = self.n_query + self.n_support
            x_dict = dataset[c]
            x_keys = np.array(x_dict.keys())
            x_idxs = np.random.choice(len(x_keys), total)
            x_sample = [[x_dict[k] for k in x_keys[x_idxs]]]
            # get sample, note this is going to fail because of the dimensions
            # maybe if you set batch_size to n_query + n_support and you set episode_length to 1
            img_x, img_p1, img_p2 = [np.squeeze(img, 0) for img in \
                                     self.generator._get_examples_for_image_configs(x_sample)]
            # flip channels axis
            img_x = np.swapaxes(img_x, 1, 3)
            q, s  = img_x[:self.n_query, :, :, :], img_x[self.n_query:, :, :, :]
            xq[i, :, :, :, :], xs[i, :, :, :, :] = q, s
        # TODO: make these torch tensors on the right device
        episode['xq'] = torch.Tensor(xq)
        episode['xs'] = torch.Tensor(xs)
        return episode

if __name__ == '__main__':
    loader = PytorchBirdsDataLoader(10, 2, 1, 1)
    #loader._sample_episode(loader.generator.train_data)
    #for x in loader:
    #    print(x)
