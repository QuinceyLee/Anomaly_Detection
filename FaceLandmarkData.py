import torch
import torch.utils.data as data
import pandas as pd


class FaceLandmarksDataset(data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file, iterator=True, index_col=0, header=0)

    def __len__(self):
        # print len(self.landmarks_frame)
        # return len(self.landmarks_frame)
        return 1800000

    def __getitem__(self, idx):
        landmarks = self.landmarks_frame.get_chunk(128).as_matrix().astype('float')
        # landmarks = self.landmarks_frame.ix[idx, 1:].as_matrix().astype('float')
        # 采用这个，不错。
        return landmarks
