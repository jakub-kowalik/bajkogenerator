import torchdata.dataloader2 as dl2
import torchdata.datapipes as dp


class StoryDataset:
    def __init__(
        self,
        root,
        batch_size=1,
        num_workers=1,
        shuffle=True,
        drop_last=False,
        sequence_size=32,
        pad_idx=2,
    ):
        self.sequence_size = sequence_size
        self.pad_idx = pad_idx

        datapipe = dp.iter.FileLister(root, recursive=True).filter(
            filter_fn=self.filter_fn
        )
        datapipe = dp.iter.FileOpener(datapipe, mode="rt")
        datapipe = dp.iter.StreamReader(datapipe)
        datapipe = dp.iter.Mapper(datapipe, fn=self.map_fn)
        datapipe = (
            dp.iter.FlatMapper(datapipe, fn=self.batch_fn).shuffle().sharding_filter()
        )
        datapipe = dp.iter.Batcher(datapipe, batch_size=batch_size, drop_last=drop_last)

        self.dloader2 = dl2.DataLoader2(
            datapipe,
            reading_service=dl2.MultiProcessingReadingService(num_workers=num_workers),
            datapipe_adapter_fn=dl2.adapter.Shuffle(shuffle),
        )

    def __iter__(self):
        return self.dloader2.__iter__()

    def map_fn(self, x):
        return (self.sequence_size - 1) * [self.pad_idx] + [
            int(y) for y in x[1].split(",")
        ]

    def batch_fn(self, x):
        return [
            x[i : i + self.sequence_size + 1]
            for i in range(0, len(x) - self.sequence_size)
        ]

    @staticmethod
    def filter_fn(name):
        return name.endswith(".txt")
