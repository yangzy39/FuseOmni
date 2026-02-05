# from modelscope.msdatasets import MsDataset

# # 这会下载并解压数据
# dataset = MsDataset.load(
#     'speech_asr/speech_asr_aishell1_trainsets',
#     split='train',  # 指定下载训练集
#     namespace='speech_asr',
#     cache_dir='/mnt/afs/00036/yzy/FuseOmni/datasets/data/'
# )

from modelscope.hub.utils.utils import get_cache_dir
import os


print(os.path.join(get_cache_dir(), 'datasets'))