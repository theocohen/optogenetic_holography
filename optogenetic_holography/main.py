import os
import logging

from torch.utils.tensorboard import SummaryWriter

from optogenetic_holography import utils

output_path = './output/'

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

# Tensorboard writer
summaries_dir = os.path.join(output_path, 'summaries')
utils.cond_mkdir(summaries_dir)
writer = SummaryWriter(summaries_dir)

writer.close()

if __name__ == '__main__':
    print("hi")