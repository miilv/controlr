"""One bounded SAM process, using the existing shared text wire protocol."""
import sys
import torch
from dojo.perception.sam_masks import serve

if __name__ == '__main__':
    torch.set_num_threads(6)
    torch.cuda.set_per_process_memory_fraction(.25)
    serve(sys.argv[1], ['sam3'], max_hours=2)
