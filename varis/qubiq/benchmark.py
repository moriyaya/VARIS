"""Six-response generation latency using block medians."""
import argparse
import json
import random
import statistics
import time
from pathlib import Path
import torch
from .model import CompactUNet, QubiqDataset

ROOT = Path(__file__).resolve().parents[2]

@torch.no_grad()
def measure(model, images, device, blocks, warmup, repeats):
    model.eval()
    def run(image):
        return model(image).sigmoid() >= .5
    def synchronize():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    rows, block_medians = [], []
    for block in range(blocks):
        order = list(range(len(images)))
        random.Random(20261101 + block).shuffle(order)
        for _ in range(warmup):
            run(images[order[0]][1])
        synchronize()
        medians = []
        for index in order:
            name, image = images[index]
            times = []
            for _ in range(repeats):
                synchronize()
                start = time.perf_counter()
                run(image)
                synchronize()
                times.append(1000 * (time.perf_counter() - start))
            median = statistics.median(times)
            medians.append(median)
            rows.append(dict(block=block, case=name, median_ms=median))
        block_medians.append(statistics.median(medians))
    return dict(mean_block_median_ms=statistics.mean(block_medians),
                sd_block_median_ms=statistics.stdev(block_medians),
                block_medians_ms=block_medians, case_block_rows=rows)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--config', type=Path, default=ROOT/'configs/qubiq.json')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    cfg = json.loads(a.config.read_text())
    if not torch.cuda.is_available():
        p.error('Generation-latency measurement requires CUDA.')
    device = torch.device('cuda:0')
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    dataset = QubiqDataset(sorted(p for p in a.data.glob('case*') if p.is_dir()), cfg['task'], cfg['size'], False)
    if len(dataset) != 48:
        p.error('Expected 48 QUBIQ cases.')
    images = [(name, image[None].to(device)) for name, image, _, _ in dataset]
    model = CompactUNet(1, cfg['candidates']).to(device)
    model.load_state_dict(torch.load(a.checkpoint, map_location=device, weights_only=True)['model'])
    result = measure(model, images, device, **cfg['timing'])
    result.update(gpu=torch.cuda.get_device_name(device), cases=48, timing=cfg['timing'])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != 'case_block_rows'}))

if __name__ == '__main__':
    main()
