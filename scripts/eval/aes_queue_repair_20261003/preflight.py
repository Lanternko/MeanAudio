"""Validate process-local dependencies before the unchanged source-study preflight."""
import json
import os
import runpy
import sys
from pathlib import Path


def main():
    c = json.loads(Path(os.environ['GPU_QUEUE_CONTRACT']).read_text())
    expected = c['runtime_environment']['PYTHONPATH']
    if os.environ.get('PYTHONPATH') != expected:
        raise ValueError('registered isolated dependency path missing')
    from demucs_infer.pretrained import get_model
    import torch
    model = get_model('htdemucs_6s', repo=Path(c['demucs_model_repo'])).eval()
    assert next(model.parameters()).device.type == 'cpu'
    assert set(model.sources) == {'drums', 'bass', 'other', 'vocals', 'guitar', 'piano'}
    del model
    assert not torch.cuda.is_initialized(), 'dependency preflight must not allocate GPU'
    script = Path(c['original_preflight'])
    sys.path.insert(0, str(script.parent))
    runpy.run_path(str(script), run_name='__main__')


if __name__ == '__main__':
    main()
