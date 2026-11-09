"""Build CPUAdam before and after the GIL patch with identical CPU flags."""

from pathlib import Path
import subprocess

BASE_REVISION = '1a956270f42fbe7ffa485fdff45499eb77318469'


def main():
    from deepspeed.ops.op_builder.cpu.cpu_adam import CPUAdamBuilder
    from torch.utils.cpp_extension import load

    root = Path('upstream/DeepSpeed').resolve()
    baseline = Path('.torch_extensions/cpu_adam_base_sources').resolve()
    baseline.mkdir(parents=True,exist_ok=True)
    builder = CPUAdamBuilder()
    base_sources = []
    for source in builder.sources():
        target = baseline/Path(source).name
        target.write_bytes(subprocess.check_output(
            ['git','show',f'{BASE_REVISION}:{source}'],cwd=root))
        base_sources.append(str(target))
    for variant,sources in [('base',base_sources),
                            ('gil',[str(root/path) for path in builder.sources()])]:
        load(name=f'cpu_adam_{variant}',sources=sources,
             extra_include_paths=[str(root/path) for path in builder.include_paths()],
             extra_cflags=builder.cxx_args(),
             extra_ldflags=builder.extra_ldflags()+['-fopenmp'])
        print(f'Built cpu_adam_{variant}',flush=True)


if __name__ == '__main__':
    main()
