"""Check imports for the implemented analytical pipeline."""
import sys
from importlib import import_module, metadata


def main():
    print(f'Python: {sys.version.split()[0]}')
    for name in ['polars', 'pandas', 'pyarrow', 'openpyxl']:
        import_module(name)
        print(f'{name}: {metadata.version(name)}')
    print('Required runtime imports passed.')


if __name__ == '__main__':
    main()
