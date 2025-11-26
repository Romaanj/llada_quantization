import sys


def ensure_flag(flag: str, value: str):
    if flag in sys.argv:
        return
    sys.argv.extend([flag, value])


def main():
    ensure_flag("--quant_method", "mxfp4")
    from DuQuant.main_duquant_gptq import main as base_main
    base_main()


if __name__ == "__main__":
    main()

