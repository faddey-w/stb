import argparse
import os
from strateobots.ai.lib import model_saving


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("--version", "-v", choices=["1", "2", "3"])
    opts = parser.parse_args()
    model_dir = opts.model_dir
    target_controls = ["move", "orientation", "gun_orientation", "action"]
    model_constructor = f"strateobots.ai.nets.dnn:make_v{opts.version}"
    encoder_name = "1vs1_fully_visible"

    os.makedirs(model_dir)
    model_saving.save_model_config(
        model_dir,
        controls=target_controls,
        model_constructor=model_constructor,
        encoder_name=encoder_name,
    )


if __name__ == "__main__":
    main()
