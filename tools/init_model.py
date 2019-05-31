import os
from strateobots.ai.lib import model_saving


def main():
    # model_dir = ".data/supervised/models/anglenav2"
    model_dir = ".data/a3c/models/direct"
    target_controls = ["move", "rotate", "tower_rotate", "action"]
    # target_controls = ["move", "orientation", "gun_orientation", "action"]
    model_constructor = "strateobots.ai.nets.dnn:make_v3"
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
