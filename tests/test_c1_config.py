from precision_md.config import Gate1Config, load_config


def test_c1_config_is_independently_seeded_and_excludes_p1():
    config = load_config("configs/c1-dataset.yaml", Gate1Config)

    assert config.dataset_id == "c1-confirmatory"
    assert config.seed == 2026082601
    assert str(config.exclude_selection) == "data/frozen/p1/selection.json"
    assert str(config.output_dir) == "results/datasets/c1-confirmatory"
