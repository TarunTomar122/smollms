from smollms.train import _build_model, build_parser


def test_kimi2_defaults_to_hybrid_but_honors_all_full_ablation():
    parser = build_parser()

    default_args = parser.parse_args(["--arch", "kimi2"])
    _, default_config, default_model = _build_model(default_args, vocab_size=20)
    assert default_config.hybrid_pattern == "3L1F"
    assert default_model.attn_types == ["kda", "kda", "kda", "mla"]

    full_args = parser.parse_args(["--arch", "kimi2", "--hybrid-pattern", "all_full"])
    _, full_config, full_model = _build_model(full_args, vocab_size=20)
    assert full_config.hybrid_pattern == "all_full"
    assert full_model.attn_types == ["full"] * 4


def test_train_parser_records_a_seed():
    assert build_parser().parse_args([]).seed == 1337
