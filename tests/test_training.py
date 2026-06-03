from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from gemma_health.config import AppConfig, load_config
from gemma_health.evals.medmcqa_generation import (
    MedMcqaPrediction,
    _encode_prompts,
    _generation_token_kwargs,
    _left_pad_tokenizer,
    _summarize_segment,
    _text_tokenizer,
    extract_medmcqa_label,
    normalize_medmcqa_eval_row,
    score_medmcqa_completion,
    summarize_medmcqa_predictions,
)
from gemma_health.training.run import run_training
from gemma_health.training import unsloth_sft
from gemma_health.training.hf_policy_merge import (
    _find_target_weight_tensor,
    _module_name_candidates,
    _normalized_module_name,
)
from gemma_health.training.unsloth_sft import adapter_training_config, adapter_dataset_sources
from gemma_health.training.unsloth_grpo import (
    correct_label_from_row,
    correct_option_reward,
    grpo_policy_config,
    grpo_row_from_medmcqa,
    is_usable_grpo_source_row,
    medical_correctness_reward,
    prepare_grpo_peft_model,
    reference_overlap_reward,
)
from scripts.modal_train_adapter import ModalTrainingArgs, build_training_command
from scripts.modal_grpo import (
    ModalGrpoMedMcqaEvalArgs,
    ModalGrpoTrainingArgs,
    ModalHfPolicyMergeArgs,
    build_grpo_medmcqa_eval_command,
    build_grpo_training_command,
    build_hf_policy_merge_command,
)


def _use_temp_field_dialogues(config: AppConfig, tmp_path: Path, *, examples: int) -> None:
    data_path = tmp_path / "field_dialogues.jsonl"
    rows = [
        '{"prompt":"రోగికి జ్వరం ఉంది. ఏమి చేయాలి?","response":"ఉష్ణోగ్రత కొలిచి, నీరు తాగాలి."}',
        '{"prompt":"తలనొప్పికి సాధారణ జాగ్రత్త?","response":"విశ్రాంతి తీసుకుని, తీవ్రమైతే వైద్యుడిని సంప్రదించాలి."}',
        '{"prompt":"దగ్గు ఎక్కువైతే?","response":"మాస్క్ వాడి, లక్షణాలు కొనసాగితే పరీక్ష చేయించాలి."}',
        '{"prompt":"మందు మోతాదు మిస్సయితే?","response":"తదుపరి మోతాదును రెట్టింపు చేయకూడదు."}',
        '{"prompt":"చర్మంపై దద్దుర్లు వస్తే?","response":"కారణం తెలియకపోతే వైద్య సలహా తీసుకోవాలి."}',
    ]
    data_path.write_text("\n".join(rows[:examples]) + "\n", encoding="utf-8")
    raw_config = config.raw
    for dataset in raw_config["datasets"]:
        dataset["enabled"] = dataset["name"] == "field_dialogues"
        dataset["weight"] = 1.0 if dataset["enabled"] else 0.0
        if dataset["name"] == "field_dialogues":
            dataset["path"] = str(data_path)
            dataset["max_examples"] = examples
    raw_config["training"]["adapters"]["telugu"]["dataset_names"] = ["field_dialogues"]


def test_adapter_training_config_loads_telugu_profile() -> None:
    config = load_config(Path("config.yaml"))

    adapter = adapter_training_config(config, "telugu")

    assert adapter.name == "telugu"
    assert adapter.output_dir == Path("artifacts/adapters/telugu")
    assert "telugu_alpaca" in adapter.dataset_names


def test_medical_adapter_uses_ready_medmcqa_source() -> None:
    config = load_config(Path("config.yaml"))
    adapter = adapter_training_config(config, "medical")

    sources = adapter_dataset_sources(config, adapter)

    assert [source.name for source in sources] == ["medmcqa"]


def test_run_training_dry_run_prepares_sft_jsonl(tmp_path: Path) -> None:
    config = load_config(Path("config.yaml"))
    _use_temp_field_dialogues(config, tmp_path, examples=3)
    training = config.raw["training"]
    training["adapters"]["telugu"]["sft_jsonl"] = str(tmp_path / "telugu.jsonl")
    training["adapters"]["telugu"]["test_jsonl"] = str(tmp_path / "telugu_test.jsonl")
    training["adapters"]["telugu"]["max_examples"] = 3

    result = run_training(config, adapter_name="telugu", execute=False)

    assert result.executed is False
    assert result.dataset_path == tmp_path / "telugu.jsonl"
    assert result.dataset_path.exists()
    assert result.test_dataset_path.exists()
    assert len(result.dataset_path.read_text(encoding="utf-8").splitlines()) == 3
    assert len(result.test_dataset_path.read_text(encoding="utf-8").splitlines()) == 0


def test_run_training_allocates_adapter_cap_by_dataset_weight(tmp_path: Path) -> None:
    config = load_config(Path("config.yaml"))
    _use_temp_field_dialogues(config, tmp_path, examples=5)
    training = config.raw["training"]
    training["adapters"]["telugu"]["sft_jsonl"] = str(tmp_path / "telugu.jsonl")
    training["adapters"]["telugu"]["test_jsonl"] = str(tmp_path / "telugu_test.jsonl")
    training["adapters"]["telugu"]["max_examples"] = 10

    result = run_training(config, adapter_name="telugu", execute=False)

    rows = result.dataset_path.read_text(encoding="utf-8").splitlines()
    test_rows = result.test_dataset_path.read_text(encoding="utf-8").splitlines()
    assert len(rows) + len(test_rows) <= 10
    assert len(rows) + len(test_rows) >= 5


def test_execute_with_hub_dataset_skips_local_preparation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_config(Path("config.yaml"))
    training = config.raw["training"]
    training["adapters"]["telugu"]["hub_dataset_id"] = "user/gemma-health-telugu-sft"
    training["adapters"]["telugu"]["sft_jsonl"] = str(tmp_path / "should_not_exist.jsonl")
    called = []

    def fake_run_training(*args: object) -> None:
        called.append(args)

    monkeypatch.setattr(unsloth_sft, "_run_unsloth_training", fake_run_training)

    result = run_training(config, adapter_name="telugu", execute=True)

    assert result.executed is True
    assert called
    assert not result.dataset_path.exists()


def test_modal_multi_gpu_command_uses_torchrun_script_entrypoint() -> None:
    command = build_training_command(
        ModalTrainingArgs(
            adapter="telugu",
            hub_dataset_id="user/dataset",
            gpu="H100:8",
            timeout_hours=6,
            max_retries=3,
            max_steps=5500,
            max_examples=None,
            batch_size=4,
            gradient_accumulation_steps=1,
            skip_eval=True,
            resume_from_checkpoint=False,
            output_root="/checkpoints/adapters/h1",
            smoke_samples=0,
            hub_model_id="user/model",
            push_to_hub=True,
            detach=True,
            hf_secret="hf-secret",
            wandb_secret="wandb-secret",
        )
    )

    assert command[:9] == [
        "torchrun",
        "--standalone",
        "--nnodes",
        "1",
        "--nproc_per_node",
        "8",
        "--tee",
        "3",
        "scripts/train.py",
    ]
    assert "python" not in command[:10]
    assert "--batch-size" in command
    assert "--gradient-accumulation-steps" in command
    assert "--skip-eval" in command
    assert "--no-resume-from-checkpoint" in command


def test_grpo_row_from_medmcqa_builds_prompt_and_answer() -> None:
    row = {
        "question": "Which vitamin is only from animal source?",
        "opa": "Vitamin C",
        "opb": "Vitamin B7",
        "opc": "Vitamin B12",
        "opd": "Vitamin D",
        "cop": 2,
        "exp": "B12 comes from animal foods.",
        "telugu": "సరైన సమాధానం C.",
    }

    grpo_row = grpo_row_from_medmcqa(row)

    assert correct_label_from_row(row) == "C"
    assert grpo_row["answer"] == "C"
    assert grpo_row["correct_text"] == "Vitamin B12"
    assert grpo_row["prompt"][1]["role"] == "user"
    assert "తెలుగులో సహజంగా సమాధానం ఇవ్వండి" in grpo_row["prompt"][1]["content"]


def test_grpo_source_row_rejects_needs_review() -> None:
    row = {
        "question": "Q",
        "opa": "A",
        "opb": "B",
        "opc": "C",
        "opd": "D",
        "cop": 0,
        "synthetic_telugu": "NEEDS_REVIEW",
    }

    assert is_usable_grpo_source_row(row) is False


def test_grpo_correct_option_reward_scores_label_match() -> None:
    rewards = correct_option_reward(
        completions=["సమాధానం: C\nచిన్న వివరణ.", "సమాధానం: A", "వివరణ మాత్రమే"],
        answer=["C", "D", "B"],
    )

    assert rewards == [2.5, -1.0, -0.75]


def test_grpo_correctness_reward_accepts_correct_option_text() -> None:
    rewards = medical_correctness_reward(
        completions=["విటమిన్ B12 జంతు ఆహారాలలో లభిస్తుంది."],
        answer=["C"],
        correct_text=["Vitamin B12"],
    )

    assert rewards == [1.75]


def test_grpo_reference_overlap_rewards_medical_terms() -> None:
    rewards = reference_overlap_reward(
        completions=["విటమిన్ B12 కోబాలమిన్ జంతు ఆహారం నుండి వస్తుంది."],
        reference_telugu=["విటమిన్ B12 కోబాలమిన్ మాంసం మరియు పాల ఉత్పత్తులలో ఉంటుంది."],
        reference_explanation=["Vitamin B12 comes from animal foods."],
    )

    assert rewards[0] > 0


def test_grpo_policy_config_uses_selected_merged_base() -> None:
    config = load_config(Path("config.yaml"))

    policy = grpo_policy_config(
        config,
        base_model_id=None,
        dataset_path=None,
        hub_dataset_id=None,
        split=None,
        output_dir=None,
        hub_model_id=None,
        max_examples=None,
        batch_size=None,
        gradient_accumulation_steps=None,
        learning_rate=None,
        max_steps=None,
        num_generations=None,
        generation_batch_size=None,
        max_prompt_length=None,
        max_completion_length=None,
        use_vllm=None,
        vllm_gpu_memory_utilization=None,
        push_to_hub=None,
    )

    assert policy.base_model_id == "RohithMidigudla/gemma-health-telugu-medical-merged-h1-30-h2-70"
    assert policy.dataset_path == Path("data/staged/synthetic_telugu/medmcqa/train.parquet")
    assert policy.hub_model_id == "RohithMidigudla/gemma-health-telugu-medical-grpo-policy-v1"
    assert policy.batch_size == 16
    assert policy.gradient_accumulation_steps == 1
    assert policy.num_generations == 8
    assert policy.generation_batch_size == 64
    assert policy.max_completion_length == 128
    assert policy.use_vllm is False


def test_grpo_policy_config_cli_hub_dataset_overrides_config_path() -> None:
    config = load_config(Path("config.yaml"))

    policy = grpo_policy_config(
        config,
        base_model_id=None,
        dataset_path=None,
        hub_dataset_id="user/synthetic-grpo",
        split=None,
        output_dir=None,
        hub_model_id=None,
        max_examples=None,
        batch_size=None,
        gradient_accumulation_steps=None,
        learning_rate=None,
        max_steps=None,
        num_generations=None,
        generation_batch_size=None,
        max_prompt_length=None,
        max_completion_length=None,
        use_vllm=None,
        vllm_gpu_memory_utilization=None,
        push_to_hub=None,
    )

    assert policy.dataset_path is None
    assert policy.hub_dataset_id == "user/synthetic-grpo"


def test_prepare_grpo_peft_model_reuses_loaded_adapter() -> None:
    class FakeFastLanguageModel:
        trained = False
        added = False

        @classmethod
        def for_training(cls, model: object) -> None:
            cls.trained = True

        @classmethod
        def get_peft_model(cls, *args: object, **kwargs: object) -> object:
            cls.added = True
            raise AssertionError("must not add another LoRA adapter")

    class FakePeftModel:
        peft_config = {"default": object()}

    model = FakePeftModel()

    prepared = prepare_grpo_peft_model(
        fast_language_model=FakeFastLanguageModel,
        model=model,
        lora={},
        seed=7,
    )

    assert prepared is model
    assert FakeFastLanguageModel.trained is True
    assert FakeFastLanguageModel.added is False


def test_prepare_grpo_peft_model_adds_adapter_for_plain_model() -> None:
    class FakeFastLanguageModel:
        received_kwargs = {}

        @classmethod
        def get_peft_model(cls, model: object, **kwargs: object) -> object:
            cls.received_kwargs = kwargs
            return "peft-model"

    prepared = prepare_grpo_peft_model(
        fast_language_model=FakeFastLanguageModel,
        model=object(),
        lora={"r": 32, "alpha": 64},
        seed=7,
    )

    assert prepared == "peft-model"
    assert FakeFastLanguageModel.received_kwargs["r"] == 32
    assert FakeFastLanguageModel.received_kwargs["lora_alpha"] == 64


def test_modal_grpo_training_command_targets_policy_script() -> None:
    command = build_grpo_training_command(
        ModalGrpoTrainingArgs(
            base_model_id="user/base",
            dataset_path="/checkpoints/datasets/train.parquet",
            hub_dataset_id=None,
            split="train",
            output_dir="/checkpoints/adapters/grpo",
            hub_model_id="user/grpo",
            gpu="H100",
            timeout_hours=8,
            max_retries=1,
            max_examples=128,
            batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=5e-6,
            max_steps=50,
            num_generations=8,
            generation_batch_size=64,
            max_prompt_length=1024,
            max_completion_length=512,
            use_vllm=False,
            vllm_gpu_memory_utilization=0.55,
            push_to_hub=True,
            detach=True,
            hf_secret="hf-secret",
            wandb_secret="wandb-secret",
        )
    )

    assert command[:2] == ["python", "scripts/train_grpo.py"]
    assert "--base-model-id" in command
    assert "user/base" in command
    assert "--dataset-path" in command
    assert "/checkpoints/datasets/train.parquet" in command
    assert "--num-generations" in command
    assert "8" in command
    assert "--generation-batch-size" in command
    assert "64" in command
    assert "--no-use-vllm" in command
    assert "--push-to-hub" in command


def test_modal_hf_policy_merge_command_targets_hf_policy_merge_script() -> None:
    command = build_hf_policy_merge_command(
        ModalHfPolicyMergeArgs(
            base_model_id="google/gemma-4-E4B-it",
            adapter_model_id="user/grpo-adapter",
            output_dir="/checkpoints/merged_models/v3-hf",
            hub_model_id="user/full-v3-hf",
            dtype="bfloat16",
            max_shard_size="5GB",
            private=False,
            overwrite=True,
            safe_merge=True,
            missing_target_policy="warn",
            gpu="H100",
            timeout_hours=4,
            max_retries=0,
            detach=True,
            hf_secret="hf-secret",
        )
    )

    assert command[:2] == ["python", "scripts/hf_merge_policy_adapter.py"]
    assert "--base-model-id" in command
    assert "google/gemma-4-E4B-it" in command
    assert "--adapter-model-id" in command
    assert "user/grpo-adapter" in command
    assert "--hub-model-id" in command
    assert "user/full-v3-hf" in command
    assert "--safe-merge" in command
    assert "--missing-target-policy" in command
    assert "warn" in command
    assert "--execute" in command


def test_hf_policy_merge_maps_unsloth_gemma4_language_model_prefix() -> None:
    candidates = _module_name_candidates("model.language_model.layers.24.self_attn.k_proj")

    assert "language_model.layers.24.self_attn.k_proj" in candidates
    assert "layers.24.self_attn.k_proj" in candidates
    assert _normalized_module_name("model.language_model.layers.24.self_attn.k_proj") == "layers.24.self_attn.k_proj"


def test_hf_policy_merge_finds_wrapped_weight_by_normalized_name() -> None:
    target = object()
    model_state = {"language_model.model.layers.24.self_attn.k_proj.linear.weight": target}

    assert _find_target_weight_tensor(model_state, "model.language_model.layers.24.self_attn.k_proj") is target


def test_hf_policy_merge_finds_weight_by_normalized_suffix() -> None:
    target = object()
    model_state = {"language_model.model.decoder.layers.24.self_attn.k_proj.linear.weight": target}

    assert _find_target_weight_tensor(model_state, "model.language_model.layers.24.self_attn.k_proj") is target


def test_medmcqa_generation_eval_extracts_telugu_answer_label() -> None:
    assert extract_medmcqa_label("సమాధానం: C\nవిటమిన్ B12 సరైనది.") == "C"
    assert extract_medmcqa_label("Option B is correct.") == "B"
    assert extract_medmcqa_label("వివరణ మాత్రమే ఉంది") is None


def test_medmcqa_generation_eval_summarizes_parts() -> None:
    row = {
        "question": "Which vitamin is only from animal source?",
        "opa": "Vitamin C",
        "opb": "Vitamin B7",
        "opc": "Vitamin B12",
        "opd": "Vitamin D",
        "cop": 2,
        "subject_name": "Biochemistry",
        "topic_name": "Vitamins",
    }

    prediction = score_medmcqa_completion(
        0,
        row,
        "సమాధానం: C\nవిటమిన్ B12 జంతు ఆహారాలలో లభిస్తుంది.",
        max_new_tokens=128,
    )
    summaries = summarize_medmcqa_predictions([prediction])

    assert prediction.exact_label_match is True
    assert prediction.correct_option_text_hit is True
    assert summaries[0].segment == "overall"
    assert summaries[0].exact_label_accuracy == 1.0
    assert summaries[0].empty_rate == 0.0
    assert any(summary.segment == "subject" and summary.value == "Biochemistry" for summary in summaries)


def test_medmcqa_generation_eval_accepts_grpo_shaped_rows() -> None:
    row = {
        "prompt": [
            {"role": "system", "content": "Tutor"},
            {"role": "user", "content": "Q\nA. one\nB. two\nC. Vitamin B12\nD. four"},
        ],
        "answer": "C",
        "correct_text": "Vitamin B12",
        "subject_name": "Biochemistry",
        "topic_name": "Vitamins",
    }

    eval_row = normalize_medmcqa_eval_row(row)

    assert eval_row is not None
    assert eval_row.correct_label == "C"
    assert eval_row.correct_text == "Vitamin B12"


def test_medmcqa_generation_eval_derives_label_when_cop_is_missing() -> None:
    row = {
        "question": "Marker of endoplasmic reticulum?",
        "opa": "Acid phosphatase",
        "opb": "Glucose-6-phosphatase",
        "opc": "Catalase",
        "opd": "LDH",
        "cop": -1,
        "telugu": "సరైన సమాధానం: B. గ్లూకోజ్-6-ఫాస్ఫేటేజ్",
        "subject_name": "Physiology",
    }

    eval_row = normalize_medmcqa_eval_row(row)

    assert eval_row is not None
    assert eval_row.correct_label == "B"
    assert eval_row.correct_text == "Glucose-6-phosphatase"


def test_medmcqa_generation_eval_passes_text_to_gemma4_processor() -> None:
    calls = []

    class FakeProcessor:
        def __call__(self, *args: object, **kwargs: object) -> object:
            calls.append((args, kwargs))
            return {"input_ids": [1, 2, 3]}

    encoded = _encode_prompts(FakeProcessor(), ["hello"])

    assert encoded == {"input_ids": [1, 2, 3]}
    assert calls[0][0] == ()
    assert calls[0][1]["text"] == ["hello"]


def test_medmcqa_generation_eval_uses_inner_tokenizer_for_gemma4_processor() -> None:
    tokenizer = object()

    class FakeProcessor:
        pass

    processor = FakeProcessor()
    processor.tokenizer = tokenizer

    assert _text_tokenizer(processor) is tokenizer


def test_medmcqa_generation_eval_forces_left_padding() -> None:
    class FakeTokenizer:
        padding_side = "right"

    tokenizer = FakeTokenizer()
    _left_pad_tokenizer(tokenizer)

    assert tokenizer.padding_side == "left"


def test_medmcqa_generation_eval_bans_pad_generation() -> None:
    class FakeTokenizer:
        eos_token_id = 2
        pad_token_id = 0

    kwargs = _generation_token_kwargs(FakeTokenizer())

    assert kwargs["eos_token_id"] == 2
    assert kwargs["pad_token_id"] == 2
    assert kwargs["bad_words_ids"] == [[0]]


def test_medmcqa_generation_eval_tracks_empty_outputs() -> None:
    summary = _summarize_segment(
        "overall",
        "all",
        [
            MedMcqaPrediction(
                index=0,
                subject_name="x",
                topic_name="y",
                correct_label="A",
                predicted_label=None,
                exact_label_match=False,
                correct_option_text_hit=False,
                telugu_density=0.0,
                token_length=0,
                empty=True,
                clipped=False,
                unsafe=False,
                completion="",
            )
        ],
    )

    assert summary.empty_rate == 1.0


def test_modal_grpo_medmcqa_eval_command_targets_generation_script() -> None:
    command = build_grpo_medmcqa_eval_command(
        ModalGrpoMedMcqaEvalArgs(
            model_id="user/grpo-policy-v2",
            hub_dataset_id="user/synthetic-medmcqa",
            split="test",
            gpu="H100",
            timeout_hours=6,
            max_retries=1,
            start_index=1792,
            max_samples=None,
            batch_size=8,
            max_new_tokens=128,
            min_new_tokens=1,
            temperature=0.0,
            sample_output_limit=5,
            output_root="/checkpoints/evals/grpo_medmcqa",
            detach=True,
            hf_secret="hf-secret",
        )
    )

    assert command[:2] == ["python", "scripts/eval_grpo_medmcqa.py"]
    assert "--model-id" in command
    assert "user/grpo-policy-v2" in command
    assert "--hub-dataset-id" in command
    assert "user/synthetic-medmcqa" in command
    assert "--output-jsonl" in command
    assert "--start-index" in command
    assert "1792" in command
    assert "--min-new-tokens" in command
    assert "--sample-output-limit" in command
    assert "--max-samples" not in command


def test_local_rank_parser(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    assert unsloth_sft._local_rank() is None

    monkeypatch.setenv("LOCAL_RANK", "3")
    assert unsloth_sft._local_rank() == 3

    monkeypatch.setenv("LOCAL_RANK", "bad")
    with pytest.raises(ValueError, match="LOCAL_RANK"):
        unsloth_sft._local_rank()
