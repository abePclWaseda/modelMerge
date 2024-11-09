from ..espnet.espnet2.bin.asr_inference import Speech2Text

# (Pdb) p(speech2text_kwargs)
# {'asr_train_config': 'exp/asr_train_asr_conformer_lr2e-3_warmup15k_amp_nondeterministic_raw_en_hugging_face_openai-community-gpt2_sp/config.yaml', 'asr_model_file': 'exp/asr_train_asr_conformer_lr2e-3_warmup15k_amp_nondeterministic_raw_en_hugging_face_openai-community-gpt2_sp/valid.acc.ave.pth', 'transducer_conf': None, 'lm_train_config': None, 'lm_file': None, 'ngram_file': None, 'token_type': None, 'bpemodel': None, 'device': 'cuda', 'maxlenratio': 0.0, 'minlenratio': 0.0, 'dtype': 'float32', 'beam_size': 20, 'ctc_weight': 0.3, 'lm_weight': 0.0, 'ngram_weight': 0.9, 'penalty': 0.0, 'nbest': 1, 'normalize_length': False, 'streaming': False, 'enh_s2t_task': False, 'multi_asr': False, 'quantize_asr_model': False, 'quantize_lm': False, 'quantize_modules': ['Linear'], 'quantize_dtype': 'qint8', 'hugging_face_decoder': False, 'hugging_face_decoder_conf': {}, 'time_sync': False, 'prompt_token_file': None, 'lang_prompt_token': None, 'nlp_prompt_token': None}
speech2text = Speech2Text(
    asr_train_config="/path/to/your/config.yaml",
    asr_model_file="/path/to/your/inference_asr_model",
    beam_size=10,
    ctc_weight=0.3,
    lm_weight=0.5
)

# (Pdb) p(speech2text_kwargs)
# {'asr_train_config': 'exp/asr_train_asr_conformer_lr2e-3_warmup15k_amp_nondeterministic_raw_en_hugging_face_openai-community-gpt2_sp/config.yaml', 'asr_model_file': 'exp/asr_train_asr_conformer_lr2e-3_warmup15k_amp_nondeterministic_raw_en_hugging_face_openai-community-gpt2_sp/valid.acc.ave.pth', 'transducer_conf': None, 'lm_train_config': 'exp/lm_train_transformer_gpt2_en_hugging_face/config.yaml', 'lm_file': 'exp/lm_train_transformer_gpt2_en_hugging_face/valid.loss.ave.pth', 'ngram_file': None, 'token_type': None, 'bpemodel': None, 'device': 'cuda', 'maxlenratio': 0.0, 'minlenratio': 0.0, 'dtype': 'float32', 'beam_size': 20, 'ctc_weight': 0.3, 'lm_weight': 0.1, 'ngram_weight': 0.9, 'penalty': 0.0, 'nbest': 1, 'normalize_length': False, 'streaming': False, 'enh_s2t_task': False, 'multi_asr': False, 'quantize_asr_model': False, 'quantize_lm': False, 'quantize_modules': ['Linear'], 'quantize_dtype': 'qint8', 'hugging_face_decoder': False, 'hugging_face_decoder_conf': {}, 'time_sync': False, 'prompt_token_file': None, 'lang_prompt_token': None, 'nlp_prompt_token': None, 'partial_ar': False, 'threshold_probability': 0.99, 'max_seq_len': 5, 'max_mask_parallel': -1}
speech2text_for_lm = Speech2Text(
    asr_train_config="/path/to/asr_config.yaml",  # 音声認識モデルの設定ファイル
    asr_model_file="/path/to/asr_model.pth",  # 音声認識モデルの重み
    lm_train_config="/path/to/lm_config.yaml",  # 言語モデルの設定ファイル
    lm_file="/path/to/lm.pth",  # 学習済み言語モデルの重み
    beam_size=10,
    ctc_weight=0.3,
    lm_weight=0.5
)