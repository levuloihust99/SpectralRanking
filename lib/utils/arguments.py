import argparse


def create_parser():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    # tokenizer
    parser.add_argument("--tokenizer_path",
                        help="Path to the tokenizer to be instantiated by AutoTokenizer")
    parser.add_argument("--sep_token",
                        help="Specify sep_token for the tokenizer")

    # model
    parser.add_argument("--model_type",
                        help="Type of the model to initialize")
    parser.add_argument("--model_path",
                        help="Path to the model to be instantiated by AutoModelForSeq2SeqLM")

    # dataset
    parser.add_argument("--train_data_path",
                        help="Path to the train bytedataset directory.")
    parser.add_argument("--eval_data_path",
                        help="Path to the validation bytedataset directory.")

    # training config
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--output_dir",
                        help="Path to the output directory, which stores checkpoints.")
    parser.add_argument("--do_train", type=eval,
                        help="Whether to perform training.")
    parser.add_argument("--do_eval", type=eval,
                        help="Whether to perform evaluation.")
    parser.add_argument("--score_scale", type=float)
    parser.add_argument("--loss_margin", type=float)
    parser.add_argument("--loss_type", choices=["margin", "logsigmoid"])
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--num_train_epochs", type=int)
    parser.add_argument("--warmup_ratio", type=float)
    parser.add_argument("--warmup_steps", type=int)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--per_device_train_batch_size", type=int)
    parser.add_argument("--per_device_eval_batch_size", type=int)
    parser.add_argument("--adam_eps", type=float)
    parser.add_argument("--save_steps", type=int)
    parser.add_argument("--eval_steps", type=int)
    parser.add_argument("--save_total_limit", type=int)
    parser.add_argument("--max_grad_norm", type=float)
    parser.add_argument("--metric_for_best_model")
    parser.add_argument("--greater_is_better", type=eval)
    parser.add_argument("--resume_from_checkpoint")
    parser.add_argument("--seed", type=int)

    # data params
    parser.add_argument("--max_input_len", type=int)

    # log params
    parser.add_argument("--wandb_api_key")
    parser.add_argument("--log_level")
    parser.add_argument("--report_to", type=eval)
    parser.add_argument("--logging_dir",
                        help="Path to the logging directory, may be tensorboard logging.")
    parser.add_argument("--logging_steps", type=int)

    parser.add_argument("--hparams", default="{}")

    return parser
