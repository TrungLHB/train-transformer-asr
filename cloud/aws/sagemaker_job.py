#!/usr/bin/env python3
"""Submit an ASR training job to Amazon SageMaker.

Usage::

    python cloud/aws/sagemaker_job.py \\
        --decoder ctc \\
        --language cy \\
        --role arn:aws:iam::123456789012:role/SageMakerRole \\
        --bucket my-s3-bucket \\
        [--instance ml.p3.2xlarge]
"""

import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Submit ASR training to SageMaker")
    parser.add_argument(
        "--decoder",
        choices=["ctc", "transformer"],
        default="ctc",
        help="Decoder type to train",
    )
    parser.add_argument("--language", default="cy", help="Language code (CommonVoice)")
    parser.add_argument(
        "--role",
        required=True,
        help="ARN of the IAM role for SageMaker execution",
    )
    parser.add_argument("--bucket", required=True, help="S3 bucket name for output artifacts")
    parser.add_argument(
        "--instance",
        default="ml.p3.2xlarge",
        help="SageMaker training instance type",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Max training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per GPU")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        import sagemaker
        from sagemaker.pytorch import PyTorch
    except ImportError:
        raise ImportError(
            "The 'sagemaker' package is required.  Install with: pip install sagemaker"
        )

    session = sagemaker.Session()

    config_file = (
        "configs/ctc_config.yaml"
        if args.decoder == "ctc"
        else "configs/transformer_config.yaml"
    )
    train_script = (
        "scripts/train_ctc.py" if args.decoder == "ctc" else "scripts/train_transformer.py"
    )

    hyperparameters = {
        "config": config_file,
        "language": args.language,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "checkpoint_dir": "/opt/ml/model/checkpoints",
        "log_dir": "/opt/ml/output/tensorboard",
    }

    estimator = PyTorch(
        entry_point=train_script,
        source_dir=".",
        role=args.role,
        framework_version="2.0.1",
        py_version="py310",
        instance_count=1,
        instance_type=args.instance,
        hyperparameters=hyperparameters,
        output_path=f"s3://{args.bucket}/asr-training/output",
        checkpoint_s3_uri=f"s3://{args.bucket}/asr-training/checkpoints",
        use_spot_instances=True,
        max_wait=86400,      # 24 h max wait for spot
        max_run=72000,       # 20 h max run time
        metric_definitions=[
            {"Name": "train:loss", "Regex": r"train_loss=(\S+)"},
            {"Name": "valid:wer",  "Regex": r"WER=(\S+)"},
            {"Name": "valid:cer",  "Regex": r"CER=(\S+)"},
        ],
    )

    job_name = f"asr-{args.decoder}-{args.language}"
    print(f"Submitting SageMaker training job: {job_name}")
    estimator.fit(job_name=job_name)
    print("Job submitted.  Monitor at https://console.aws.amazon.com/sagemaker")


if __name__ == "__main__":
    main()
