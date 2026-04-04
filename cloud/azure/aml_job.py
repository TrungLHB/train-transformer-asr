#!/usr/bin/env python3
"""Submit an ASR training job to Azure Machine Learning.

Usage::

    python cloud/azure/aml_job.py \\
        --decoder ctc \\
        --language cy \\
        --subscription_id <sub_id> \\
        --resource_group <rg> \\
        --workspace_name <ws> \\
        --compute_name <gpu_cluster> \\
        [--instance Standard_NC6s_v3]
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Submit ASR training to Azure ML")
    parser.add_argument(
        "--decoder",
        choices=["ctc", "transformer"],
        default="ctc",
        help="Decoder type",
    )
    parser.add_argument("--language", default="cy", help="Language code (CommonVoice)")
    parser.add_argument("--subscription_id", required=True, help="Azure subscription ID")
    parser.add_argument("--resource_group", required=True, help="Azure resource group")
    parser.add_argument("--workspace_name", required=True, help="Azure ML workspace name")
    parser.add_argument("--compute_name", required=True, help="AML compute cluster name")
    parser.add_argument(
        "--instance",
        default="Standard_NC6s_v3",
        help="VM instance size (used when creating a new compute cluster)",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Max training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        from azure.ai.ml import MLClient, command
        from azure.ai.ml.entities import Environment, AmlCompute
        from azure.identity import DefaultAzureCredential
    except ImportError:
        raise ImportError(
            "azure-ai-ml and azure-identity are required.  "
            "Install with: pip install azure-ai-ml azure-identity"
        )

    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=args.subscription_id,
        resource_group_name=args.resource_group,
        workspace_name=args.workspace_name,
    )

    # Ensure compute cluster exists (create if missing)
    try:
        ml_client.compute.get(args.compute_name)
        print(f"Using existing compute cluster: {args.compute_name}")
    except Exception:
        print(f"Creating compute cluster: {args.compute_name} ({args.instance}) …")
        gpu_cluster = AmlCompute(
            name=args.compute_name,
            size=args.instance,
            min_instances=0,
            max_instances=4,
            idle_time_before_scale_down=120,
        )
        ml_client.compute.begin_create_or_update(gpu_cluster).result()

    train_script = (
        "scripts/train_ctc.py" if args.decoder == "ctc" else "scripts/train_transformer.py"
    )
    config_file = (
        "configs/ctc_config.yaml"
        if args.decoder == "ctc"
        else "configs/transformer_config.yaml"
    )

    aml_command = command(
        code=".",
        command=(
            f"python {train_script} "
            f"--config {config_file} "
            f"--language {args.language} "
            f"--epochs {args.epochs} "
            f"--batch_size {args.batch_size} "
            "--checkpoint_dir ./outputs/checkpoints "
            "--log_dir ./outputs/tensorboard"
        ),
        environment="AzureML-pytorch-2.0-ubuntu20.04-py38-cuda11-gpu@latest",
        compute=args.compute_name,
        display_name=f"asr-{args.decoder}-{args.language}",
        description=(
            f"Train {args.decoder.upper()} ASR model for language '{args.language}'"
        ),
    )

    returned_job = ml_client.jobs.create_or_update(aml_command)
    print(f"Job submitted: {returned_job.name}")
    print(f"Studio URL:    {returned_job.studio_url}")


if __name__ == "__main__":
    main()
