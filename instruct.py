import argparse
import os

from instructlab.training import (
    run_training,
    TorchrunArgs,
    TrainingArgs
)

outputs_dir = "workspace/data/outputs"
checkpoints_dir = "workspace/data/saved_checkpoints"


def list_files_in_directory(directory):
    try:
        files = os.listdir(directory)
        return files
    except Exception as e:
        return str(e)


def main():
    parser = argparse.ArgumentParser(
        prog='instruct',
        description='Demo tool for InstructLab training library'
    )
    parser.add_argument('-f', '--file', type=str, required=True, help='Filepath to a training dataset')
    args = parser.parse_args()

    rank = int(os.environ.get("RANK", 0))
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    master_port = int(os.environ.get("MASTER_PORT", 23456))
    nnodes = int(os.environ.get("PET_NNODES", 1))

    # define training-specific arguments
    training_args = TrainingArgs(
        model_path="ibm-granite/granite-7b-base",
        data_path=args.file,
        ckpt_output_dir=checkpoints_dir,
        data_output_dir=outputs_dir,
        max_seq_len=1024,
        max_batch_len=2048,
        num_epochs=1,
        effective_batch_size=100,
        save_samples=1000,
        learning_rate=2e-6,
        warmup_steps=800,
        is_padding_free=True,
        random_seed=42,
    )

    torchrun_args = TorchrunArgs(
        nnodes=nnodes,
        nproc_per_node=1,
        node_rank=rank,
        rdzv_id=123,
        rdzv_endpoint=f'{master_addr}:{master_port}'
    )

    run_training(
        torch_args=torchrun_args,
        train_args=training_args,
    )

    # Print post-training messages and list folder contents
    print("Training completed successfully!")
    print(f"\nContents of {outputs_dir}:")
    output_files = list_files_in_directory(outputs_dir)
    if isinstance(output_files, str):
        print(f"Error listing files in {outputs_dir}: {output_files}")
    else:
        for file in output_files:
            print(file)

    print(f"\nContents of {checkpoints_dir}:")
    checkpoint_files = list_files_in_directory(checkpoints_dir)
    if isinstance(checkpoint_files, str):
        print(f"Error listing files in {checkpoints_dir}: {checkpoint_files}")
    else:
        for file in checkpoint_files:
            print(file)


if __name__ == '__main__':
    main()